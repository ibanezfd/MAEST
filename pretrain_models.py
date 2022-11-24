import torch
import torch.nn as nn

from einops import rearrange, repeat
from utils import get_sinusoid_encoding_table
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
#-------------------------------------------------------------------------------
class Residual(nn.Module): 
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
#-------------------------------------------------------------------------------
class PreNorm(nn.Module):   
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
#-------------------------------------------------------------------------------
class FeedForward(nn.Module):  
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
#-------------------------------------------------------------------------------
class Attention(nn.Module): 
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
#-------------------------------------------------------------------------------
class Transformer(nn.Module):  
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth-2):   
            self.skipcat.append(nn.Conv2d(num_channel+1, num_channel+1, [1, 2], 1, 0))

    def forward(self, x, mask = None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:  
                    x = self.skipcat[nl-2](torch.cat([x.unsqueeze(3), last_output[nl-2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x, mask = mask)
                x = ff(x)
                nl += 1
        return x
#-------------------------------------------------------------------------------
class VisionTransformerEncoder(nn.Module): 

    def __init__(self,
                 image_size,  
                 near_band,   
                 num_patches, 
                 num_classes, 
                 dim,    
                 depth,  
                 heads,   
                 mlp_dim, 
                 pool='cls',    
                 channels=1,
                 dim_head = 16, 
                 dropout=0.,
                 emb_dropout=0.,
                 mode='ViT',    
                 mask_ratio=0.75,
                 init_scaler=0.
                 ):
        super().__init__()
        patch_dim = image_size ** 2 * near_band  
        self.use_cls = True   
        self.num_classes = num_classes
        self.num_patches = num_patches
        if self.use_cls:
            patch_cls = 1
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) 
        else:
            patch_cls = 0
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + patch_cls, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim) 
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim, num_classes)) if num_classes > 0 else nn.Identity()  

    def forward_features(self, x):
        x = self.patch_to_embedding(x) 
        b, n, c = x.shape
        
        if self.use_cls:   
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) 
            x = torch.cat((cls_tokens, x), dim = 1) 
            x += self.pos_embedding[:, :(n + 1)]
            x = self.dropout(x)
        else:
            x = self.dropout(x + self.pos_embedding)
        return x

    def forward(self, x, masking_pos, mask = None):
        x = self.forward_features(x)    
        b, _, c = x.shape

        if self.num_classes == 0:   
            if self.use_cls:   
                cls_em_tokens = x[:,0:1,:]
                x_vis = x
                x_vis = x[:,1:,:]
                x = x_vis[~masking_pos].reshape(b, -1, c)
                x = torch.cat((cls_em_tokens, x), dim = 1)
            else:
                x = x[~masking_pos].reshape(b, -1, c)

        x = self.transformer(x, mask)  

        if self.num_classes > 0:   
            x = x.mean(axis=1) if self.pool == 'mean' else self.to_latent(x[:, 0])
        elif (self.use_cls and self.num_classes == 0):
            x = x[:,1:]
        else:   
            x = self.to_latent(x)
        return self.mlp_head(x)
#-------------------------------------------------------------------------------
class VisionTransformerDecoder(nn.Module):

    def __init__(self,
                 image_size=1,
                 near_band=1,
                 num_patches=200,
                 num_classes=147,
                 dim=128,
                 depth=5,
                 dim_head = 16,
                 heads=4,
                 mlp_dim=8,
                 pool='cls',
                 dropout=0.1,
                 emb_dropout=0.1,
                 mode='ViT'
                 ):
        super().__init__()

        self.num_classes = num_classes
        assert num_classes == near_band * image_size ** 2
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, dim, mode)
        self.norm =  nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num, mask=None):

        x = self.transformer(x, mask)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) 
        else:
            x = self.head(self.norm(x)) 
        return x
#-------------------------------------------------------------------------------
class PretrainVisionTransformer(nn.Module):

    def __init__(self,
                 image_size=1,
                 near_band=1,
                 num_patches=200,
                 encoder_num_classes=0,
                 encoder_dim=147,
                 encoder_depth=5,
                 encoder_heads=4,
                 encoder_dim_head=16,
                 encoder_mode='ViT',
                 encoder_pool='cls',
                 decoder_num_classes=147,
                 decoder_dim=128,
                 decoder_depth=3,
                 decoder_heads=3,
                 decoder_dim_head=12,
                 decoder_mode='ViT',
                 decoder_pool='cls',
                 mlp_dim=8,
                 dropout=0.1,
                 emb_dropout=0.1,
                 mask_ratio=0.75
                 ):

        super().__init__()
        self.encoder = VisionTransformerEncoder(
            image_size=image_size,
            near_band=near_band,
            num_patches=num_patches,
            num_classes=encoder_num_classes,
            dim=encoder_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            mlp_dim=mlp_dim,
            pool=encoder_pool,
            dim_head = encoder_dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            mode=encoder_mode,
            mask_ratio=mask_ratio)

        self.decoder = VisionTransformerDecoder(
            image_size=image_size,
            near_band=near_band,
            num_patches=num_patches,
            num_classes=decoder_num_classes,
            dim=decoder_dim,
            depth=decoder_depth,
            dim_head = decoder_dim_head,
            heads=decoder_heads,
            mlp_dim=mlp_dim,
            pool=decoder_pool,
            dropout=dropout,
            emb_dropout=emb_dropout,
            mode=decoder_mode)


        self.learn_pos = False
        if self.learn_pos == True:
            self.pos_emb = nn.Parameter(torch.randn(1, num_patches, decoder_dim))
        else:
            self.pos_emb = get_sinusoid_encoding_table(num_patches, decoder_dim)

        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

    def forward(self, x, masking_pos, mask=None):
        b, _, _ = x.shape
        x_vis = self.encoder(x, masking_pos, mask)
        x_visd = self.encoder_to_decoder(x_vis) 

        _, _, cv = x_visd.shape

        pos_embed = self.pos_emb.expand(b, -1, -1).type_as(x).to(x.device).detach().clone()
        pos_embed_vis = pos_embed[~masking_pos].reshape(b,-1, cv)
        pos_embed_mask = pos_embed[masking_pos].reshape(b,-1, cv)

        x_full = torch.cat([x_visd + pos_embed_vis, self.mask_token + pos_embed_mask],dim=1)
        x = self.decoder(x_full, pos_embed_mask.shape[1]) 
        return x
#-------------------------------------------------------------------------------
class ViT(nn.Module):
    def __init__(self, 
    image_size, 
    near_band, 
    num_patches, 
    num_classes, 
    dim, 
    depth, 
    heads, 
    mlp_dim, 
    pool='cls', 
    channels=1, 
    dim_head = 16, 
    dropout=0., 
    emb_dropout=0., 
    mode='ViT'
    ):
        super().__init__()

        patch_dim = image_size ** 2 * near_band
        self.num_patches = num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    def forward(self, x, masking_pos, mask = None):
        x = self.patch_to_embedding(x) 
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b) 
        x = torch.cat((cls_tokens, x), dim = 1) 
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.to_latent(x[:,0])
        return self.mlp_head(x)
#-------------------------------------------------------------------------------
