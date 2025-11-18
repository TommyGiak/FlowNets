"""
@author: Tommaso Giacometti
"""

import math
import torch
import warnings

from torch import nn
from einops import rearrange


# --------------------------- Layer dynamic dimensionality -------------------------
def convolution(image_dimensionality, *args, **kwargs):
    if image_dimensionality == 1:
        return nn.Conv1d(*args, **kwargs)
    elif image_dimensionality == 2:
        return nn.Conv2d(*args, **kwargs)
    elif image_dimensionality == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f'The dimenstionality of the images must be 2 or 3, not {image_dimensionality}!!! :(')

def convolution_transpose(image_dimensionality, *args, **kwargs):
    if image_dimensionality == 1:
        return nn.ConvTranspose1d(*args, **kwargs)
    elif image_dimensionality == 2:
        return nn.ConvTranspose2d(*args, **kwargs)
    elif image_dimensionality == 3:
        return nn.ConvTranspose3d(*args, **kwargs)
    else:
        raise ValueError(f'The dimenstionality of the images must be 2 or 3, not {image_dimensionality}!!! :(')
    
def maxpool(image_dimensionality, *args, **kwargs):
    if image_dimensionality == 1:
        return nn.MaxPool1d(*args, **kwargs)
    elif image_dimensionality == 2:
        return nn.MaxPool2d(*args, **kwargs)
    elif image_dimensionality == 3:
        return nn.MaxPool3d(*args, **kwargs)
    else:
        raise ValueError(f'The dimenstionality of the images must be 2 or 3, not {image_dimensionality}!!! :(')

def zero_init_(module):
    with torch.no_grad():
        for p in module.parameters():
            if p is not None:
                p.zero_()
    return module


# ---------------------------  Embeddings -------------------------
# Sinusoidal time embedding
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half = self.dim // 2

        # clearer check: dimension must be even
        assert self.dim % 2 == 0, f'The dimension of the Positional embedding must be a multiple of 2, not {dim}!'

    def forward(self, t):
        # t: (B,) floats in [0,1]
        device = t.device
        # use 'half' in the denominator so we don't divide by zero when half == 1
        freqs = torch.exp(torch.arange(0, self.half, device=device, dtype=torch.float32) * -(math.log(10000.0) / self.half))
        args = t.view(-1,1) * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb
    

class PositionalEncodingND(nn.Module):
    def __init__(self, image_size, dim, patch_size):
        super().__init__()
        assert dim % (2 * len(image_size)) == 0, \
               "dim must be divisible by 2 * spatial_dims"

        self.spatial_dims = len(image_size)
        self.dim = dim
        self.patch_size = patch_size
        
        grid_shape = [s // patch_size for s in image_size]
        
        # Build grid: e.g. for 2D (H', W'), for 3D (Z', H', W')
        coords = [torch.arange(n, dtype=torch.float32) for n in grid_shape]
        mesh = torch.meshgrid(*coords, indexing='ij')
        
        # flatten: (N,)
        flattened = [m.reshape(-1) for m in mesh]  # list of dim components
        
        pe = self.build_pe(flattened)  # (N, dim)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, N, dim)

    def build_pe(self, coords):

        ax = len(coords)
        dim_per_axis = self.dim // ax
        half = dim_per_axis // 2

        embeddings = []
        for c in coords:
            div = torch.exp(torch.arange(half).float() * (-math.log(10000.0) / half))
            out = torch.zeros(c.shape[0], dim_per_axis)
            out[:, 0:half] = torch.sin(c.unsqueeze(1) * div)
            out[:, half:] = torch.cos(c.unsqueeze(1) * div)
            embeddings.append(out)
        
        return torch.cat(embeddings, dim=1)

    def forward(self, x):
        # x = (B, N, dim)
        return x + self.pe


class TimeSequential(nn.Sequential):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass
      
    def forward(self, input, t_emb=None):
        for module in self:
            input = module(input, t_emb)
        return input


# --------------------------- Building blocks -------------------------
class ConvBlock(nn.Module):
    def __init__(self, image_dimensionality, in_ch, out_ch, kernel_size=3, padding=1, use_norm=True, num_groups=16, dropout=0, zero_init=False):
        super().__init__()
        self.seq = nn.Sequential()
        
        if use_norm:
            
            num_groups = out_ch if num_groups>out_ch else num_groups

            assert out_ch%num_groups==0, f'In the Conv2dBlock, the number of output channels ({out_ch}) must be a multiple of the number of groups ({num_groups})!'
            if zero_init:
                self.seq.append(zero_init_(convolution(image_dimensionality, in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False)))
            else:
                self.seq.append(convolution(image_dimensionality, in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False))
            self.seq.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))
        else:
            if zero_init:
                self.seq.append(zero_init_(convolution(image_dimensionality, in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)))
            else:
                self.seq.append(convolution(image_dimensionality, in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True))
        
        self.seq.append(nn.SiLU())
        
        if dropout > 0:
            self.seq.append(nn.Dropout(p=dropout))

    def forward(self, x):
        return self.seq(x)


class ResidualBlock(nn.Module):
    def __init__(self, image_dimensionality, ch, time_emb_dim=None, num_groups=16, dropout=0):
        super().__init__()
        self.img_dim = image_dimensionality
        self.conv1 = ConvBlock(image_dimensionality, ch, ch, num_groups=num_groups, dropout=dropout)
        
        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, ch)
        else:
            self.time_proj = None
        
        self.conv2 = ConvBlock(image_dimensionality, ch, ch, num_groups=num_groups, dropout=dropout, zero_init=True)

    def forward(self, x, t_emb=None):
        h = self.conv1(x)

        if self.time_proj is not None:
            if t_emb is None:
                raise RuntimeError("time embedding expected but not provided")
            proj = self.time_proj(t_emb)  # [B, ch]
            # reshape to [B, ch, 1, 1, ...]
            shape = [proj.size(0), proj.size(1)] + [1] * (h.dim() - 2)
            proj = proj.view(*shape)
            h = h + proj

        h = self.conv2(h)
        return x + h


# --------------------------- Resizing blocks -------------------------
# Simple Downsample/Upsample
class Downsample(nn.Module):
    def __init__(self, image_dimensionality, in_ch, out_ch=None, conv_layer=True, num_groups=8):
        super().__init__()
        self.op = nn.Sequential()
        if conv_layer:
            out_ch = out_ch or in_ch*2
            self.op.append(convolution(image_dimensionality, in_ch, out_ch, kernel_size=3, stride=2, padding=1))
            if num_groups > 0:
                self.op.append(nn.GroupNorm(num_groups, out_ch))
        else:
            if out_ch is not None:
                raise ValueError(f"If you don't use a conv layer the output channels must be the same of the input channels, not {out_ch}")
            self.op.append(maxpool(image_dimensionality, kernel_size=2, stride=2))

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, image_dimensionality, in_ch, out_ch=None, conv_layer=True, num_groups=16, upsample_mode='nearest'):
        super().__init__()
        
        self.op = nn.Sequential()
        
        if conv_layer:
            out_ch = out_ch or in_ch//2
            
            num_groups = out_ch if num_groups>out_ch else num_groups
            
            self.op.append(convolution_transpose(image_dimensionality, in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            if num_groups:
                self.op.append(nn.GroupNorm(num_groups, out_ch))
        else:
          if out_ch is not None:
                raise ValueError(f"If you don't use a conv layer the output channels must be the same of the input channels, not {out_ch}")
          self.op.append(nn.Upsample(scale_factor=2, mode=upsample_mode))

    def forward(self, x):
        return self.op(x)


# ------------------------- Tokenizer -------------------
class Tokenizer(nn.Module):
    def __init__(self, image_size, patch_size):
        super().__init__()
        self.p = patch_size
        self.image_size = image_size

        for d in image_size:
            assert d%patch_size==0, f'The image size at this layer {image_size} must be a multiple of the patch size {patch_size}! :=('
        
        if len(image_size) == 2:
            self.fancy_reshape = lambda x: rearrange(
                x,
                'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                p1=self.p, p2=self.p
            )
            self.invert_fancy_reshape = lambda tokens: rearrange(
                tokens,
                "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                h=image_size[0]//self.p, w=image_size[1]//self.p, 
                p1=self.p, p2=self.p
            )
        elif len(image_size) == 3:
            self.fancy_reshape = lambda x: rearrange(
                x,
                'b c (z p1) (h p2) (w p3) -> b (z h w) (c p1 p2 p3)',
                p1=self.p, p2=self.p, p3=self.p
            )
            self.invert_fancy_reshape = lambda tokens: rearrange(
                tokens,
                "b (z h w) (c p1 p2 p3) -> b c (z p1) (h p2) (w p3)",
                z=image_size[0]//self.p, h=image_size[1]//self.p, w=image_size[2]//self.p, 
                p1=self.p, p2=self.p, p3=self.p
            )
        else:
            raise ValueError(f'The image dimensionality must be 2 or 3, not {len(image_size)}')
        pass

    def tokenization(self, x):
        # x: (B, C, *IMG_SIZE)
        return self.fancy_reshape(x)
    
    def invert_tokenization(self,tokens):
        return self.invert_fancy_reshape(tokens)


# ------------------------- Self-Attention -------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, image_size, dim, patch_size, heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.tokenizer = Tokenizer(image_size=image_size, patch_size=patch_size)
        patch_volume = patch_size**len(image_size)
        self.tokens_projection = nn.Linear(channels*patch_volume, dim)
        self.pos_enc = PositionalEncodingND(image_size, dim, patch_size)

        self.norm1 = nn.RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
            )
        
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.RMSNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.out_reshape = nn.Linear(dim, channels*patch_volume)
        
        warnings.warn('The attention mechanism is implemented with torch, you may use flash attention... Flash attention is wanderful!')
        warnings.warn('The attention transformer block do not use ADAPTIVE layer/group norm... We do not like mutants')

    def forward(self, x):
        # x: (B, C, *IMG_SIZE)
        x = self.tokenizer.tokenization(x) # x: (B, N, dim)
        x = self.tokens_projection(x)
        x = self.pos_enc(x)
        h = self.norm1(x)
        
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        attn_out = self.attn_dropout(attn_out)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x)) # x: (B, N, dim)
        x = self.out_reshape(x)
        return self.tokenizer.invert_tokenization(x)


# ----------------------- Cross-Attention modules -----------------------
class CrossAttention(nn.Module):
  def __init__(self, q_dim, k_dim=None, v_dim=None, heads=8, dropout=0.0, bias=False):
    super().__init__()
    
    if (k_dim is None) and (v_dim is None):
        k_dim = q_dim
        v_dim = q_dim
    elif (k_dim is None) or (v_dim is None):
        raise ValueError(f'Mmmm something suspicious happened, k_dim ({k_dim}) and v_dim ({v_dim}) must be equal...')
    
    if (q_dim % heads != 0) or (k_dim % heads != 0) :
      raise ValueError(f'The dimensions of the embeddings ({q_dim} and {k_dim}) in CrossAttention must be multiples of the number of heads ({heads})!')
    self.mha = nn.MultiheadAttention(num_heads=heads, embed_dim=q_dim, kdim=k_dim, vdim=v_dim,
                                     dropout=dropout, bias=bias, add_zero_attn=False, batch_first=True)

  def forward(self, x, cond):
    # return only the attention output; residual connection is applied in the transformer block
    attn_out, _ = self.mha(x, cond, cond, need_weights=False)
    return attn_out



