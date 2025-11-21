"""
@author: Tommaso Giacometti
"""

import math
import torch
import warnings

from torch import nn
import torch.nn.functional as F

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

def get_num_groups_for_channels(channels: int, default_groups: int = 32):

    max_g = min(default_groups, channels)

    for g in range(max_g, 0, -1):
        if channels % g == 0:
            return g

    return 1

class Identity(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, *args):
        return input


# ---------------------------  Embeddings -------------------------
class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Time embedding dim must be even."
        
        self.dim = dim
        half = dim // 2

        # Precompute frequencies (CPU is fine)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
        )

        # Register as buffer so it automatically moves to the model's device
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        # t : (B,) or (B,1)
        t = t.view(-1, 1)                         # (B,1)
        args = t * self.freqs[None, :]            # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb                                # (B, dim)

    

class PositionalEncodingND(nn.Module):
    def __init__(self, image_size, dim, patch_size):
        super().__init__()
        assert dim % (2 * len(image_size)) == 0, 'dim must be divisible by 2 * spatial_dims'

        self.spatial_dims = len(image_size)
        self.dim = dim
        self.patch_size = patch_size
        
        grid_shape = [s // patch_size for s in image_size]
        
        # Build grid: for 2D (H', W'), for 3D (Z', H', W')
        coords = [torch.arange(n, dtype=torch.float32) for n in grid_shape]
        mesh = torch.meshgrid(*coords, indexing='ij')
        
        # flatten: (N,)
        flattened = [m.reshape(-1) for m in mesh]  # list of dim components
        
        pe = self.build_pe(flattened)  # (N, dim)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, N, dim)

    def build_pe(self, coords):
        D = len(coords)
        dim_per_axis = self.dim // D
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
class ResidualBlock(nn.Module):
    def __init__(self, image_dimensionality, ch, time_emb_dim=None, num_groups=32, dropout=0):
        super().__init__()
        self.img_dim = image_dimensionality

        reshape_view = (1 for _ in range(image_dimensionality))
        self.reshape_view = (ch, *reshape_view)

        num_groups = get_num_groups_for_channels(ch, num_groups)

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups,ch),
            nn.SiLU(),
            convolution(image_dimensionality, ch, ch, kernel_size=3, padding=1),
        )

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, ch)
        else:
            self.time_proj = None
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups,ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_init_(convolution(image_dimensionality, ch, ch, kernel_size=3, padding=1))
        )
        pass

    def forward(self, x, t_emb=None):
        h = self.in_layers(x)

        if self.time_proj is not None:
            proj = self.time_proj(t_emb)  # [B, ch]
            # reshape to [B, ch, 1, 1, ...]
            proj = proj.view(proj.shape[0], *self.reshape_view)
            h = h + proj

        h = self.out_layers(h)
        return x + h


#------------------------------------------------------------------------------------------------------------------------------------------------------------
class SelfConvAttentionBlock(nn.Module):
    """An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        image_size,
        dim,
        patch_size,
        heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        dropout = 0.,
    ):
        super().__init__()
        self.channels = channels
        num_heads = heads
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(get_num_groups_for_channels(channels),channels)
        self.qkv = convolution(1, channels, channels * 3, 1)
        # split qkv before split heads
        self.attention = QKVAttention(self.num_heads)

        self.proj_out = zero_init_(convolution(1, channels, channels, 1))

    def forward(self, x, emb):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)
    

class QKVAttention(nn.Module):
    """A module which performs QKV attention and splits in a different order."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

#------------------------------------------------------------------------------------------------------------------------------------------------------------
    


# --------------------------- Resizing blocks -------------------------
# Simple Downsample/Upsample
class Downsample(nn.Module):
    def __init__(self, image_dimensionality, in_ch, out_ch=None, conv_layer=True, num_groups=16):
        super().__init__()
        self.op = nn.Sequential()
        if conv_layer:
            out_ch = out_ch or in_ch*2
            num_groups = get_num_groups_for_channels(in_ch, num_groups)
            if num_groups > 0:
                self.op.append(nn.GroupNorm(num_groups, in_ch))
            self.op.append(nn.SiLU())
            self.op.append(convolution(image_dimensionality, in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False))
            if num_groups > 0:
                self.op.append(nn.GroupNorm(num_groups, in_ch))
            self.op.append(nn.SiLU())
            self.op.append(convolution(image_dimensionality, in_ch, out_ch, kernel_size=3, stride=2, padding=1))
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
            
            num_groups = get_num_groups_for_channels(in_ch,num_groups)

            if num_groups:
                self.op.append(nn.GroupNorm(num_groups, in_ch))
            self.op.append(nn.SiLU())
            self.op.append(convolution(image_dimensionality, in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False))
            if num_groups:
                self.op.append(nn.GroupNorm(num_groups, in_ch))
            self.op.append(nn.SiLU())
            self.op.append(convolution_transpose(image_dimensionality, in_ch, out_ch, kernel_size=4, stride=2, padding=1))

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
            assert d%patch_size==0, f'Image dimension {d} must be divisible by patch size {patch_size}'

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
    

# ------------------------- adaLNZero -------------------
class AdaLNZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 6 * dim)

        zero_init_(self.linear)

    def forward(self, emb):
        params = self.linear(emb) # (B, 6*dim)
        params = params.unsqueeze(1) # (B,1,6*dim)
        sA, bA, gA, sM, bM, gM = params.chunk(6, dim=-1)
        return sA, bA, gA, sM, bM, gM



# ------------------------- Self-Attention -------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, image_size, dim, patch_size, heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.dropout = dropout

        self.tokenizer = Tokenizer(image_size, patch_size)
        patch_volume = patch_size**len(image_size)
        
        self.tokens_projection = nn.Linear(channels*patch_volume, dim)
        
        self.pos_enc = PositionalEncodingND(image_size, dim, patch_size)

        self.adaln = AdaLNZero(dim)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

        #-------------- Attention --------------
        self.heads = heads
        assert dim % heads == 0, f'dimension of th embedding {dim} must be divisible by number of heads {heads}'
        self.head_dim = dim // heads

        #self.qkv = nn.Linear(dim, 3 * dim)
        self.qkv = convolution(1, dim, 3 * dim, 1)

        '''self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
        )'''

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        
        self.out_proj = nn.Linear(dim, channels*patch_volume)

    def _split_heads(self, x):
        # x: (B, N, dim) -> (B, heads, N, head_dim)
        B, N, D = x.shape
        x = x.view(B, N, self.heads, self.head_dim)   # (B, N, H, Hd)
        x = x.permute(0, 2, 1, 3).contiguous()        # (B, H, N, Hd)
        return x
    
    def _combine_heads(self, x):
        # x: (B, heads, N, head_dim) -> (B, N, dim)
        B, H, N, Hd = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()        # (B, N, H, Hd)
        x = x.view(B, N, H * Hd)
        return x


    def forward(self, x, emb):
        # x: (B, C, *IMG_SIZE)
        t = self.tokenizer.tokenization(x)
        t = self.tokens_projection(t) # x: (B, N, dim)
        t = self.pos_enc(t)

        sA, bA, gA, sM, bM, gM = self.adaln(emb)
        
        h = self.norm1(t) * (1 + sA) + bA
        
        q, k, v = self.qkv(h.permute(0,2,1)).permute(0,2,1).chunk(3, dim=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v) # (B, H, N, Hd)
        h_att = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                  dropout_p=self.dropout,
                                                  is_causal=False)  # (B, H, N, Hd)
        h_att = self._combine_heads(h_att)
        # h_att, _ = self.attn(h, h, h, need_weights=False)
        t = t + gA * h_att

        h = self.norm2(t) * (1 + sM) + bM
        h = self.mlp(h)
        t = t + gM * h

        t = self.out_proj(t)
        return self.tokenizer.invert_tokenization(t)


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

