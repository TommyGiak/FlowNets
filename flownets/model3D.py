# hybrid_unet_cfm_3d.py
# 3D Hybrid U-Net + Transformer (bottleneck) for Conditional Flow Matching (CFM)
# Works in latent space (3D volumes). Includes spatial cross-attention for masks
# and global conditioning for clinical embeddings. Training loop skeleton included.

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------- Utilities ---------------------------------
# Sinusoidal time embedding
class SinusoidalPosEmb(nn.Module):
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
        freqs = torch.exp(torch.arange(self.half, device=device, dtype=torch.float32) * -(math.log(100.0) / self.half))
        args = t.view(-1,1) * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb
      

class TimeSequential(nn.Sequential):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass
      
    def forward(self, input, t_emb=None):
        for module in self:
            input = module(input, t_emb)
        return input


# --------------------------- 3D building blocks -------------------------
class Conv3dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1, use_norm=True, num_groups=8):
        super().__init__()
        self.seq = nn.Sequential()
        if use_norm:
          if num_groups>out_ch:
            num_groups=out_ch
          assert out_ch%num_groups==0, f'In the Conv3dBlock, the number of output channels ({out_ch}) must be a multiple of the number of groups ({num_groups})!'
          self.seq.append(nn.Conv3d(in_ch, out_ch, kernel, padding=padding, bias=False))
          self.seq.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))
        else:
          self.seq.append(nn.Conv3d(in_ch, out_ch, kernel, padding=padding))
        self.seq.append(nn.SiLU())

    def forward(self, x):
        return self.seq(x)


class Residual3dBlock(nn.Module):
    def __init__(self, ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = Conv3dBlock(ch, ch)
        self.time_proj = nn.Linear(time_emb_dim, ch) if time_emb_dim is not None else None
        self.conv2 = Conv3dBlock(ch, ch)

    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        if self.time_proj is not None and t_emb is not None:
            # broadcast time embed -> (B, C, 1,1,1)
            proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            h = h + proj
        elif self.time_proj is not None and t_emb is None:
          raise RuntimeError('In the Module is defined a time projection embetting layer but no time embedding was passed in the forward!')
        h = self.conv2(h)
        return x + h


# Simple Downsample/Upsample
class Downsample3d(nn.Module):
    def __init__(self, in_ch, out_ch=None, conv_layer=True):
        super().__init__()
        if conv_layer:
          out_ch = out_ch or in_ch*2
          self.op = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        else:
          if out_ch is not None:
              raise ValueError(f"If you don't use a conv layer the output channels must be the same of the input channels, not {out_ch}")
          self.op = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class Upsample3d(nn.Module):
    def __init__(self, in_ch, out_ch=None, conv_layer=True, upsample_mode='nearest'):
        super().__init__()
        if conv_layer:
          out_ch = out_ch or in_ch//2
          self.op = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
          if out_ch is not None:
              raise ValueError(f"If you don't use a conv layer the output channels must be the same of the input channels, not {out_ch}")
          self.op = nn.Upsample(scale_factor=2, mode=upsample_mode)

    def forward(self, x):
        return self.op(x)


# ----------------------- Cross-Attention modules -----------------------
# A simple tokenized cross-attention where spatial features are flattened
# to tokens and attend to condition tokens (mask tokens or clinical tokens).

class CrossAttention(nn.Module):
  def __init__(self, dim, cond_dim, heads=8, dropout=0.0, bias=False):
    super().__init__()
    if (dim % heads != 0) or (cond_dim % heads != 0):
      raise ValueError(f'The dimensions of the embeddings ({dim} and {cond_dim}) in CrossAttention must be multiples of the number of heads ({heads})!')
    self.mha = nn.MultiheadAttention(num_heads=heads, embed_dim=dim, kdim=cond_dim, vdim=cond_dim,
                                     dropout=dropout, bias=bias, add_zero_attn=False, batch_first=True)

  def forward(self, x, cond):
    # return only the attention output; residual connection is applied in the transformer block
    attn_out, _ = self.mha(x, cond, cond, need_weights=False)
    return attn_out


# ------------------------- Transformer (bottleneck) -------------------
class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        warnings.warn('In SimpleTransformerBlock LayerNorm are used, not GroupNorm, but why? IDK maybe we are carzy')
        warnings.warn('The attention mechanism is implemented with torch, you may use flash attention... Flash attention is wanderful!')
        warnings.warn('The attention transformer block do not use ADAPTIVE layer/group norm... We do not like mutants')

    def forward(self, x):
        # x: (B, N, dim)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim, heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, cond_dim, heads=heads, dropout=dropout, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond_tokens):
        # cross-attention with residual
        h = self.norm1(x)
        attn_out = self.cross_attn(h, cond_tokens)
        x += attn_out
        # feed-forward with its own normalization and residual
        x += self.mlp(self.norm2(x))
        return x


# --------------------------- Hybrid U-Net CFM --------------------------
class HybridUNetCFM3D(nn.Module):
    def __init__(self,
                 img_size : tuple, # 3d tuple (Axial slices, Coronal slices, Saggital slices)
                 in_channels=1,
                 channels_per_down=[8,16,32,64],
                 n_residuals_blocks=1,
                 time_emb_dim=256,
                 cond_global_dim=0,
                 attn_patch_size=2,
                 num_bottleneck_blocks=2):
      
        super().__init__()
        self.img_size = img_size #! not really needed!!!
        self.patch_size = attn_patch_size
        self.in_channels = in_channels
        self.chs = channels_per_down
        self.n_residuals_blocks = n_residuals_blocks
        
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())

        # Encoder
        self.res_blocks_encoder = nn.ModuleList(
                            TimeSequential(*[Residual3dBlock(ch=c, time_emb_dim=time_emb_dim)
                                              for _ in range(n_residuals_blocks)])
                            for c in [in_channels] + self.chs[:-1]
                          )
            
        self.downs = nn.ModuleList([Downsample3d(in_ch=ic, out_ch=oc) for ic, oc in zip([in_channels]+self.chs[:-1], self.chs)])
        
        assert len(self.res_blocks_encoder) == len(self.downs)
        
        # Bottleneck 
        bottleneck_ch = self.chs[-1]
        self.bottleneck_res = Residual3dBlock(bottleneck_ch, time_emb_dim)
        self.transformer_blocks = nn.ModuleList([
            CrossTransformerBlock(bottleneck_ch*(self.patch_size**3), cond_global_dim) if cond_global_dim > 0 else SimpleTransformerBlock(bottleneck_ch*(self.patch_size**3))
            for _ in range(num_bottleneck_blocks)
        ])

        # Up path
        self.ups = nn.ModuleList([Upsample3d(in_ch=ic, out_ch=oc) for ic, oc in zip(reversed(self.chs), reversed([in_channels]+self.chs[:-1]))])
        self.conv_skip_connection_decored = nn.ModuleList([Conv3dBlock(in_ch=oc*2, out_ch=oc) for oc in reversed([in_channels]+self.chs[:-1])])
        self.res_blocks_decoder = nn.ModuleList(
                    TimeSequential(*[Residual3dBlock(ch=c, time_emb_dim=time_emb_dim)
                                        for _ in range(n_residuals_blocks)])
                    for c in reversed([in_channels] + self.chs[:-1])
                ) # per ora va ben così, ma poi magari va aggiunto un pos encoding per accettare anche diverse image sizes...!!! Molta fatica sta maledetta attenzione
        
        assert len(self.res_blocks_decoder) == len(self.ups) == len(self.conv_skip_connection_decored)

        pass

    def patching_and_tokenizer(self, layer, channels, current_img_size):
        B, C, Z, H, W = layer.shape
        assert (C,Z,H,W) == (channels, *current_img_size) # TO REMOVE
        p = self.patch_size

        layer = layer.unfold(2,p,p).unfold(3,p,p).unfold(4,p,p)
        layer = layer.reshape(B, C, -1, p, p, p).transpose(1,2).flatten(start_dim=2).contiguous() # Ci ho perso tipo 45 minuti per arrivare a questo cazzo di unfolding quindi abbiatene cura (portrebbe non funzionare comunque maledizione)
        return layer
    
    def tokens_to_patches(self, tokens, channels, current_img_size):
        B, N, D = tokens.shape
        Z, H, W = current_img_size
        assert N*D == math.prod((channels, *current_img_size)) # TO REMOVE
        p = self.patch_size

        tokens = tokens.unflatten(-1, (channels, p, p, p)).transpose(1, 2)  # (B,3,64,2,2,2)
        tokens = tokens.view(B, channels, Z//p, H//p, W//p, p, p, p).permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(B, channels, *current_img_size)
        assert tokens.shape == (B, channels, Z, H, W) # TO REMOVE
        return tokens
    
    def get_parameters_number(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, t, z_t, cond_global: Optional[torch.Tensor] = None):
        # z_t: (B, C, Z, H, W)
        t_emb = self.time_mlp(t)

        # Encoder
        skips = []
        x = z_t
        for i, (res, down) in enumerate(zip(self.res_blocks_encoder, self.downs)):
            x = res(x, t_emb)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck_res(x, t_emb)

        # tokens
        B, C, Z, H, W = x.shape
        x = self.patching_and_tokenizer(layer=x, channels=C, current_img_size=(Z, H, W)) # (B, N, D), N = (Z//p)*(H//p)*(W//p), D = C*(p**3)

        # prepare global cond tokens if provided
        cond_tokens = None
        if cond_global is not None:
            # cond_global: (B, cond_dim) -> expand to tokens M=1
            cond_tokens = cond_global.unsqueeze(1)  # (B, 1, cond_dim)
        for tb in self.transformer_blocks:
            if isinstance(tb, CrossTransformerBlock) and cond_tokens is not None:
                x = tb(x, cond_tokens)
            else:
                x = tb(x)
        x = self.tokens_to_patches(x, channels=C, current_img_size=(Z, H, W))
        
        # Decoder
        for up, conv_skip, res in zip(self.ups, self.conv_skip_connection_decored, self.res_blocks_decoder):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)  # channel concat
            x = conv_skip(x)
            x = res(x, t_emb)

        return x

