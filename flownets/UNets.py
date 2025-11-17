"""
@author: Tommaso Giacometti
"""

import torch

from torch import nn
from .BlocksAndLayers import SinusoidalTimeEmb, TimeSequential, ResidualBlock, Downsample, Upsample, ConvBlock, convolution, SelfAttentionBlock, zero_init_


class SimpleUNet(nn.Module): # without attention
    def __init__(self,
                 img_size : tuple,
                 in_channels=1,
                 channels_per_down=[8,16,32,64,128],
                 n_residuals_blocks=1,
                 time_emb_dim=256,
                 dropout=0,
                 ):
      
        super().__init__()
        assert (len(img_size) == 2) or (len(img_size) == 3), f'Image dimensionality must be 2 or 3, not {len(img_size)}'
        self.image_dimensionality = len(img_size)
        d = self.image_dimensionality
        self.img_size = img_size
        self.in_channels = in_channels
        self.chs = channels_per_down
        self.n_residuals_blocks = n_residuals_blocks
        
        self.first_num_groups = 4 if self.chs[0]>=4 else self.chs[0]
        
        self.time_mlp = nn.Sequential(SinusoidalTimeEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())
        
        self.in_res = nn.Sequential(
            convolution(d, in_channels=in_channels, out_channels=self.chs[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.first_num_groups,self.chs[0]),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

        # Encoder
        self.res_blocks_encoder = nn.ModuleList(
                            TimeSequential(*[ResidualBlock(d, ch=c, time_emb_dim=time_emb_dim)
                                              for _ in range(n_residuals_blocks)])
                            for c in self.chs[:-1]
                            )
            
        self.downs = nn.ModuleList([Downsample(d, in_ch=ic, out_ch=oc) for ic, oc in zip(self.chs[:-1], self.chs[1:])])
        
        assert len(self.res_blocks_encoder) == len(self.downs)
        
        # Bottleneck 
        bottleneck_ch = self.chs[-1]
        self.bottleneck_res = TimeSequential(*[ResidualBlock(d, ch=bottleneck_ch, time_emb_dim=time_emb_dim) for _ in range(n_residuals_blocks)])

        # Up path
        self.ups = nn.ModuleList([Upsample(d, in_ch=ic, out_ch=oc) for ic, oc in zip(reversed(self.chs[1:]), reversed(self.chs[:-1]))])
        self.conv_skip_connection_decored = nn.ModuleList([ConvBlock(d, in_ch=oc*2, out_ch=oc, kernel_size=3, padding=1) for oc in reversed(self.chs[:-1])])
        self.res_blocks_decoder = nn.ModuleList(
                    TimeSequential(*[ResidualBlock(d, ch=c, time_emb_dim=time_emb_dim)
                                        for _ in range(n_residuals_blocks)])
                    for c in reversed(self.chs[:-1])
                )

        assert len(self.res_blocks_decoder) == len(self.ups) == len(self.conv_skip_connection_decored)
        
        self.out_res = nn.Sequential(
            zero_init_(convolution(d, in_channels=self.chs[0], out_channels=in_channels, kernel_size=3, padding=1, bias=True)),
        )

        pass
    
    def get_parameters_number(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, t, z_t):
        # z_t: (B, C, *IMG_SIZE)
        t_emb = self.time_mlp(t)
        
        x = z_t
        x = self.in_res(x)
        
        # Encoder
        skips = []
        for i, (res, down) in enumerate(zip(self.res_blocks_encoder, self.downs)):
            x = res(x, t_emb)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck_res(x, t_emb)
        
        # Decoder
        for up, conv_skip, res in zip(self.ups, self.conv_skip_connection_decored, self.res_blocks_decoder):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = conv_skip(x)
            x = res(x, t_emb)

        return self.out_res(x)


class SelfUNet(nn.Module): # without attention
    def __init__(self,
                 img_size : tuple,
                 in_channels=1,
                 channels_per_down=[8,16,32,64,128], # image size [1,2,4,8,16]**(-1)
                 attn_p_per_down=[None,None,2,None,1],
                 attn_p_per_up=None,
                 n_residuals_blocks=1,
                 time_emb_dim=264, # must be multiple of the image dimensionality!
                 dropout=0,
                 ):
      
        super().__init__()
        assert (len(img_size) == 2) or (len(img_size) == 3), f'Image dimensionality must be 2 or 3, not {len(img_size)}'
        assert len(attn_p_per_down) == len(channels_per_down)
        if attn_p_per_up is not None:
            assert len(attn_p_per_up) == len(channels_per_down)
        else:
            attn_p_per_up = attn_p_per_down
        
        self.attn_p_per_down = attn_p_per_down
        self.attn_p_per_up = attn_p_per_up
        
        if len(img_size) == 2:
            H, W = img_size
            self.sequential_image_shape = [(C,H//(2**(i+1)),W//(2**(i+1))) for i,C in enumerate(channels_per_down[1:])]
            self.sequential_image_shape = [(channels_per_down[0],H,W)] + self.sequential_image_shape
        elif len(img_size) == 3:
            Z, H, W = img_size
            self.sequential_image_shape = [(C,Z//(2**(i+1)),H//(2**(i+1)),W//(2**(i+1))) for i,C in enumerate(channels_per_down[1:])]
            self.sequential_image_shape = [(channels_per_down[0],Z,H,W)] + self.sequential_image_shape        
        
        self.image_dimensionality = len(img_size)
        d = self.image_dimensionality
        self.img_size = img_size
        self.in_channels = in_channels
        self.chs = channels_per_down
        self.n_residuals_blocks = n_residuals_blocks
        
        self.first_num_groups = 4 if self.chs[0]>=4 else self.chs[0]
        
        self.time_mlp = nn.Sequential(SinusoidalTimeEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())
        
        self.in_res = nn.Sequential(
            convolution(d, in_channels=in_channels, out_channels=self.chs[0], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.first_num_groups,self.chs[0]),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

        # Encoder
        self.res_blocks_encoder = nn.ModuleList(
                            TimeSequential(*[ResidualBlock(d, ch=c, time_emb_dim=time_emb_dim)
                                              for _ in range(n_residuals_blocks)])
                            for c in self.chs[:-1]
                            )
            
        self.downs = nn.ModuleList([Downsample(d, in_ch=ic, out_ch=oc) for ic, oc in zip(self.chs[:-1], self.chs[1:])])
        
        self.attn_down = nn.ModuleList()
        for shape, patch in zip(self.sequential_image_shape, attn_p_per_down):
            if patch is not None:
                self.attn_down.append(SelfAttentionBlock(
                    channels=shape[0],
                    image_size=shape[1:],
                    dim=time_emb_dim,
                    patch_size=patch,
                    dropout=dropout,
                ))
            else:
                self.attn_down.append(nn.Identity())
        
        assert len(self.res_blocks_encoder) == len(self.downs) == len(self.attn_down) - 1
        
        # Bottleneck 
        bottleneck_ch = self.chs[-1]
        self.bottleneck_res = TimeSequential(*[ResidualBlock(d, ch=bottleneck_ch, time_emb_dim=time_emb_dim) for _ in range(n_residuals_blocks)])

        # Up path
        self.ups = nn.ModuleList([Upsample(d, in_ch=ic, out_ch=oc) for ic, oc in zip(reversed(self.chs[1:]), reversed(self.chs[:-1]))])
        self.conv_skip_connection_decored = nn.ModuleList([ConvBlock(d, in_ch=oc*2, out_ch=oc, kernel_size=3, padding=1) for oc in reversed(self.chs[:-1])])
        self.res_blocks_decoder = nn.ModuleList(
                    TimeSequential(*[ResidualBlock(d, ch=c, time_emb_dim=time_emb_dim)
                                        for _ in range(n_residuals_blocks)])
                    for c in reversed(self.chs[:-1])
                )
        
        self.attn_up = nn.ModuleList()
        for shape, patch in zip(reversed(self.sequential_image_shape),reversed(attn_p_per_down)):
            if patch is not None:
                self.attn_up.append(SelfAttentionBlock(
                    channels=shape[0],
                    image_size=shape[1:],
                    dim=time_emb_dim,
                    patch_size=patch,
                    dropout=dropout,
                ))
            else:
                self.attn_up.append(nn.Identity())

        assert len(self.res_blocks_decoder) == len(self.ups) == len(self.conv_skip_connection_decored) == len(self.attn_up) - 1
        
        self.out_res = nn.Sequential(
            zero_init_(convolution(d, in_channels=self.chs[0], out_channels=in_channels, kernel_size=3, padding=1, bias=True)),
        )

        pass
    
    def get_parameters_number(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, t, z_t):
        # z_t: (B, C, *IMG_SIZE)
        t_emb = self.time_mlp(t)
        
        x = z_t
        x = self.in_res(x)
        
        # Encoder
        skips = []
        for i, (res, attn, down) in enumerate(zip(self.res_blocks_encoder, self.attn_down[:-1], self.downs)):
            x = res(x, t_emb)
            skips.append(x)
            x = attn(x)
            x = down(x)

        x = self.attn_down[-1](x)
        # Bottleneck
        x = self.bottleneck_res(x, t_emb)
        x = self.attn_up[0](x)
        
        # Decoder
        for up, conv_skip, res, attn in zip(self.ups, self.conv_skip_connection_decored, self.res_blocks_decoder, self.attn_up[1:]):
            x = up(x)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = conv_skip(x)
            x = res(x, t_emb)
            x = attn(x)

        return self.out_res(x)

