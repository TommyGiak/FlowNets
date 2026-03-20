# FlowNets

FlowNets is a lightweight PyTorch package that provides U-Net backbones for continuous-time generative models such as conditional flow matching (CFM), flow matching, and diffusion-style models.

The project currently exposes two main architectures:

- `SimpleUNet`: a residual U-Net with sinusoidal time conditioning.
- `SelfUNet`: a residual U-Net with patch-based self-attention blocks in the encoder, bottleneck, and decoder.

The code is intentionally small, readable, and easy to adapt for research projects.

## Features

- PyTorch implementations for 2D and 3D inputs.
- Sinusoidal time embeddings for continuous-time models.
- Residual blocks with `GroupNorm`, `SiLU`, dropout, and zero-initialized residual output convolutions.
- Patch-based self-attention with positional encoding and AdaLN-Zero-style conditioning in `SelfUNet`.
- Lightweight utility layers for convolution, transpose convolution, pooling, upsampling, tokenization, and attention.
- Minimal `pytest` test suite for importability and forward-pass validation.

## Installation

Install the package from PyPI:

```bash
pip install flownets
```

## Quick Start

`SimpleUNet` is the basic residual U-Net. It is the easiest starting point if you want a compact backbone with time conditioning and no attention.

```python
import torch
from flownets import SimpleUNet, SelfUNet

simple_unet = SimpleUNet(img_size=(32, 32), in_channels=1)
x = torch.randn(4, 1, 32, 32)  # (batch, channels, height, width)
t = torch.rand(4)

simple_out = simple_unet(t, x)
print(simple_out.shape)  # torch.Size([4, 1, 32, 32])
```

`SelfUNet` adds self-attention blocks at selected resolutions. You control where attention is active with `attn_p_per_down` and `attn_p_per_up`.

```python
import torch
from flownets import SelfUNet

self_unet = SelfUNet(
    img_size=(32, 32),
    in_channels=1,
    channels_per_down=[8, 16, 32, 64, 128],
    attn_p_per_down=[None, None, 2, 1, 1],
    attn_p_per_up=[1, 1, 2, None, None],
    time_emb_dim=264,
)

x = torch.randn(4, 1, 32, 32)
t = torch.rand(4)

self_out = self_unet(t, x)
print(self_out.shape)  # torch.Size([4, 1, 32, 32])
```

### How the attention configuration works

`attn_p_per_down` and `attn_p_per_up` are lists aligned with `channels_per_down`.

Example:

```python
channels_per_down = [8, 16, 32, 64, 128]
attn_p_per_down  = [None, None, 2, 1, 1]
attn_p_per_up    = [1, 1, 2, None, None]
```

This means:

- at the highest resolutions, attention is disabled with `None`
- at deeper resolutions, attention is enabled
- the integer value is the `patch_size` used to tokenize the feature map before self-attention

In practice, the attention block does not attend directly over single pixels or voxels. It first groups the spatial grid into non-overlapping patches, projects each patch into a token embedding, applies self-attention over the resulting token sequence, and then reconstructs the feature map.

For a 2D feature map:

- `patch_size=1` means one token per spatial location
- `patch_size=2` means one token per `2 x 2` patch
- larger patch sizes reduce the number of tokens and therefore reduce attention cost

For a 3D feature map:

- `patch_size=1` means one token per voxel location
- `patch_size=2` means one token per `2 x 2 x 2` patch

### Why use different patch sizes at different levels

The deeper the U-Net goes, the smaller the spatial resolution becomes. This makes attention cheaper in the bottleneck and lower-resolution stages.

Typical pattern:

- disable attention at very high resolutions
- enable attention only in deeper stages
- use `patch_size=2` when you want fewer attention tokens
- use `patch_size=1` when you want the finest spatial attention at already small resolutions

The default:

```python
attn_p_per_down = [None, None, 2, 1, 1]
```

is a reasonable compromise:

- no attention in the first high-resolution stages
- coarser attention with `2 x 2` patches in an intermediate stage
- full token-per-location attention in the deepest stages

### Important constraints

- the length of `attn_p_per_down` must match `channels_per_down`
- if provided, `attn_p_per_up` must also match `channels_per_down`
- if `attn_p_per_up=None`, the model reuses `attn_p_per_down`
- any spatial resolution where attention is active must be divisible by the chosen `patch_size`
- `time_emb_dim` must be divisible by `2 * len(img_size)`

## Architectures

### `SimpleUNet`

`SimpleUNet` is the lighter backbone and is appropriate when you want a compact U-Net without attention.

Constructor arguments:

- `img_size: tuple`
  Spatial size of the input. Only 2D and 3D inputs are supported.
- `in_channels: int = 1`
  Number of input channels.
- `channels_per_down: list[int] = [8, 16, 32, 64, 128]`
  Channel widths across encoder stages.
- `n_residuals_blocks: int = 1`
  Number of residual blocks per stage.
- `time_emb_dim: int = 256`
  Dimension of the sinusoidal time embedding MLP.
- `dropout: float = 0`
  Dropout probability used in the input stem and residual blocks.

Forward signature:

```python
output = model(t, z_t)
```

- `t`: shape `(B,)` or `(B, 1)`
- `z_t`: shape `(B, C, *img_size)`
- `output`: same shape as `z_t`

### `SelfUNet`

`SelfUNet` extends the residual U-Net with patch-based self-attention blocks.

Constructor arguments:

- `img_size: tuple`
  Spatial size of the input. Only 2D and 3D inputs are supported.
- `in_channels: int = 1`
  Number of input channels.
- `channels_per_down: list[int] = [8, 16, 32, 64, 128]`
  Channel widths across encoder stages.
- `attn_p_per_down: list[int | None] = [None, None, 2, 1, 1]`
  Patch size for each encoder-resolution attention block. Use `None` to disable attention at a stage.
- `attn_p_per_up: list[int | None] | None = None`
  Patch size for each decoder-resolution attention block. If `None`, it reuses `attn_p_per_down`.
- `n_residuals_blocks: int = 1`
  Number of residual blocks per stage.
- `time_emb_dim: int = 264`
  Time embedding dimension. It must be divisible by `2 * spatial_dims`.
- `dropout: float = 0`
  Dropout probability used in the model.

Forward signature:

```python
output = model(t, z_t)
```

- `t`: shape `(B,)` or `(B, 1)`
- `z_t`: shape `(B, C, *img_size)`
- `output`: same shape as `z_t`

## Internal Building Blocks

The package also contains reusable modules in `flownets/BlocksAndLayers.py`, including:

- `SinusoidalTimeEmb`
- `ResidualBlock`
- `Downsample`
- `Upsample`
- `Tokenizer`
- `PositionalEncodingND`
- `SelfAttentionBlock`
- `CrossAttention`

These are useful if you want to assemble a custom backbone around the same conventions used by `SimpleUNet` and `SelfUNet`.

## Shape and Configuration Constraints

Some constraints come directly from the implementation:

- Only 2D and 3D image tensors are supported by the exported U-Nets.
- `SelfUNet.time_emb_dim` must be divisible by `2 * len(img_size)`.
- Any spatial resolution that passes through a patch-based attention block must be divisible by that stage's `patch_size`.
- Inputs should be large enough to survive repeated downsampling by a factor of `2` across the chosen depth.
- Very small values for the first channel count can cause invalid `GroupNorm` configurations. In practice, use channel schedules such as `[8, 16, 32, ...]`.

## Typical Use in Flow Matching or Diffusion Training

The networks are generic backbones. They do not implement a full training loop, loss, sampler, or probability path by themselves.

Typical usage:

1. Sample a time `t`.
2. Construct the noisy or interpolated state `z_t`.
3. Pass `t` and `z_t` through the network.
4. Train the model to predict the desired target field, for example velocity, score surrogate, or noise target depending on your objective.

Example sketch:

```python
import torch
from flownets import SimpleUNet

model = SimpleUNet(img_size=(32, 32), in_channels=1)

x = torch.randn(8, 1, 32, 32)
t = torch.rand(8)
target = torch.randn_like(x)

pred = model(t, x)
loss = torch.mean((pred - target) ** 2)
loss.backward()
```

## Development

For local development, clone the repository and install it in editable mode:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
pytest -q
```

## Project Structure

```plaintext
FlowNets/
├── flownets/
│   ├── __init__.py
│   ├── BlocksAndLayers.py
│   ├── README.md
│   ├── UNets.py
│   └── version.py
├── tests/
│   ├── conftest.py
│   ├── test_blocks_and_layers.py
│   └── test_unets.py
├── LICENSE
├── MANIFEST.in
├── README.md
├── requirements.txt
└── setup.py
```

## Versioning

The package version is defined in `flownets/version.py`.

Current release:

```python
__version__ = "0.0.6"
```

## Notes

- This package is focused on model backbones, not full end-to-end training pipelines.
- The code favors directness and hackability over a large abstraction layer.
- If you plan to publish or reuse these models broadly, it is worth adding more tests around custom channel schedules, 3D attention configurations, and training-time numerical stability.
