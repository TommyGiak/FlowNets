import pytest
import torch

from flownets import SimpleUNet, SelfUNet


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (SimpleUNet, {}),
        (SelfUNet, {}),
    ],
)
def test_unet_forward_preserves_2d_shape(model_cls, kwargs):
    batch_size, in_channels, height, width = 2, 1, 32, 32
    model = model_cls(img_size=(height, width), in_channels=in_channels, **kwargs)
    x = torch.randn(batch_size, in_channels, height, width)
    t = torch.rand(batch_size)

    out = model(t, x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_simple_unet_forward_preserves_3d_shape():
    batch_size, in_channels = 2, 1
    depth, height, width = 16, 16, 16
    model = SimpleUNet(
        img_size=(depth, height, width),
        in_channels=in_channels,
        channels_per_down=[8, 16, 32],
    )
    x = torch.randn(batch_size, in_channels, depth, height, width)
    t = torch.rand(batch_size)

    out = model(t, x)

    assert out.shape == x.shape


def test_self_unet_rejects_invalid_time_embedding_dim():
    with pytest.raises(AssertionError, match="time_emb_dim must be divisible"):
        SelfUNet(
            img_size=(32, 32),
            in_channels=1,
            time_emb_dim=258,
        )
