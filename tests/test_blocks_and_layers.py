import torch

from flownets.BlocksAndLayers import SinusoidalTimeEmb, Tokenizer, get_num_groups_for_channels


def test_get_num_groups_for_channels_returns_divisor():
    assert get_num_groups_for_channels(8) == 4
    assert get_num_groups_for_channels(32) == 16
    assert get_num_groups_for_channels(7) == 1


def test_sinusoidal_time_embedding_shape():
    emb = SinusoidalTimeEmb(64)
    t = torch.rand(5)
    out = emb(t)

    assert out.shape == (5, 64)


def test_tokenizer_round_trip_2d():
    tokenizer = Tokenizer(image_size=(8, 8), patch_size=2)
    x = torch.randn(2, 3, 8, 8)

    tokens = tokenizer.tokenization(x)
    restored = tokenizer.invert_tokenization(tokens)

    assert tokens.shape == (2, 16, 12)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)
