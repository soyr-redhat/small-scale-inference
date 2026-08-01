import torch
import torch.nn as nn
from layers.feedforward import FeedForward


def test_output_shape_matches_input():
    batch, seq_len, d_model, d_ff = 2, 10, 64, 256
    x = torch.randn(batch, seq_len, d_model)
    ff = FeedForward(d_model, d_ff)
    out = ff(x)
    assert out.shape == x.shape


def test_custom_activation():
    d_model, d_ff = 64, 256
    ff = FeedForward(d_model, d_ff, activation=nn.ReLU())
    x = torch.randn(1, 5, d_model)
    out = ff(x)
    assert out.shape == x.shape


def test_default_activation_is_gelu():
    ff = FeedForward(64, 256)
    assert isinstance(ff.activation, nn.GELU)


def test_expansion_ratio():
    d_model, d_ff = 64, 256
    ff = FeedForward(d_model, d_ff)
    assert ff.expand.in_features == d_model
    assert ff.expand.out_features == d_ff
    assert ff.down_proj.in_features == d_ff
    assert ff.down_proj.out_features == d_model
