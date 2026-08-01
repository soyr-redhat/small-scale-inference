import torch
from layers.transformer import TransformerBlock


def test_output_shape_matches_input():
    batch, seq_len, d_model, n_heads, d_ff = 2, 10, 64, 4, 256
    x = torch.randn(batch, seq_len, d_model)
    block = TransformerBlock(d_model, n_heads, d_ff)
    out = block(x)
    assert out.shape == x.shape


def test_residual_changes_output():
    d_model, n_heads, d_ff = 64, 4, 256
    block = TransformerBlock(d_model, n_heads, d_ff)
    x = torch.randn(1, 5, d_model)
    out = block(x)
    assert not torch.allclose(x, out)


def test_stacked_blocks():
    d_model, n_heads, d_ff, n_layers = 64, 4, 256, 6
    blocks = torch.nn.ModuleList(
        [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
    )
    x = torch.randn(2, 10, d_model)
    out = x
    for block in blocks:
        out = block(out)
    assert out.shape == x.shape
