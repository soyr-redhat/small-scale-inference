import torch
import torch.nn as nn
import math

from single_mha_block import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.expand = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand -> activate -> project back down
        x = self.expand(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.sub_1 = nn.LayerNorm(d_model)
        self.sub_2 = nn.LayerNorm(d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.mha(self.sub_1(x))
        x = x + self.ff(self.sub_2(x))
        return x


# --- Test it ---
if __name__ == "__main__":
    batch, seq_len, d_model, n_heads = 2, 10, 64, 4
    d_ff = d_model * 4  # 256

    x = torch.randn(batch, seq_len, d_model)
    block = TransformerBlock(d_model, n_heads, d_ff)
    out = block(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch! Got {out.shape}, expected {x.shape}"
    print("Single block works!")

    # Stack multiple blocks — this is what makes a transformer deep
    n_layers = 6
    blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])

    out = x
    for block in blocks:
        out = block(out)

    print(f"\n{n_layers}-layer output shape: {out.shape}")
    assert out.shape == x.shape
    print(f"{n_layers}-layer stack works!")
