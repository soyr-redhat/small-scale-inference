import torch
import torch.nn as nn

from layers.attention import MultiHeadAttention
from layers.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, activation=None):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
            self, 
            x: torch.Tensor, 
            kv_cache=None) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm residual connections
        attn_out, new_cache = self.mha(self.norm1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, new_cache
