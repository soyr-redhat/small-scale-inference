import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation=None):
        super().__init__()
        self.expand = nn.Linear(d_model, d_ff)
        self.activation = activation or nn.GELU(approximate="tanh")
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand -> activate -> project back down
        x = self.expand(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x
