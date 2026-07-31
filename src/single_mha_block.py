import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into multiple heads: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        batch, seq_len, _ = x.shape
        Q = Q.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        scores = (torch.matmul(Q, K.transpose(-2, -1))) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        # Merge heads back: (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(output)


if __name__ == "__main__":
    batch, seq_len, d_model, n_heads = 2, 10, 64, 4

    x = torch.randn(batch, seq_len, d_model)
    mha = MultiHeadAttention(d_model, n_heads)
    out = mha(x)

    print(f"Input shape:  {x.shape}")   # should be (2, 10, 64)
    print(f"Output shape: {out.shape}")  # should be (2, 10, 64)
    assert out.shape == x.shape, f"Shape mismatch! Got {out.shape}, expected {x.shape}"
    print("Shapes match!")