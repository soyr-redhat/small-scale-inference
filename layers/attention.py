import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
            self, 
            x: torch.Tensor, 
            kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Project input to Q, K, V
        batch, seq_len, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into multiple heads: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if kv_cache is not None:
            past_K, past_V = kv_cache
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)

        full_seq_len = K.shape[2]
        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask — prevent attending to future tokens
        # Offset rows so cached positions are visible to new queries
        lower_tri = torch.tril(torch.ones(full_seq_len, full_seq_len, device=x.device))
        lower_tri = lower_tri[full_seq_len - seq_len:]  # take last seq_len rows
        scores = scores.masked_fill(lower_tri == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        # Merge heads back: (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.d_k)
        return self.out_proj(output), (K, V)
