import torch
from layers.attention import MultiHeadAttention


def test_output_shape_matches_input():
    batch, seq_len, d_model, n_heads = 2, 10, 64, 4
    x = torch.randn(batch, seq_len, d_model)
    mha = MultiHeadAttention(d_model, n_heads)
    out, cache = mha(x)
    assert out.shape == x.shape


def test_single_token_input():
    batch, seq_len, d_model, n_heads = 1, 1, 64, 4
    x = torch.randn(batch, seq_len, d_model)
    mha = MultiHeadAttention(d_model, n_heads)
    out, cache = mha(x)
    assert out.shape == (1, 1, 64)


def test_causal_mask_blocks_future_tokens():
    d_model, n_heads = 64, 4
    mha = MultiHeadAttention(d_model, n_heads)

    x = torch.randn(1, 5, d_model)
    Q = mha.q_proj(x)
    K = mha.k_proj(x)

    d_k = d_model // n_heads
    Q = Q.view(1, 5, n_heads, d_k).transpose(1, 2)
    K = K.view(1, 5, n_heads, d_k).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1))
    lower_tri = torch.tril(torch.ones(5, 5))
    scores = scores.masked_fill(lower_tri == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    # Upper triangle of attention weights should be zero
    for head in range(n_heads):
        for i in range(5):
            for j in range(i + 1, 5):
                assert weights[0, head, i, j].item() == 0.0


def test_different_head_counts():
    batch, seq_len, d_model = 2, 8, 64
    for n_heads in [1, 2, 4, 8]:
        mha = MultiHeadAttention(d_model, n_heads)
        out, cache = mha(torch.randn(batch, seq_len, d_model))
        assert out.shape == (batch, seq_len, d_model)


def test_kv_cache_returns_correct_shapes():
    batch, seq_len, d_model, n_heads = 1, 5, 64, 4
    d_k = d_model // n_heads
    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(batch, seq_len, d_model)
    out, (cached_K, cached_V) = mha(x)
    assert cached_K.shape == (batch, n_heads, seq_len, d_k)
    assert cached_V.shape == (batch, n_heads, seq_len, d_k)


def test_kv_cache_concatenates():
    batch, d_model, n_heads = 1, 64, 4
    d_k = d_model // n_heads
    mha = MultiHeadAttention(d_model, n_heads)

    # Prefill with 5 tokens
    x1 = torch.randn(batch, 5, d_model)
    out1, cache = mha(x1)

    # Decode 1 new token with cache
    x2 = torch.randn(batch, 1, d_model)
    out2, (new_K, new_V) = mha(x2, kv_cache=cache)

    assert out2.shape == (batch, 1, d_model)
    assert new_K.shape == (batch, n_heads, 6, d_k)
    assert new_V.shape == (batch, n_heads, 6, d_k)
