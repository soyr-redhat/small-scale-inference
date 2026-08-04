import torch
from engine.sampling import sample_next_token


def test_output_shape():
    for batch in [1, 4]:
        logits = torch.randn(batch, 100)
        result = sample_next_token(logits)
        assert result.shape == (batch, 1)


def test_low_temperature_is_greedy():
    torch.manual_seed(42)
    logits = torch.randn(1, 100)
    expected = torch.argmax(logits, dim=-1, keepdim=True)
    for _ in range(10):
        result = sample_next_token(logits, temperature=0.01)
        assert torch.equal(result, expected)


def test_top_k_filters_tokens():
    torch.manual_seed(42)
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
    for _ in range(10):
        result = sample_next_token(logits, top_k=1)
        assert result.item() == 0


def test_top_p_filters_tokens():
    torch.manual_seed(42)
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
    for _ in range(10):
        result = sample_next_token(logits, top_p=0.1)
        assert result.item() == 0


def test_temperature_scaling():
    torch.manual_seed(42)
    logits = torch.randn(1, 20)

    hot_tokens = set()
    for _ in range(100):
        result = sample_next_token(logits, temperature=100.0)
        hot_tokens.add(result.item())

    cold_tokens = set()
    for _ in range(100):
        result = sample_next_token(logits, temperature=0.01)
        cold_tokens.add(result.item())

    assert len(hot_tokens) > len(cold_tokens)
