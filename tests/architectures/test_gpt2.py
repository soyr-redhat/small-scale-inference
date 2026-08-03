import torch
import pytest
from architectures.gpt2 import GPT2
from weights.loader import load_gpt2_weights


def test_forward_output_shape():
    model = GPT2(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=32)
    token_ids = torch.randint(0, 100, (2, 10))
    logits, caches = model(token_ids)
    assert logits.shape == (2, 10, 100)


def test_generate_output_length():
    model = GPT2(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=32)
    model.eval()
    token_ids = torch.randint(0, 100, (1, 5))
    with torch.no_grad():
        output = model.generate(token_ids, max_new_tokens=10)
    assert output.shape[1] == 15


def test_generate_preserves_prompt():
    model = GPT2(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=32)
    model.eval()
    token_ids = torch.randint(0, 100, (1, 5))
    with torch.no_grad():
        output = model.generate(token_ids, max_new_tokens=5)
    assert torch.equal(output[:, :5], token_ids)


def test_output_matches_huggingface():
    pytest.importorskip("transformers")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    token_ids = tokenizer.encode("The quick brown fox", return_tensors="pt")

    # Our model
    our_model = GPT2()
    load_gpt2_weights(our_model)
    our_model.eval()
    with torch.no_grad():
        our_logits, _ = our_model(token_ids)

    # HuggingFace model
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(token_ids).logits

    assert torch.allclose(our_logits, hf_logits, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_forward_on_gpu():
    model = GPT2(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=32)
    model.to("cuda")
    model.eval()
    token_ids = torch.randint(0, 100, (1, 5), device="cuda")
    with torch.no_grad():
        logits, caches = model(token_ids)
    assert logits.device.type == "cuda"
    assert logits.shape == (1, 5, 100)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_generate_on_gpu():
    model = GPT2(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=32)
    model.to("cuda")
    model.eval()
    token_ids = torch.randint(0, 100, (1, 5), device="cuda")
    with torch.no_grad():
        output = model.generate(token_ids, max_new_tokens=10)
    assert output.device.type == "cuda"
    assert output.shape[1] == 15


def test_cached_output_matches_uncached():
    model = GPT2(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=32)
    model.eval()
    token_ids = torch.randint(0, 100, (1, 5))

    with torch.no_grad():
        # Full forward pass without cache
        logits_no_cache, _ = model(token_ids)

        # Prefill + decode with cache
        prefill_ids = token_ids[:, :-1]
        logits_prefill, caches = model(prefill_ids)

        last_id = token_ids[:, -1:]
        logits_cached, _ = model(last_id, kv_caches=caches)

    # Last token's logits should match whether computed with or without cache
    assert torch.allclose(logits_no_cache[:, -1, :], logits_cached[:, 0, :], atol=1e-5)
