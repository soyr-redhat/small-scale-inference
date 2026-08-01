import torch
import pytest
from architectures.gpt2 import GPT2
from weights.loader import load_gpt2_weights


def test_forward_output_shape():
    model = GPT2(vocab_size=100, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=32)
    token_ids = torch.randint(0, 100, (2, 10))
    logits = model(token_ids)
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
        our_logits = our_model(token_ids)

    # HuggingFace model
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(token_ids).logits

    assert torch.allclose(our_logits, hf_logits, atol=1e-2)
