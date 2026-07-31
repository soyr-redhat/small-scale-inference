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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input to Q, K, V
        batch, seq_len, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into multiple heads: (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask — prevent attending to future tokens
        lower_tri = torch.tril(torch.ones(seq_len, seq_len))
        scores = scores.masked_fill(lower_tri == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, V)

        # Merge heads back: (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.d_k)
        return self.out_proj(output)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.expand = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.mha(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_seq_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Embed tokens and positions, then sum
        _, seq_len = token_ids.shape
        tok_emb = self.token_emb(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.pos_emb(positions)
        x = tok_emb + pos_emb

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and project to vocabulary logits
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x


    def generate(self, token_ids: torch.Tensor, max_new_tokens: int = 50) -> torch.Tensor:
        # Greedy autoregressive decoding
        for _ in range(max_new_tokens):
            # Truncate to max context window
            token_ids = token_ids[:, -self.max_seq_len:]
            logits = self.forward(token_ids)
            # Take logits at last position and pick highest-probability token
            logits = logits[:, -1, :]
            top_one = torch.argmax(logits, dim=-1, keepdim=True)
            token_ids = torch.concat([token_ids, top_one], dim=1)

        return token_ids


def load_gpt2_weights(model: GPT2):
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_sd = hf.state_dict()

    with torch.no_grad():
        model.token_emb.weight.copy_(hf_sd["transformer.wte.weight"])
        model.pos_emb.weight.copy_(hf_sd["transformer.wpe.weight"])
        model.final_norm.weight.copy_(hf_sd["transformer.ln_f.weight"])
        model.final_norm.bias.copy_(hf_sd["transformer.ln_f.bias"])

        for i, block in enumerate(model.blocks):
            # HF GPT-2 stores Q,K,V as one combined matrix and uses Conv1D (transposed weights)
            qkv_w = hf_sd[f"transformer.h.{i}.attn.c_attn.weight"]
            qkv_b = hf_sd[f"transformer.h.{i}.attn.c_attn.bias"]
            q_w, k_w, v_w = qkv_w.chunk(3, dim=1)
            q_b, k_b, v_b = qkv_b.chunk(3)

            block.mha.q_proj.weight.copy_(q_w.t())
            block.mha.q_proj.bias.copy_(q_b)
            block.mha.k_proj.weight.copy_(k_w.t())
            block.mha.k_proj.bias.copy_(k_b)
            block.mha.v_proj.weight.copy_(v_w.t())
            block.mha.v_proj.bias.copy_(v_b)

            out_w = hf_sd[f"transformer.h.{i}.attn.c_proj.weight"]
            block.mha.out_proj.weight.copy_(out_w.t())
            block.mha.out_proj.bias.copy_(hf_sd[f"transformer.h.{i}.attn.c_proj.bias"])

            block.norm1.weight.copy_(hf_sd[f"transformer.h.{i}.ln_1.weight"])
            block.norm1.bias.copy_(hf_sd[f"transformer.h.{i}.ln_1.bias"])
            block.norm2.weight.copy_(hf_sd[f"transformer.h.{i}.ln_2.weight"])
            block.norm2.bias.copy_(hf_sd[f"transformer.h.{i}.ln_2.bias"])

            fc_w = hf_sd[f"transformer.h.{i}.mlp.c_fc.weight"]
            block.ff.expand.weight.copy_(fc_w.t())
            block.ff.expand.bias.copy_(hf_sd[f"transformer.h.{i}.mlp.c_fc.bias"])

            proj_w = hf_sd[f"transformer.h.{i}.mlp.c_proj.weight"]
            block.ff.down_proj.weight.copy_(proj_w.t())
            block.ff.down_proj.bias.copy_(hf_sd[f"transformer.h.{i}.mlp.c_proj.bias"])

        model.lm_head.weight.copy_(model.token_emb.weight)

    print("Weights loaded!")


# --- Test it ---
if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    model = GPT2()
    load_gpt2_weights(model)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "The meaning of life is"
    token_ids = tokenizer.encode(prompt, return_tensors="pt")

    print(f"Prompt: {prompt}")
    with torch.no_grad():
        output_ids = model.generate(token_ids, max_new_tokens=30)

    print(f"Output: {tokenizer.decode(output_ids[0])}")
