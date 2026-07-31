import torch
import torch.nn as nn

from layers.transformer import TransformerBlock


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
