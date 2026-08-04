import torch
import torch.nn as nn

from layers.transformer import TransformerBlock
from engine.sampling import sample_next_token

class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=12, n_layers=12, d_ff=3072, max_seq_len=1024):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self,
                token_ids: torch.Tensor, 
                kv_caches=None) -> tuple[torch.Tensor, list]:
        # Embed tokens and positions, then sum
        _, seq_len = token_ids.shape
        tok_emb = self.token_emb(token_ids)
        if kv_caches is not None:
            past_len = kv_caches[0][0].shape[2]  # cached seq length
        else:
            past_len = 0
        positions = torch.arange(past_len, past_len + seq_len, device=token_ids.device)
        pos_emb = self.pos_emb(positions)
        x = tok_emb + pos_emb

                # Pass through all transformer blocks
        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches else None
            x, cache = block(x, kv_cache=layer_cache)
            new_caches.append(cache)

        # Final norm and project to vocabulary logits
        x = self.final_norm(x)
        x = self.lm_head(x)
        return (x, new_caches)

    def generate(self, 
                token_ids: torch.Tensor,
                max_new_tokens: int = 50,
                kv_caches = None,
                temperature: float = 1.0,
                top_k: int | None = None,
                top_p: float | None = None
                ) -> torch.Tensor:
        # Greedy autoregressive decoding
        for _ in range(max_new_tokens):
            if kv_caches is None:
                input_ids = token_ids
            else:
                input_ids = token_ids[:, -1:]

            # Truncate to max context window
            logits, kv_caches = self.forward(input_ids, kv_caches)
            # Take logits at last position and pick highest-probability token
            logits = logits[:, -1, :]
            top_one = sample_next_token(logits, temperature, top_k, top_p)
            token_ids = torch.concat([token_ids, top_one], dim=1)

        return token_ids
