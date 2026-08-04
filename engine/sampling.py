import torch


def sample_next_token(
   logits: torch.Tensor,
   temperature: float = 1.0,
   top_k: int | None = None,
   top_p: float | None = None) -> torch.Tensor:
   """
   Sample the next token from logits.
   logits shape: (batch, vocab_size) — raw scores for each token in the vocabulary
   Returns: token IDs, shape (batch, 1)
   """
   logits = logits / temperature

   if top_k is not None:
      values, _ = torch.topk(logits, top_k)
      min_value = values[:, -1:]
      logits[logits < min_value] = float("-inf")

   if top_p is not None:
      sorted_logits, sorted_i = torch.sort(logits, descending=True)
      cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
      sorted_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
      sorted_logits[sorted_mask] = float("-inf")
      logits = sorted_logits.scatter(1, sorted_i, sorted_logits)

   probs = torch.softmax(logits, dim=-1)
   return torch.multinomial(probs, num_samples=1)
         
