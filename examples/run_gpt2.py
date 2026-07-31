import torch
from transformers import GPT2Tokenizer

from architectures.gpt2 import GPT2
from weights.loader import load_gpt2_weights


def main():
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


if __name__ == "__main__":
    main()
