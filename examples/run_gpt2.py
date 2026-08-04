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

    configs = [
        {"label": "Greedy (temperature=0.01)", "temperature": 0.01},
        {"label": "Creative (temperature=0.9, top_k=50)", "temperature": 0.9, "top_k": 50},
        {"label": "Nucleus (temperature=0.8, top_p=0.9)", "temperature": 0.8, "top_p": 0.9},
    ]

    for config in configs:
        label = config.pop("label")
        print(f"\n--- {label} ---")
        print(f"Prompt: {prompt}")
        with torch.no_grad():
            output_ids = model.generate(token_ids, max_new_tokens=30, **config)
        print(f"Output: {tokenizer.decode(output_ids[0])}")


if __name__ == "__main__":
    main()
