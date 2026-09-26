"""
Milestone 6: inference CLI.

Usage:
    python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80
"""
import argparse
import torch

from src.tokenizer import CharTokenizer
from src.gpt import MiniGPT


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    tokenizer = CharTokenizer.load(ckpt["vocab_path"])

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ffn_dim=cfg.get("ffn_dim"),
        dropout=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, tokenizer, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, cfg = load_model(args.checkpoint, device)

    idx = torch.tensor([tokenizer.encode(args.prompt)], device=device)
    out = model.generate(idx, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    generated = tokenizer.decode(out[0].tolist())

    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated}")
    print(f"model parameters: {model.num_parameters() / 1e6:.1f}M")
    print(f"context length: {cfg['context_length']}")
    print(f"temperature: {args.temperature}")


if __name__ == "__main__":
    main()
