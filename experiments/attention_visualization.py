"""
Produces two of the required evidence figures:
  Fig 2 - causal mask heatmap
  Fig 4 - positional encoding visualization
  Fig 7 - attention from a trained model (if a checkpoint is passed)

Usage:
    python -m experiments.attention_visualization
    python -m experiments.attention_visualization --checkpoint checkpoints/model.pt --prompt "To be"
"""
import argparse
import torch
import matplotlib.pyplot as plt

from src.attention import scaled_dot_product_attention
from src.positional_encoding import SinusoidalPositionalEncoding


def plot_causal_mask(seq_len=10, save_path="outputs/figures/causal_mask.png"):
    torch.manual_seed(0)
    q = torch.randn(1, seq_len, 4)
    k = torch.randn(1, seq_len, 4)
    v = torch.randn(1, seq_len, 4)
    _, weights = scaled_dot_product_attention(q, k, v, causal_mask=True)

    plt.figure(figsize=(5, 5))
    plt.imshow(weights[0].detach().numpy(), cmap="viridis")
    plt.title("Causal attention weights (query row x key col)")
    plt.xlabel("key position")
    plt.ylabel("query position")
    plt.colorbar()
    plt.savefig(save_path)
    print(f"Saved {save_path}")


def plot_positional_encoding(d_model=64, max_len=100, save_path="outputs/figures/positional_encoding.png"):
    pe = SinusoidalPositionalEncoding(d_model, max_len)
    matrix = pe.pe[0, :max_len, :].numpy()

    plt.figure(figsize=(8, 5))
    plt.imshow(matrix.T, aspect="auto", cmap="RdBu")
    plt.title("Sinusoidal positional encoding")
    plt.xlabel("position")
    plt.ylabel("embedding dimension")
    plt.colorbar()
    plt.savefig(save_path)
    print(f"Saved {save_path}")


def plot_trained_attention(checkpoint_path, prompt, save_path="outputs/figures/trained_attention.png"):
    from src.generate import load_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_model(checkpoint_path, device)
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    _, _, attn = model(idx)

    plt.figure(figsize=(6, 6))
    plt.imshow(attn[0, 0].detach().cpu().numpy(), cmap="viridis")
    plt.title(f"Trained-model attention (head 0), prompt: '{prompt}'")
    plt.xlabel("key position")
    plt.ylabel("query position")
    plt.colorbar()
    plt.savefig(save_path)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="To be or not")
    args = parser.parse_args()

    plot_causal_mask()
    plot_positional_encoding()
    if args.checkpoint:
        plot_trained_attention(args.checkpoint, args.prompt)
