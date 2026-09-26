"""
Milestone 4/5: training loop.

Usage:
    python -m src.train --config configs/tiny_gpt.yaml
    python -m src.train --config configs/tiny_gpt.yaml --overfit   # debugging gate
"""
import argparse
import time
import yaml
import torch
from pathlib import Path

from src.tokenizer import CharTokenizer
from src.dataset import load_and_split
from src.gpt import MiniGPT


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, dataset, batch_size, device, eval_iters=20):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch(batch_size, device)
        _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tiny_gpt.yaml")
    parser.add_argument("--overfit", action="store_true",
                         help="Debugging gate: train on one tiny repeated batch.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = get_device()
    print(f"Using device: {device}")

    text = Path(cfg["data_path"]).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save("checkpoints/vocab.json")

    train_ds, val_ds = load_and_split(text, tokenizer, cfg["context_length"], cfg["val_fraction"])

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ffn_dim=cfg.get("ffn_dim"),
        dropout=cfg["dropout"],
    ).to(device)

    print(f"Model parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    log_path = Path("outputs/logs/train_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("step,train_loss,val_loss,lr\n")

    steps = cfg["overfit_steps"] if args.overfit else cfg["max_steps"]
    fixed_batch = train_ds.get_batch(cfg["batch_size"], device) if args.overfit else None

    start = time.time()
    for step in range(1, steps + 1):
        x, y = fixed_batch if args.overfit else train_ds.get_batch(cfg["batch_size"], device)
        _, loss, _ = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg["eval_interval"] == 0 or step == steps:
            val_loss = estimate_loss(model, val_ds, cfg["batch_size"], device) if not args.overfit else float("nan")
            print(f"step {step}/{steps} | train_loss {loss.item():.4f} | val_loss {val_loss:.4f}")
            with open(log_path, "a") as f:
                f.write(f"{step},{loss.item():.4f},{val_loss},{cfg['learning_rate']}\n")

    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s")

    ckpt_path = "checkpoints/model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "vocab_path": "checkpoints/vocab.json",
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # sample generation from a fixed prompt, for the required evidence figure
    prompt = "The "
    idx = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = model.generate(idx, max_new_tokens=100, temperature=cfg.get("temperature", 0.8))
    print("Sample generation:")
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
