"""
Milestone 6: required ablation experiments (Section 13 of the handout).
Change one factor at a time; record val loss, param count, and training time.

Usage:
    python -m experiments.ablation --factor depth
    python -m experiments.ablation --factor heads
    python -m experiments.ablation --factor context_length
"""
import argparse
import csv
import time
import yaml
from pathlib import Path
import torch

from src.tokenizer import CharTokenizer
from src.dataset import load_and_split
from src.gpt import MiniGPT
from src.train import get_device, estimate_loss

FACTOR_VALUES = {
    "depth": {"key": "n_layers", "values": [2, 4, 6]},
    "heads": {"key": "n_heads", "values": [2, 4, 8]},
    "context_length": {"key": "context_length", "values": [64, 128, 256]},
}


def run_one(cfg, device, text):
    tokenizer = CharTokenizer.from_text(text)
    train_ds, val_ds = load_and_split(text, tokenizer, cfg["context_length"], cfg["val_fraction"])

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    start = time.time()
    for _ in range(cfg["ablation_steps"]):
        x, y = train_ds.get_batch(cfg["batch_size"], device)
        _, loss, _ = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start

    val_loss = estimate_loss(model, val_ds, cfg["batch_size"], device)
    return {
        "params": model.num_parameters(),
        "val_loss": val_loss,
        "train_time_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", choices=list(FACTOR_VALUES.keys()), required=True)
    parser.add_argument("--config", type=str, default="configs/tiny_gpt.yaml")
    parser.add_argument("--ablation-steps", type=int, default=500,
                         help="Fewer steps than full training -- this is a comparison, not a final model.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["ablation_steps"] = args.ablation_steps
    device = get_device()
    text = Path(cfg["data_path"]).read_text(encoding="utf-8")

    spec = FACTOR_VALUES[args.factor]
    results = []
    for value in spec["values"]:
        run_cfg = dict(cfg)
        run_cfg[spec["key"]] = value
        print(f"Running {args.factor}={value} ...")
        metrics = run_one(run_cfg, device, text)
        metrics[spec["key"]] = value
        results.append(metrics)
        print(metrics)

    out_path = Path(f"outputs/logs/ablation_{args.factor}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
