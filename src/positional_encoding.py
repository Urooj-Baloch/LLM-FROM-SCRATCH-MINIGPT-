"""
Milestone 2: sinusoidal positional encoding (Vaswani et al. formulation).
"""
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]


if __name__ == "__main__":
    pe = SinusoidalPositionalEncoding(d_model=32, max_len=100)
    x = torch.zeros(1, 20, 32)
    out = pe(x)
    print("shape:", out.shape)
    # For a visualization: plot pe.pe[0, :50, :].numpy() as a heatmap
    # (see experiments/attention_visualization.py for a full example)
