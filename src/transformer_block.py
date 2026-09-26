"""
Milestone 3: Transformer decoder block = MHA + FFN + residual + LayerNorm.
Uses pre-norm (norm before sublayer), which trains more stably than post-norm
for small models -- justify this choice in your report if asked.
"""
import torch.nn as nn
from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.ln1(x), causal_mask=True)
        x = x + attn_out                 # residual around attention
        x = x + self.ffn(self.ln2(x))    # residual around FFN
        return x, attn_weights
