"""
Milestone 1 & 2: scaled dot-product attention + multi-head attention.

The hand-calculated numeric example (3 tokens, d_k=2, d_v=2) belongs in the
report (Section 6 of the handout) -- this file is the general, testable
implementation that your hand calculation should match.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, causal_mask: bool = False):
    """
    q, k, v: (..., seq_len, d_k) / (..., seq_len, d_v)
    Returns: (output, attention_weights)
    """
    d_k = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)  # (..., seq_len, seq_len)

    if causal_mask:
        seq_len = scores.size(-1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    output = weights @ v
    return output, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask: bool = True):
        B, T, C = x.shape  # batch, seq_len, d_model

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # split into heads: (B, T, C) -> (B, n_heads, T, d_head)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        out, attn_weights = scaled_dot_product_attention(q, k, v, causal_mask=causal_mask)
        # out: (B, n_heads, T, d_head)

        # merge heads back: (B, n_heads, T, d_head) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.out_proj(out))
        return out, attn_weights


if __name__ == "__main__":
    # shape sanity check + causal mask sanity check
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)  # batch=2, seq_len=5, d_model=16
    mha = MultiHeadAttention(d_model=16, n_heads=4)
    out, attn = mha(x, causal_mask=True)
    print("output shape:", out.shape)          # (2, 5, 16)
    print("attn weights shape:", attn.shape)   # (2, 4, 5, 5)
    # confirm upper triangle (future positions) is ~0 after softmax
    print("future attention (should be ~0):", attn[0, 0, 0, 1:].sum().item())
