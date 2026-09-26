"""
Milestone 5: assemble the full GPT-style causal language model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.positional_encoding import SinusoidalPositionalEncoding
from src.transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 4, ffn_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model

        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=context_length)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx: (B, T) token ids
        x = self.token_emb(idx)          # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.drop(x)

        last_attn = None
        for block in self.blocks:
            x, last_attn = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss, last_attn

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
