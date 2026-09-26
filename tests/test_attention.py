import torch
from src.attention import scaled_dot_product_attention, MultiHeadAttention


def test_scaled_dot_product_shapes():
    B, T, d_k, d_v = 2, 4, 8, 8
    q = torch.randn(B, T, d_k)
    k = torch.randn(B, T, d_k)
    v = torch.randn(B, T, d_v)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, T, d_v)
    assert weights.shape == (B, T, T)


def test_attention_weights_sum_to_one():
    q = torch.randn(1, 5, 4)
    k = torch.randn(1, 5, 4)
    v = torch.randn(1, 5, 4)
    _, weights = scaled_dot_product_attention(q, k, v)
    sums = weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_multihead_attention_output_shape():
    B, T, d_model, n_heads = 2, 6, 32, 4
    x = torch.randn(B, T, d_model)
    mha = MultiHeadAttention(d_model, n_heads)
    out, weights = mha(x, causal_mask=True)
    assert out.shape == (B, T, d_model)
    assert weights.shape == (B, n_heads, T, T)
