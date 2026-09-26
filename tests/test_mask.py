import torch
from src.attention import scaled_dot_product_attention


def test_future_positions_are_masked():
    """A token at position t must not attend to any position > t."""
    torch.manual_seed(0)
    q = torch.randn(1, 5, 4)
    k = torch.randn(1, 5, 4)
    v = torch.randn(1, 5, 4)
    _, weights = scaled_dot_product_attention(q, k, v, causal_mask=True)

    T = weights.size(-1)
    for t in range(T):
        future_weight = weights[0, t, t + 1:].sum().item()
        assert future_weight < 1e-6, f"position {t} attends to the future: {future_weight}"


def test_no_mask_allows_future_attention():
    """Sanity check: without the mask, future attention is (generically) nonzero."""
    torch.manual_seed(1)
    q = torch.randn(1, 5, 4)
    k = torch.randn(1, 5, 4)
    v = torch.randn(1, 5, 4)
    _, weights = scaled_dot_product_attention(q, k, v, causal_mask=False)
    assert weights[0, 0, 1:].sum().item() > 0
