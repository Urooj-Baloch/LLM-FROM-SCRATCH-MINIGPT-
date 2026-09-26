import torch
from src.gpt import MiniGPT


def test_forward_shapes():
    vocab_size, context_length = 50, 16
    model = MiniGPT(vocab_size=vocab_size, context_length=context_length,
                     d_model=32, n_heads=4, n_layers=2)
    x = torch.randint(0, vocab_size, (3, context_length))
    y = torch.randint(0, vocab_size, (3, context_length))
    logits, loss, attn = model(x, y)
    assert logits.shape == (3, context_length, vocab_size)
    assert loss.item() > 0
    assert attn.shape == (3, 4, context_length, context_length)


def test_generate_extends_sequence():
    vocab_size, context_length = 20, 8
    model = MiniGPT(vocab_size=vocab_size, context_length=context_length,
                     d_model=16, n_heads=2, n_layers=2)
    idx = torch.randint(0, vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 14)
