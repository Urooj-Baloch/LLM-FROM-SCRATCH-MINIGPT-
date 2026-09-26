"""
Debugging gate: on a single tiny repeated batch, training loss must fall
substantially. If this fails, gradients or target-shifting are broken --
don't move on to full training until this passes.
"""
import torch
from src.gpt import MiniGPT


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    vocab_size, context_length = 30, 8
    model = MiniGPT(vocab_size=vocab_size, context_length=context_length,
                     d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    x = torch.randint(0, vocab_size, (4, context_length))
    y = torch.randint(0, vocab_size, (4, context_length))

    first_loss = None
    last_loss = None
    for step in range(150):
        _, loss, _ = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()

    assert last_loss < first_loss * 0.5, (
        f"loss did not drop enough: {first_loss:.3f} -> {last_loss:.3f}"
    )
