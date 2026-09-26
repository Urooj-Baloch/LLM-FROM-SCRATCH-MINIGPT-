"""
Builds contiguous token sequences and shifted (input, target) pairs for
next-token prediction, per Milestone 4.
"""
import torch


class CharDataset:
    def __init__(self, token_ids, context_length: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def get_batch(self, batch_size: int, device: str = "cpu"):
        """Sample a random batch of (x, y) where y is x shifted by one token."""
        ix = torch.randint(0, len(self), (batch_size,))
        x = torch.stack([self.data[i:i + self.context_length] for i in ix])
        y = torch.stack([self.data[i + 1:i + 1 + self.context_length] for i in ix])
        return x.to(device), y.to(device)


def load_and_split(text: str, tokenizer, context_length: int, val_fraction: float = 0.1):
    ids = tokenizer.encode(text)
    n = len(ids)
    split = int(n * (1 - val_fraction))
    train_ds = CharDataset(ids[:split], context_length)
    val_ds = CharDataset(ids[split:], context_length)
    return train_ds, val_ds
