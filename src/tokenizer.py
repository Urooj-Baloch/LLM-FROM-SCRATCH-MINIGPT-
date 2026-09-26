"""
Deterministic character-level tokenizer.

Vocabulary policy:
- Vocabulary = sorted set of unique characters seen in the training corpus.
- One special token: <PAD> (id 0), used only for padding batches if needed.
  (Not strictly required for fixed-length LM training, but documented here
  per the assignment's "documented special-token policy" requirement.)
"""
import json
from pathlib import Path


class CharTokenizer:
    PAD_TOKEN = "<PAD>"

    def __init__(self, chars=None):
        self.chars = chars or []
        self._build_maps()

    def _build_maps(self):
        vocab = [self.PAD_TOKEN] + self.chars
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)

    @classmethod
    def from_text(cls, text: str):
        chars = sorted(list(set(text)))
        return cls(chars)

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str):
        Path(path).write_text(json.dumps({"chars": self.chars}))

    @classmethod
    def load(cls, path: str):
        data = json.loads(Path(path).read_text())
        return cls(data["chars"])


if __name__ == "__main__":
    # quick manual check
    tok = CharTokenizer.from_text("hello world")
    ids = tok.encode("hello")
    print("vocab_size:", tok.vocab_size)
    print("ids:", ids)
    print("decoded:", tok.decode(ids))
