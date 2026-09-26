# MiniGPT From Scratch

A small GPT-style causal language model implemented from scratch (no pretrained
checkpoints, no `transformers` model classes) for the AI Engineering
Transformer/GPT assignment.

## Repository layout
- `src/` — tokenizer, dataset, attention, positional encoding, transformer
  block, full model, training loop, inference CLI.
- `tests/` — shape checks, causal-mask correctness, tiny-overfit debugging gate.
- `experiments/` — attention/positional-encoding visualizations, ablation study.
- `configs/tiny_gpt.yaml` — all hyperparameters (edit here, not in code).
- `app.py` — FastAPI wrapper for deployed inference.
- `outputs/` — logs, figures, generation samples produced by the scripts above.
- `checkpoints/` — trained weights + tokenizer vocab (not committed — see below).

## Dataset
[Tiny Shakespeare](https://huggingface.co/datasets/tiny_shakespeare) — ~1.11MB,
~40,000 lines. Download it into `data/tiny_shakespeare.txt` (see "Setup" below).

## Setup (local or Colab)
```bash
git clone <your-repo-url>
cd transformer-gpt-assignment
pip install -r requirements.txt

# get the dataset
mkdir -p data
curl -L -o data/tiny_shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Run the tests (do this before training)
```bash
pytest tests/ -v
```
All tests should pass, including the causal-mask test and the tiny-overfit
debugging gate.

## Train
```bash
# quick debugging run — loss should drop sharply on a repeated batch
python -m src.train --config configs/tiny_gpt.yaml --overfit

# full training run
python -m src.train --config configs/tiny_gpt.yaml
```
This writes `checkpoints/model.pt`, `checkpoints/vocab.json`, and a loss log to
`outputs/logs/train_log.csv`.

## Generate text
```bash
python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80
```
Expected output shape:
```
Prompt: The future of artificial intelligence
Generated: The future of artificial intelligence ...
model parameters: 3.2M
context length: 128
temperature: 0.8
```
(Your exact numbers will differ — that's expected.)

## Visualizations and ablations
```bash
python -m experiments.attention_visualization --checkpoint checkpoints/model.pt --prompt "To be"
python -m experiments.ablation --factor depth
python -m experiments.ablation --factor heads
python -m experiments.ablation --factor context_length
```

## Serve the model (API)
```bash
uvicorn app:app --reload
# POST http://127.0.0.1:8000/generate  {"prompt": "...", "max_new_tokens": 50}
```
To deploy: push to GitHub, connect the repo on Render (or similar) as a Web
Service with start command `uvicorn app:app --host 0.0.0.0 --port $PORT`.

## Known limitations
- Character-level tokenizer keeps things simple but is less sample-efficient
  than a subword tokenizer.
- Model scale is intentionally small; do not expect long-range coherence.
- No KV-cache, so generation re-runs the full forward pass every step (fine at
  this scale).

## Constraints followed
- No pretrained GPT/BERT/Llama/Mistral checkpoints loaded.
- No `transformers` model classes (`AutoModelForCausalLM`, etc.) used as the
  submitted architecture.
- PyTorch tensors, autograd, optimizers, and data loaders are used as allowed.
