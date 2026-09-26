"""
Milestone 6: package the trained model behind a simple API.

Run locally:
    uvicorn app:app --reload
Then POST to http://127.0.0.1:8000/generate

Deploy: push this repo to GitHub, connect it on render.com (or similar) as a
Web Service, start command: uvicorn app:app --host 0.0.0.0 --port $PORT
"""
from fastapi import FastAPI
from pydantic import BaseModel
import torch

from src.generate import load_model

app = FastAPI(title="MiniGPT Inference API")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model, _tokenizer, _cfg = None, None, None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 80
    temperature: float = 0.8


@app.on_event("startup")
def startup():
    global _model, _tokenizer, _cfg
    _model, _tokenizer, _cfg = load_model("checkpoints/model.pt", _device)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    idx = torch.tensor([_tokenizer.encode(req.prompt)], device=_device)
    out = _model.generate(idx, max_new_tokens=req.max_new_tokens, temperature=req.temperature)
    text = _tokenizer.decode(out[0].tolist())
    return {"prompt": req.prompt, "generated": text}
