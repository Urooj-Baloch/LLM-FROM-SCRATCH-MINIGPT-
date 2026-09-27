"""
Milestone 6: package the trained model behind a simple API, with a built-in
chat UI served at the root URL.

Run locally:
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/ in a browser.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import torch

from src.generate import load_model

app = FastAPI(title="MiniGPT Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MiniGPT Text Continuation</title>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; background: #0f0f14; color: #eee; }
  h1 { font-size: 1.4rem; }
  p.note { color: #999; font-size: 0.9rem; margin-top: -10px; }
  #chatBox { border: 1px solid #333; border-radius: 10px; min-height: 300px; padding: 16px; margin-bottom: 16px; background: #16161d; }
  .msg { margin-bottom: 14px; }
  .prompt { color: #7dd3fc; font-weight: 600; }
  .response { color: #d1d5db; white-space: pre-wrap; margin-top: 4px; }
  #inputRow { display: flex; gap: 8px; }
  #promptInput { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #333; background: #1a1a22; color: #eee; font-size: 1rem; }
  button { padding: 12px 20px; border-radius: 8px; border: none; background: #7c3aed; color: white; font-weight: 600; cursor: pointer; }
  button:disabled { background: #444; cursor: not-allowed; }
  #status { font-size: 0.85rem; color: #888; margin-top: 8px; }
</style>
</head>
<body>
  <h1>MiniGPT Text Continuation</h1>
  <p class="note">This is a raw language model, not a chatbot — it continues your text in Shakespeare's style rather than answering questions.</p>
  <div id="chatBox"></div>
  <div id="inputRow">
    <input id="promptInput" type="text" placeholder="Type the start of a sentence..." />
    <button id="sendBtn">Send</button>
  </div>
  <div id="status"></div>

<script>
const chatBox = document.getElementById('chatBox');
const promptInput = document.getElementById('promptInput');
const sendBtn = document.getElementById('sendBtn');
const statusEl = document.getElementById('status');

async function sendPrompt() {
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  const msgDiv = document.createElement('div');
  msgDiv.className = 'msg';
  msgDiv.innerHTML = `<div class="prompt">You: ${prompt}</div><div class="response">Generating...</div>`;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
  promptInput.value = "";
  sendBtn.disabled = true;
  statusEl.textContent = "";

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: prompt, max_new_tokens: 80, temperature: 0.8 })
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();
    msgDiv.querySelector('.response').textContent = data.generated || JSON.stringify(data);
  } catch (err) {
    msgDiv.querySelector('.response').textContent = "Error: " + err.message;
    statusEl.textContent = "Request failed.";
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', sendPrompt);
promptInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendPrompt(); });
</script>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    idx = torch.tensor([_tokenizer.encode(req.prompt)], device=_device)
    out = _model.generate(idx, max_new_tokens=req.max_new_tokens, temperature=req.temperature)
    text = _tokenizer.decode(out[0].tolist())
    return {"prompt": req.prompt, "generated": text}
