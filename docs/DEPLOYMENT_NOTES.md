# Deployment Notes — OncoTriage AI

---

## Running Locally

```bash
# 1. Clone and set up a virtual environment
git clone https://github.com/Daniyal0100101/oncotriage-ai.git
cd oncotriage-ai
python -m venv .venv
.venv\Scripts\Activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set your API key
cp .env.example .env
# Edit .env: NVIDIA_API_KEY=nvapi-...

# 4. Run
streamlit run app.py
```

The app launches in **DEMO MODE** if `NVIDIA_API_KEY` is absent. All demo-mode outputs are clearly labeled.

---

## Handling Secrets

| Context | Method |
|---|---|
| Local | `.env` file (never commit — excluded by `.gitignore`) |
| Hugging Face Spaces | Space Settings → Secrets → add `NVIDIA_API_KEY` |
| Docker | `-e NVIDIA_API_KEY=nvapi-...` runtime flag |
| CI/CD | GitHub Actions Secrets |

**Never hardcode API keys in source files or Dockerfiles.**

---

## Hugging Face Spaces (Docker SDK)

The `Dockerfile` is pre-configured for Hugging Face Spaces:

1. Create a new Space → choose **Docker** SDK.
2. Connect this GitHub repository.
3. In **Space Settings → Secrets**, add `NVIDIA_API_KEY`.
4. The app detects the key at runtime and switches to LIVE MODE automatically.

Port `7860` is the required HF Spaces port and is already exposed.

---

## Local Docker

```bash
docker build -t oncotriage .
docker run -p 7860:7860 -e NVIDIA_API_KEY=nvapi-... oncotriage
# Open http://localhost:7860
```

To run in demo mode without a key:
```bash
docker run -p 7860:7860 oncotriage
```

---

## Live Inference Caveat

The hosted NVIDIA NIM platform (`health.api.nvidia.com`) **does not expose the `/forward` logits endpoint**. Attempting to use it returns an internal `StripedHyena has no attribute output_layer` error.

**Default behaviour** (`EVO2_USE_FORWARD_PASS=false`):
- Scoring uses the `/generate` endpoint with `enable_logits=True` and `num_tokens=1` as a log-likelihood proxy.
- This is a functional approximation, not direct per-residue logit scoring.

**Private NIM deployment** (`EVO2_USE_FORWARD_PASS=true`):
- Uses the full `/forward` path for direct logit extraction.
- Only set this on a private NIM deployment that actually exposes the endpoint.

Both paths fall back to demo mode gracefully on error.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | — | NVIDIA NIM API key. App runs in demo mode without it. |
| `EVO2_USE_FORWARD_PASS` | `false` | Enable `/forward` path. Leave `false` for hosted NVIDIA NIM. |
