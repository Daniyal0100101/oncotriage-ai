# 🧬 OncoTriage AI

**Research-use hereditary cancer variant review copilot powered by the Evo 2 (40B) genomic foundation model.**

> ⚠️ **Research / Educational Use Only — Not for Clinical Diagnosis or Treatment Decisions.**
> This application is a prototype research tool. Outputs have **not** been clinically validated and must
> never be used as a basis for medical decisions, diagnostic conclusions, or patient treatment.
> Always consult a qualified medical professional or certified clinical geneticist.

---

## 📌 Overview

OncoTriage AI is a **clinician-support style prototype** that helps researchers triage hereditary cancer variants (BRCA1 / BRCA2) by combining:

- **Evo 2 foundation-model inference** — derives a Δlog-likelihood between reference and alternate DNA sequences as a proxy signal for sequence-model disruption.
- **A lightweight RandomForest ranking head** — maps Evo 2 scoring features to a *variant review priority score* (0–1). Currently trained on **synthetic** ClinVar-like data; real ClinVar validation is the next scientific milestone.
- **External database lookups** — cross-references ClinVar and BRCA Exchange independently of model inference.

The tool is built for researchers who want an evidence-oriented triage workflow — not a diagnostic oracle.

---

## 🚀 Why It Exists

Classifying variants of uncertain significance (VUS) in hereditary cancer genes is slow, resource-intensive, and inconsistently prioritized. OncoTriage AI explores whether a large genomic foundation model (Evo 2, 40B parameters) can surface preliminary review-priority signals that help researchers decide which variants to examine further — without replacing expert clinical classification.

---

## 📊 Current Status

| Area | Status |
|---|---|
| UI and Streamlit shell | ✅ Complete |
| Evo 2 live inference (`/generate` proxy path) | ✅ Working |
| Demo mode fallback | ✅ Working |
| RF ranking head (synthetic training data) | ✅ Working — synthetic only |
| ClinVar / BRCA Exchange lookup | ✅ Working |
| Scientific validation against real ClinVar labels | ⏳ Not yet performed |
| `/forward` logits endpoint | ⚠️ Disabled (not exposed by hosted NVIDIA NIM) |
| Institutional or clinical pilot | ⏳ Not yet |

---

## ✨ Key Features

- **Single Variant Scoring** — Enter a BRCA1/BRCA2 variant in HGVS or positional notation; receive an Evo 2-derived review priority score with a risk gauge and sequence-context visualization.
- **Batch Scoring** — Upload a CSV of variants; the app scores each row and returns a ranked table for download.
- **ClinVar Explorer** — Fetch live variant summaries directly from NCBI ClinVar E-utilities.
- **BRCA Exchange Lookup** — Cross-reference BRCA Exchange independently of model inference.
- **Sequence Generation Tab** — Use Evo 2's generative endpoint to explore DNA continuations from a known context sequence.
- **Result Provenance** — Every result includes inference mode, scoring path, model, and fallback status so you always know exactly what produced the output.

---

## ⚙️ Inference Modes

### 🟢 LIVE Mode
Activated when `NVIDIA_API_KEY` is present. Calls the hosted NVIDIA NIM Evo 2 endpoint (`/generate` with `enable_logits=True`) to extract a log-likelihood proxy for ref vs alt sequences. All results are labeled **LIVE**.

### 🟡 DEMO Mode
Activated automatically when no API key is found. Review priority scores are generated from a calibrated random distribution for UI demonstration only. All results are labeled **DEMO — not live predictions**.

### 🔄 Live → Demo Fallback
If a live API call fails due to an internal NVIDIA API error, the app catches the error gracefully, surfaces an inline warning, and falls back to demo estimates. Results will be labeled **DEMO (fallback)**.

> **Note on `/forward`:** The hosted NVIDIA NIM platform does not expose the `/forward` logits endpoint (it returns an internal `StripedHyena has no attribute output_layer` error). The `/forward` path is disabled by default via `EVO2_USE_FORWARD_PASS=false`. Set to `true` only on private NIM deployments that expose it.

---

## 🏗️ Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                     app.py  (Streamlit UI)                      │
│   Single Variant ─── Batch Scoring ─── Generate ─── ClinVar    │
└────────────────────────────┬────────────────────────────────────┘
                             │
               ┌─────────────▼─────────────┐
               │         model.py           │
               │  Evo2Client → /generate    │  ← LIVE path
               │  (fallback) → demo RNG     │  ← DEMO path
               │  BRCAScorer (RF ranking)   │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │         utils.py           │
               │  HGVS parsing, sequence    │
               │  mutation, Plotly charts   │
               └─────────────┬─────────────┘
                             │
               ┌─────────────▼─────────────┐
               │      clinvar_data.py       │
               │  ClinVar E-utilities       │
               │  BRCA Exchange API         │
               └────────────────────────────┘
```

The RandomForest ranking head is trained **offline** on **synthetic** ClinVar-like data using 13 features derived from Evo 2 log-likelihood outputs and sequence properties. Real ClinVar integration (scoring known variants and validating outputs) is the primary scientific milestone before any production-readiness claim.

---

## 📂 Repository Structure

```text
oncotriage-ai/
├── app.py                  # Streamlit application
├── model.py                # Evo 2 API client + RF ranking head
├── utils.py                # Variant parsing (HGVS) & visualizations
├── clinvar_data.py         # ClinVar E-utilities & BRCA Exchange wrappers
├── brca_classifier.pkl     # Pre-trained RF model (synthetic training data)
├── demo_data/              # Sample CSVs and reference sequences
├── docs/
│   ├── PROJECT_STATUS.md   # Detailed current status
│   ├── DEPLOYMENT_NOTES.md # Hosting and secrets guide
│   └── screenshots/        # UI screenshots (see guide inside)
├── test_app.py             # Pytest test suite
├── requirements.txt        # Python dependencies
├── Dockerfile              # Hugging Face Spaces / Docker deployment
├── ROADMAP.md              # Development roadmap
├── SECURITY.md             # Security and data privacy policy
└── .env.example            # Environment variable template
```

---

## 🚀 Quickstart (Local)

**1. Clone**
```bash
git clone https://github.com/Daniyal0100101/oncotriage-ai.git
cd oncotriage-ai
```

**2. Virtual environment**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\Activate
# macOS / Linux:
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment** *(optional — app runs in DEMO MODE without a key)*
```bash
cp .env.example .env
# Edit .env and add your NVIDIA NIM key:
# NVIDIA_API_KEY=nvapi-...
```

**5. Run**
```bash
streamlit run app.py
```

The app launches in **DEMO MODE** if no key is present, or **LIVE MODE** once `NVIDIA_API_KEY` is set.

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `NVIDIA_API_KEY` | No | — | NVIDIA NIM API key (`nvapi-...`). Get one free at [build.nvidia.com](https://build.nvidia.com/arc/evo2-40b). App runs in demo mode without it. |
| `EVO2_USE_FORWARD_PASS` | No | `false` | Set `true` only for private NIM deployments that expose `/forward`. Leave `false` for hosted NVIDIA NIM. |

---

## ☁️ Deployment

See [`docs/DEPLOYMENT_NOTES.md`](docs/DEPLOYMENT_NOTES.md) for full instructions. The included `Dockerfile` is pre-configured for **Hugging Face Spaces (Docker SDK)** on port 7860.

Quick local Docker run:
```bash
docker build -t oncotriage .
docker run -p 7860:7860 -e NVIDIA_API_KEY=nvapi-... oncotriage
```

---

## 📸 Screenshots

> **Done:** Add screenshots to `docs/screenshots/`.
> See [`docs/screenshots/README.md`](docs/screenshots/README.md) for the required list and capture guidance.

---

## ⚠️ Safety & Medical Disclaimer

This tool:

- Is **not a diagnostic device** and produces no clinically validated pathogenicity determinations.
- Computes a *review priority score* — a research-oriented signal for helping investigators decide which variants to examine further.
- Has **not** been reviewed, approved, or cleared by any regulatory body.
- Must **not** be used to make any medical, treatment, or diagnostic decisions.
- Should **never** process or store identifiable patient health information (PHI / PII).

All result outputs carry an inline disclaimer and are labeled with their inference mode (LIVE / DEMO).

---

## 🚧 Known Limitations

- Variant coverage limited to BRCA1 and BRCA2 contexts.
- The RF ranking head is trained on **synthetic** data only; validation against real ClinVar Evo 2-scored variants has not been performed.
- Deep intronic variants (beyond the 512 bp context window) may not be adequately captured.
- The `/forward` logits endpoint is not available on hosted NVIDIA NIM; scoring uses a generate-proxy approach with single-token logits.
- Demo mode review priority scores are sampled from a calibrated distribution — they do not reflect real model outputs.
- No multi-user authentication or data isolation; do not host with shared access to sensitive sequences.

---

## 🗺️ Roadmap

See [`ROADMAP.md`](ROADMAP.md) for milestones across product polish, scientific validation, deployment, and institutional pilot readiness.

---

## 📄 License & Rights

Copyright © Daniyal Asif. All rights reserved.
This code is provided for portfolio review and research demonstration purposes. No open-source reuse rights are granted unless explicitly stated.
