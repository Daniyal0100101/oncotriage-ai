# 🧬 OncoTriage AI

**Research-use hereditary cancer variant review copilot driven by the Evo 2 (40B) genomic foundation model.**

> ⚠️ **Research/Educational Use Only — Not for Clinical Diagnosis or Treatment Decisions.**  
> This application is a demonstration and research tool. It is not validated for clinical use. Results must never be used as a basis for medical decisions or patient treatment. Always consult qualified medical professionals or a certified clinical geneticist. 

---

## 📌 Project Status

**PORTFOLIO-READY | PENDING LIVE DEPLOYMENT**

OncoTriage AI is fully architected with its UI, machine-learning pipeline, and data integrations (ClinVar, BRCA Exchange). Currently, it operates in **DEMO MODE** because the live NVIDIA NIM API key for Evo 2 inference is pending. Once a key is provided, the application seamlessly switches to live inference.

## 🚀 Core Features

- **Single Variant Scoring**: Evaluates BRCA1/BRCA2 mutations via Evo 2 forward-pass log-likelihoods.
- **Batch Processing**: Score thousands of variants overnight via CSV upload.
- **Database Integration**: Real-time cross-referencing with ClinVar and BRCA Exchange.
- **Explainable UI**: Provides visualization of transition/transversion, variant impact, and regional risk gauges.
- **Auto Demo Mode**: Fully functional offline mock-inference mode allowing UI exploration without API limits.

## 🛠️ Demo Mode vs Live Mode

- **Live Mode**: When an `NVIDIA_API_KEY` is present in the `.env` file or Streamlit sidebar, the app communicates with `health.api.nvidia.com` to run 40B-parameter models on real DNA sequences.
- **Demo Mode**: If no API key is present, the application detects this securely and falls back to Demo Mode. In Demo Mode, API responses are mocked locally to demonstrate the user flow (scoring, batching, UI visualizations) without requiring external credentials.

## 📂 Repository Structure

```text
oncotriage-ai/
├── app.py                  # Main Streamlit application
├── model.py                # Evo 2 API client + ML scoring head
├── utils.py                # Variant parsing (HGVS) & visualizations
├── clinvar_data.py         # ClinVar E-utilities & BRCA Exchange wrappers
├── demo_data/              # Sample CSVs and JSON responses for Demo Mode
├── docs/                   # Screenshots, deployment notes, and project status
├── test_app.py             # Pytest test suite
├── requirements.txt        # Python dependencies
├── Dockerfile              # Hugging Face Spaces deployment container
└── .env.example            # Environment variables template
```

## ⚙️ Quickstart (Running Locally)

**1. Clone the repository**
```bash
git clone https://github.com/Daniyal0100101/oncotriage-ai.git
cd oncotriage-ai
```

**2. Setup Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the Application**
```bash
streamlit run app.py
```
*The app will automatically launch in DEMO MODE unless you provide an NVIDIA API key in the sidebar.*

## ☁️ Deployment Path

This repository is optimized for deployment via **Hugging Face Spaces (Docker SDK)**. The included `Dockerfile` securely exposes port 7860. See `docs/DEPLOYMENT_NOTES.md` for detailed instructions on configuring cloud secrets and live inference.

## 🚧 Limitations

- Currently limited to BRCA1 and BRCA2 contexts.
- Deep intronic variants are outside the base sequence context window (512bp).
- The intermediate Random Forest risk-calibrator relies on mock clinical data when operating offline.

---

**Copyright © Daniyal Asif. All rights reserved.**
*This code is provided for portfolio and review purposes only. No open-source reuse rights are granted.*
