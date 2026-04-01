# OncoTriage AI — Roadmap

> All milestones are research and engineering goals only. No item on this roadmap implies clinical approval, regulatory clearance, or production medical use.

---

## 🎨 Product Polish

- [x] Demo mode fallback with clear DEMO labeling throughout the UI
- [x] Live → demo fallback on API errors with user-visible warning
- [x] Provenance metadata on every result (model, path, mode, fallback status)
- [x] BRCA Exchange lookup decoupled from Evo 2 inference
- [x] Batch CSV scoring with downloadable results
- [x] Sequence generation tab (Evo 2 `/generate` endpoint)
- [x] Add screenshots to `docs/screenshots/` before public launch
- [ ] Mobile / narrow-viewport layout review
- [ ] Improve batch error reporting (per-row error details column)

---

## 🔬 Scientific Validation

- [x] Synthetic training data pipeline for the RF ranking head (offline demo)
- [ ] Fetch real Evo 2 log-likelihoods for known ClinVar pathogenic / benign variants
- [ ] Retrain RF ranking head on real ClinVar-scored data
- [ ] Benchmark review priority scores against ClinVar ground-truth labels (AUC, calibration)
- [ ] Validate generate-proxy scoring approach against forward-pass logits on a private NIM deployment
- [ ] Disclose training data provenance clearly in app and docs once real data is integrated
- [ ] Extend beyond BRCA1/BRCA2 to additional hereditary cancer genes (MLH1, MSH2, PALB2, etc.)

---

## ☁️ Deployment

- [x] Dockerfile configured for Hugging Face Spaces (Docker SDK, port 7860)
- [x] EVO2_USE_FORWARD_PASS feature flag for private NIM deployments
- [ ] Deploy interactive demo to Hugging Face Spaces
- [ ] Add GitHub Actions CI: lint + pytest on push
- [ ] Pin dependency versions in `requirements.txt` for reproducibility
- [ ] Evaluate Streamlit Community Cloud as an alternative hosting path

---

## 🏥 Institutional Pilot Readiness

- [ ] Engage genetic counseling researchers for usability feedback
- [ ] Add audit logging (which variants were scored, no patient data)
- [ ] Integrate PubMed / literature retrieval to surface supporting evidence alongside scores
- [ ] Formal IRB / ethical review scoping if used with any real biological materials
- [ ] Write a brief methods note describing the scoring approach for use in research context

---

*This roadmap reflects the current prototype stage. Items may be re-prioritized as feedback is collected.*
