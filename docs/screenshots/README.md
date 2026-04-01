# Screenshots — OncoTriage AI

This directory should contain screenshots of the live application for use in the README and portfolio.

**Add these screenshots before making the repository public:**

---

## Required Screenshots

| Filename | What to Capture |
|---|---|
| `01_home_screen.png` | The landing state of the app — hero title, disclaimer box, empty Single Variant tab, sidebar with model settings |
| `02_single_variant_scored.png` | A complete scoreed result — risk gauge, metrics row (Δ LL, Review Priority, Elapsed), review signal label, annotation, provenance expander open |
| `03_brca_lookup.png` | The BRCA Exchange Database Lookup expander open with a real or demo result |
| `04_batch_scoring.png` | Batch tab with a results table displayed after scoring a sample CSV |
| `05_generate_tab.png` | Generate tab with a seed sequence and the generated output visible |
| `06_clinvar_explorer.png` | ClinVar Explorer tab showing a fetched variant table |
| `07_demo_mode_banner.png` | Demo mode active — the yellow warning banner at the top of the screen |
| `08_live_fallback_warning.png` | The orange fallback warning that appears when live inference is unavailable |

---

## How to Capture

1. Run the app locally: `streamlit run app.py`
2. For **live mode** screenshots, set `NVIDIA_API_KEY` in `.env`
3. For **demo mode** screenshots, run without a key
4. Use a browser zoom of 90–100% and a window width of ~1400px for consistency
5. Crop to the app content — exclude browser chrome

---

## Embedding in README

Once screenshots are added, update the `## 📸 Screenshots` section in `README.md`:

```markdown
## 📸 Screenshots

![Home screen](docs/screenshots/01_home_screen.png)
*Home screen — demo mode active, no API key required*

![Scored result](docs/screenshots/02_single_variant_scored.png)
*Single variant scored — review priority gauge and provenance metadata*

![Batch scoring](docs/screenshots/04_batch_scoring.png)
*Batch scoring results table with CSV download*
```
