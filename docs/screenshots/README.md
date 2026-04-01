# Screenshots - OncoTriage AI

This directory contains the screenshot checklist for the live application assets referenced by the README and roadmap.

**Completed:** Add screenshots to `docs/screenshots/`.

---

## Release Screenshots

| Filename | What to Capture |
|---|---|
| `home.png` | Landing state of the app with the sidebar visible |
| `single-variant.png` | Single-variant LIVE result with review-priority visualization, metrics, and annotation |
| `batch.png` | Batch scoring tab with uploaded CSV workflow visible |
| `generate.png` | Generate tab with generated output shown |
| `07_demo_mode_banner.png` | Demo mode banner reference image kept from earlier documentation work |

---

## How to Capture

1. Run the app locally: `streamlit run app.py`
2. For live mode screenshots, set `NVIDIA_API_KEY` in `.env`
3. For demo mode screenshots, run without a key
4. Use a browser zoom of 90-100% and a window width of about 1400px for consistency
5. Crop to the app content and exclude browser chrome

---

## Embedding in README

The `## Screenshots` section in `README.md` should stay aligned with these release assets:

```markdown
## Screenshots

![Home](docs/screenshots/home.png)

![Single Variant Review](docs/screenshots/single-variant.png)

![Batch Scoring](docs/screenshots/batch.png)
```
