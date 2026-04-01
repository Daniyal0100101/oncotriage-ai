# Security Policy

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.
Contact the repository owner directly via the email address on their GitHub profile.

---

## Data Privacy

### 🚫 No Patient Data
This application is intended for **research use only**.

- Do **not** input, upload, or process any identifiable patient data (PHI / PII) of any kind.
- Do **not** use real clinical sequence data from patients without appropriate ethical clearance, IRB approval, and institutional data-use agreements.
- This tool has no user authentication, no data isolation, and no audit logging suitable for handling protected health information.

### 🚫 No PHI in Sequences
Genomic sequences entered into this application are processed in-memory and sent to the NVIDIA NIM API for inference. They are not stored by this application, but:
- NVIDIA's data retention and processing policies apply to any data sent to their hosted API.
- Review NVIDIA's terms of service before submitting any non-public sequence data.

---

## Secrets and API Keys

### 🚫 Never Commit Secrets
- Never commit your `.env` file, `NVIDIA_API_KEY`, or any other credentials to the repository.
- The `.gitignore` is pre-configured to exclude `.env`, but always verify before committing.
- API keys entered via the Streamlit sidebar are held in session state only and are never written to disk by this application.

### Storing Secrets Safely
| Environment | Recommended approach |
|---|---|
| Local development | `.env` file (excluded from git via `.gitignore`) |
| Hugging Face Spaces | Space Secrets (Settings → Secrets) |
| Docker | `-e NVIDIA_API_KEY=...` runtime flag or a secrets manager |
| CI/CD | GitHub Actions Secrets |

---

## Dependency Security

- Dependencies are listed in `requirements.txt`. Pin versions before any production or institutional deployment.
- Run `pip audit` or `safety check` periodically to detect known vulnerabilities in dependencies.

---

## Scope of This Project

OncoTriage AI is a **research prototype**. It is not a medical device and has not undergone any security review appropriate for clinical or regulated environments. Do not deploy it in settings where:
- Real patient data may be submitted.
- Outputs could influence clinical, diagnostic, or treatment decisions.
- Uptime or access-control guarantees are required.
