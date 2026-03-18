# Security Policy

## Reporting a Vulnerability

Please do not open a public issue for security vulnerabilities. Instead, contact the repository owner directly.

## Data Privacy and API Security

- **NO PATIENT DATA**: This application is for research use only. Do not input, upload, or process any identifiable patient data (PHI/PII).
- **NO SECRETS IN COMMITS**: Never commit your `.env` file, `NVIDIA_API_KEY`, or any other credentials to the repository. The `.gitignore` is configured to prevent this, but manual care must be taken.
- **API Keys**: Access to external APIs (like NVIDIA NIM) relies on user-provided keys, which are not stored by this application persistently.
