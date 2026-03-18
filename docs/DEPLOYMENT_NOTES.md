# Deployment Notes

## Hugging Face Spaces (Docker SDK)

The repository is pre-configured with a `Dockerfile` for Hugging Face Spaces.
Port `7860` is exposed as required by HF Spaces.

### Next Steps for Cloud Deployment:
1. Create a new Space on Hugging Face (Docker SDK).
2. Connect this GitHub repository.
3. In the Space Settings, add your `NVIDIA_API_KEY` as a Secret.
4. The application will automatically detect the key at runtime and disable DEMO MODE, switching to live inference.

## Local Docker
```bash
docker build -t oncotriage .
docker run -p 7860:7860 -e NVIDIA_API_KEY=your_key oncotriage
```
