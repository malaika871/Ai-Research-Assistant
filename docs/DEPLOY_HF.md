# Deploy to Hugging Face Spaces

Build the Docker Space bundle from the repository root:

```bash
python scripts/make_hf_bundle.py
```

In the Space Settings, add these secrets by name:

- `HF_TOKEN` (required for hosted Hugging Face inference)
- `TAVILY_API_KEY` (optional; enables web search fallback)
- `API_KEYS` (optional; only when API-key authentication is enabled)

Publish the generated folder to a Space you have created:

```bash
cd hf_deploy
git init
git lfs install
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git add .
git commit -m "Deploy AI Research Assistant"
git push space main
```
