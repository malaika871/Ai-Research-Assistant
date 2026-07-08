FROM python:3.11-slim

WORKDIR /app

# System deps needed by pymupdf / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces runs containers as a non-root user with no write access to /app by
# default, and always maps the app to port 7860 — so we create writable dirs
# and expose 7860 instead of 8000.
RUN mkdir -p uploads chroma_db && \
    useradd -m -u 1000 appuser && \
    chown -R appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]