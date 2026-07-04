# IndicRAG — FastAPI + BGE-M3 + ChromaDB. Models download at first run into /app/models
# (mount it as a volume so they persist across container rebuilds — see docker-compose.yml).
FROM python:3.11-slim

# libgomp1 is needed by torch/onnxruntime; the rest keep the image slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer caches when only source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as a non-root user to limit blast radius of any RCE. The bind-mounted
# data dirs (models/, chroma_db/, papers/ — see docker-compose.yml) must be
# writable by this UID; chown them on the host or run `chown -R 10001` there.
RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser . .
RUN mkdir -p /app/models /app/chroma_db /app/papers \
    && chown -R appuser:appuser /app/models /app/chroma_db /app/papers
USER appuser

EXPOSE 8080

# start_server.py runs pre-flight checks then uvicorn on 0.0.0.0:8080.
CMD ["python", "start_server.py"]
