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

COPY . .

EXPOSE 8080

# start_server.py runs pre-flight checks then uvicorn on 0.0.0.0:8080.
CMD ["python", "start_server.py"]
