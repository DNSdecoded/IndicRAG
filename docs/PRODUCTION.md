# Multilingual Scientific RAG System - Production Deployment Guide

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Docker and Docker Compose installed
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Deploy with Docker

```bash
# 1. Clone the repository
git clone https://github.com/DNSdecoded/IndicRAG.git
cd IndicRAG

# 2. Set your keys
cp .env.example .env
#    LLM_API_KEYS=...   (Gemini)
#    API_KEYS=...       (endpoint auth — required in production mode)
#    ADMIN_API_KEY=...  (guards DELETE /purge/*)

# 3. Build and start
docker compose up -d

# 4. Check status
docker compose logs -f
```

The compose service is named `indicrag` and publishes `8080:8080`. First boot
downloads BGE-M3, the reranker and the NLI model, so the healthcheck allows a
120s `start_period`.

**API will be available at:** `http://localhost:8080`

---

## 📚 API Documentation

### Interactive Docs
- **Swagger UI**: http://localhost:8080/api/docs
- **ReDoc**: http://localhost:8080/api/redoc

### Example Requests

#### Query Endpoint

`API_KEYS` is set in production, so every example below sends `X-API-Key`.

```bash
# English query
curl -X POST "http://localhost:8080/query" \
  -H "X-API-Key: key1" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the treatment for diabetes?",
    "strategy": "A",
    "top_k": 5
  }'

# Hindi query
curl -X POST "http://localhost:8080/query" \
  -H "X-API-Key: key1" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "मधुमेह का इलाज क्या है?",
    "strategy": "A"
  }'
```

#### Health Check

```bash
curl -H "X-API-Key: key1" http://localhost:8080/health
```

#### Statistics

```bash
curl -H "X-API-Key: key1" http://localhost:8080/stats
```

---

## 🔐 Security

### API Key Authentication (required in production)

The server binds `0.0.0.0`. Without `API_KEYS` every endpoint — including the
destructive `/purge/*` routes — accepts anonymous requests, so
`start_server.py` refuses to boot in production mode until it is set
(`ALLOW_UNAUTHENTICATED=1` opts out on a private host; `--dev` only warns).

```bash
# In .env file
API_KEYS=key1,key2,key3
```

Then include the key in requests:

```bash
curl -X POST "http://localhost:8080/query" \
  -H "X-API-Key: key1" \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "strategy": "A"}'
```

### Environment Variables

```bash
# Required
LLM_API_KEYS=key1,key2          # comma-separated, load balanced (LLM_API_KEY also read)
API_KEYS=comma,separated,keys   # endpoint auth

# Optional
LLM_MODEL_NAME=gemini-3.6-flash
OPENROUTER_API_KEY=sk-or-...    # only for `vendor/model` slugs
ADMIN_API_KEY=admin-only-key
LOG_LEVEL=INFO
```

`.env.example` documents the full set; defaults live in `config.py`.

---

## 📁 Data Management

### Adding Documents

Place PDF files in the `papers/` directory:

```bash
# Copy PDFs to papers directory
cp /path/to/papers/*.pdf ./papers/

# Ingest via API (paths resolve inside papers/)
curl -X POST "http://localhost:8080/ingest" \
  -H "X-API-Key: key1" \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "your-paper.pdf"}'

# Or ingest everything in papers/ as a background job
curl -X POST "http://localhost:8080/ingest/all" -H "X-API-Key: key1"
```

Or upload straight from the web UI's **Library** view, which also supports
ingesting a PDF by URL (`POST /ingest/from-url`).

### Or use CLI:

```bash
docker compose exec indicrag python ingest.py papers/
```

---

## 🔧 Configuration

### Docker Compose

Edit `docker-compose.yml` to customize:

```yaml
services:
  indicrag:
    ports:
      - "8080:8080"  # Change the host side here
    env_file:
      - .env
    volumes:
      - ./models:/app/models
      - ./chroma_db:/app/chroma_db
      - ./papers:/app/papers
```

The container runs as UID 10001, so the bind-mounted `models/`, `chroma_db/`
and `papers/` directories must be writable by that UID on the host
(`chown -R 10001 models chroma_db papers`).

### Model Selection

`LLM_MODEL_NAME` sets the default. `LLM_SELECTABLE_MODELS` is the
comma-separated dropdown offered to the UI — its first entry is the default,
and it must keep at least one `vendor/model` slug, because cross-vendor
failover picks the first slug in that list.

```bash
LLM_MODEL_NAME=gemini-3.6-flash
LLM_SELECTABLE_MODELS=gemini-3.6-flash,gemini-3.5-flash,anthropic/claude-haiku
```

A bare name routes to Gemini; anything containing `/` routes to OpenRouter and
requires `OPENROUTER_API_KEY`.

---

## 📊 Monitoring

### Logs

```bash
# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f indicrag
```

### Health Monitoring

```bash
# Check health
curl http://localhost:8080/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2026-01-01T00:00:00",
  "version": "...",
  "gemini_configured": true
}

# Component-level checks (ChromaDB, embeddings, reranker)
curl "http://localhost:8080/health?deep=true"
```

### Metrics

Access statistics:

```bash
curl http://localhost:8080/stats
```

---

## 🛠️ Troubleshooting

### Container won't start

```bash
# Check logs
docker compose logs

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### API key not working

```bash
# Verify .env file
cat .env

# Restart containers to pick up changes
docker-compose restart
```

### Out of memory

```bash
# Increase Docker memory limit
# Docker Desktop: Settings → Resources → Memory

# Or shed local models — the LLM is remote, the RAM goes to these
USE_COLBERT_RERANK=false
USE_RERANKER=false
```

### ChromaDB errors

```bash
# Reset database (WARNING: deletes all data)
rm -rf chroma_db/*
docker-compose restart
```

---

## 🌐 Cloud Deployment

### AWS (ECS/Fargate)

1. **Push image to ECR:**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag multilingual-rag:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/multilingual-rag:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/multilingual-rag:latest
```

2. **Create ECS task definition**
3. **Deploy with Fargate**
4. **Set environment variables via AWS Secrets Manager**

### Google Cloud (Cloud Run)

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/multilingual-rag

# Deploy
gcloud run deploy multilingual-rag \
  --image gcr.io/PROJECT_ID/multilingual-rag \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars LLM_API_KEY=your-key
```

### Azure (Container Instances)

```bash
# Push to ACR
az acr build --registry myregistry --image multilingual-rag .

# Deploy
az container create \
  --resource-group myResourceGroup \
  --name multilingual-rag \
  --image myregistry.azurecr.io/multilingual-rag:latest \
  --dns-name-label multilingual-rag \
  --ports 8080 \
  --environment-variables LLM_API_KEY=your-key
```

---

## 🔄 Updates & Maintenance

### Updating the System

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Backup Data

```bash
# Backup ChromaDB
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz chroma_db/

# Backup papers
tar -czf papers_backup_$(date +%Y%m%d).tar.gz papers/
```

### Restore Data

```bash
# Restore ChromaDB
tar -xzf chroma_backup_20241122.tar.gz

# Restart container
docker-compose restart
```

---

## 📈 Scaling

### Horizontal Scaling

Use a load balancer with multiple instances:

```yaml
# docker-compose-scaled.yml
services:
  indicrag:
    deploy:
      replicas: 3
```

### Vertical Scaling

```yaml
services:
  indicrag:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

---

## 💰 Cost Optimization

### LLM API Costs

Per-query cost depends entirely on the selected model and context size. Check
current per-token pricing at https://ai.google.dev/pricing (Gemini) and
https://openrouter.ai/models (everything else). Flash-class models are the
cheap default; the agentic pipeline costs several times a classic query because
it makes multiple LLM turns.

Cut cost by keeping the LLM cache enabled and lowering `top_k` /
`AGENT_MAX_CONTEXT_CHUNKS`.

**Monitor usage:**
```bash
docker compose logs | grep "Generating answer"
```

### Infrastructure Costs

- **Local**: Free (just electricity)
- **Cloud Run**: Pay per request (~$0.40 per million requests)
- **ECS Fargate**: ~$30-50/month for small deployment

---

## 🎯 Production Checklist

- [ ] API key configured and secure
- [ ] `.env` not committed to git
- [ ] Health checks working
- [ ] Logs configured and accessible
- [ ] Documents ingested successfully
- [ ] API endpoints tested
- [ ] HTTPS configured (if public)
- [ ] Backup strategy in place
- [ ] Monitoring dashboards setup
- [ ] Rate limiting configured (if needed)

---

## 📞 Support

### Common Issues

1. **"API key not configured"**: Check `.env` file exists and has correct key
2. **"ChromaDB not found"**: Run `docker-compose down && docker-compose up -d`
3. **"No documents found"**: Ingest PDFs using `/ingest` endpoint or CLI
4. **"Out of memory"**: disable `USE_RERANKER` / `USE_COLBERT_RERANK`, or increase Docker memory
5. **Server exits with "API_KEYS not set"**: set `API_KEYS` in `.env`, or `ALLOW_UNAUTHENTICATED=1` on a private host

### Getting Help

- Check logs: `docker compose logs -f`
- Review documentation: `/api/docs` endpoint
- Check health: `/health` endpoint

---

**Your multilingual scientific RAG system is production-ready! 🚀**

For generic research use, anyone can:
1. Deploy with Docker in 5 minutes
2. Add their own PDF papers
3. Query in any Indian language
4. Get accurate, cited answers
