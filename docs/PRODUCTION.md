# Multilingual Scientific RAG System - Production Deployment Guide

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Docker and Docker Compose installed
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Deploy with Docker

```bash
# 1. Clone/download the repository
cd d:/RAG

# 2. Set your API key
echo "LLM_API_KEY=your-gemini-api-key-here" > .env

# 3. Build and start
docker-compose up -d

# 4. Check status
docker-compose logs -f
```

**API will be available at:** `http://localhost:8000`

---

## 📚 API Documentation

### Interactive Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Example Requests

#### Query Endpoint

```bash
# English query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the treatment for diabetes?",
    "strategy": "A",
    "top_k": 5
  }'

# Hindi query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "मधुमेह का इलाज क्या है?",
    "strategy": "A"
  }'
```

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Statistics

```bash
curl http://localhost:8000/stats
```

---

## 🔐 Security

### API Key Authentication (Optional)

Enable authentication by setting API keys:

```bash
# In .env file
API_KEYS=key1,key2,key3
```

Then include the key in requests:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "X-API-Key: key1" \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "strategy": "A"}'
```

### Environment Variables

```bash
# Required
LLM_API_KEY=your-gemini-api-key

# Optional
LLM_MODEL_NAME=gemini-2.5-flash
LOG_LEVEL=INFO
API_KEYS=comma,separated,keys
```

---

## 📁 Data Management

### Adding Documents

Place PDF files in the `papers/` directory:

```bash
# Copy PDFs to papers directory
cp /path/to/papers/*.pdf ./papers/

# Ingest via API
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "your-paper.pdf"}'
```

### Or use CLI:

```bash
docker-compose exec rag-api python ingest.py
```

---

## 🔧 Configuration

### Docker Compose

Edit `docker-compose.yml` to customize:

```yaml
services:
  rag-api:
    ports:
      - "8000:8000"  # Change port here
    environment:
      - LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
    volumes:
      - ./papers:/app/papers
      - ./chroma_db:/app/chroma_db
      - ./models:/app/models
```

### Model Selection

Choose Gemini model in `.env`:

```bash
# Fast and cheap (recommended)
LLM_MODEL_NAME=gemini-2.5-flash

# Higher quality
LLM_MODEL_NAME=gemini-2.5-pro
```

---

## 📊 Monitoring

### Logs

```bash
# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f rag-api
```

### Health Monitoring

```bash
# Check health
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2024-11-22T08:00:00",
  "version": "1.0.0",
  "gemini_configured": true
}
```

### Metrics

Access statistics:

```bash
curl http://localhost:8000/stats
```

---

## 🛠️ Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs

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

# Or use smaller models
LLM_MODEL_NAME=gemini-2.5-flash
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
  --ports 8000 \
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
  rag-api:
    deploy:
      replicas: 3
```

### Vertical Scaling

```yaml
services:
  rag-api:
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

### Gemini API Costs

- **gemini-2.5-flash**: ~$0.0001 per query
- **gemini-2.5-pro**: ~$0.002 per query

**Monitor usage:**
```bash
# Check logs for API calls
docker-compose logs | grep "Generating answer"
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
4. **"Out of memory"**: Use `gemini-2.5-flash` model, increase Docker memory

### Getting Help

- Check logs: `docker-compose logs -f`
- Review documentation: `/docs` endpoint
- Check health: `/health` endpoint

---

**Your multilingual scientific RAG system is production-ready! 🚀**

For generic research use, anyone can:
1. Deploy with Docker in 5 minutes
2. Add their own PDF papers
3. Query in any Indian language
4. Get accurate, cited answers
