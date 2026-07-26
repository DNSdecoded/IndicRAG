# Deployment Guide

## 🚀 Quick Deploy (3 Steps)

### 1. Install Dependencies
```bash
git clone https://github.com/DNSdecoded/IndicRAG.git
cd IndicRAG
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy example
cp .env.example .env        # copy .env.example .env  on Windows
```

Set at minimum:

```bash
LLM_API_KEYS=your-gemini-api-key   # get one at https://aistudio.google.com/app/apikey
API_KEYS=a-long-random-string      # endpoint auth — production mode refuses to start without it
ADMIN_API_KEY=another-random-string  # guards the destructive /purge/* routes
```

### 3. Start Server

```bash
python start_server.py
```

**Done!** Web UI at http://localhost:8080, interactive API docs at
http://localhost:8080/api/docs

---

## 📚 Deployment Options

### Local Development

```bash
# Development mode (auto-reload)
python start_server.py --dev

# Custom port
python start_server.py --port 9000
```

### Production Server

#### Linux/Mac Background Service

```bash
# Run as background process
nohup python start_server.py > server.log 2>&1 &

# Check if running
ps aux | grep start_server

# View logs
tail -f server.log
```

#### Windows Background Service

```bash
# Using PowerShell
Start-Process python -ArgumentList "start_server.py" -WindowStyle Hidden

# Or create a scheduled task for auto-start
```

---

### Docker

`Dockerfile` and `docker-compose.yml` ship with the repo (service name
`indicrag`, port 8080, non-root user, `models/` + `chroma_db/` + `papers/`
bind-mounted so they survive rebuilds):

```bash
cp .env.example .env    # fill in LLM_API_KEYS + API_KEYS first
docker compose up -d
docker compose logs -f
```

First boot downloads BGE-M3, the reranker and the NLI model, so the healthcheck
has a 120s `start_period`.

---

## ☁️ Cloud Deployment

### Google Cloud Run (Easiest)

```bash
# Deploy directly from source
gcloud run deploy rag-api \
  --source . \
  --set-env-vars LLM_API_KEY=your-key \
  --allow-unauthenticated
```

### AWS (EC2)

```bash
# 1. Launch EC2 instance (Ubuntu)
# 2. SSH into instance
ssh -i yourkey.pem ubuntu@your-ec2-ip

# 3. Setup
sudo apt update
sudo apt install python3-pip
git clone your-repo-url
cd RAG
pip3 install -r requirements.txt

# 4. Configure
echo "LLM_API_KEY=your-key" > .env

# 5. Run
nohup python3 start_server.py &
```

### Azure (App Service)

```bash
# Deploy using Azure CLI
az webapp up \
  --name rag-app \
  --runtime "PYTHON|3.11" \
  --sku B1

# Set environment variables
az webapp config appsettings set \
  --name rag-app \
  --settings LLM_API_KEY=your-key
```

---

## 🔧 Configuration

### Environment Variables

Edit `.env` file:

```bash
# Required
LLM_API_KEYS=key1,key2          # comma-separated; load balanced. LLM_API_KEY also accepted
API_KEYS=client-key-1,client-key-2   # endpoint auth (see below)

# Optional
LLM_MODEL_NAME=gemini-3.6-flash
OPENROUTER_API_KEY=sk-or-...    # only needed for `vendor/model` slugs
ADMIN_API_KEY=admin-only-key    # guards DELETE /purge/*
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
```

The full annotated list lives in `.env.example`; defaults are in `config.py`.

### API Authentication

The server binds `0.0.0.0`. Without `API_KEYS`, every endpoint — including the
destructive `/purge/*` routes — accepts anonymous requests, so
`start_server.py` **refuses to start in production mode** until `API_KEYS` is
set. On a private host you can opt out with `ALLOW_UNAUTHENTICATED=1`;
`--dev` mode only warns.

```bash
# In .env, add:
API_KEYS=key1,key2,key3
ADMIN_API_KEY=separate-admin-key
```

Then use in requests:
```bash
curl -H "X-API-Key: key1" http://localhost:8080/query ...
```

---

## 📊 Performance

### Expected Performance
- **Query Time**: 2-5 seconds for the classic pipeline; the agentic pipeline is
  slower (multiple LLM turns plus tool calls)
- **Concurrent Users**: 10-20 (single instance, `workers=1`)
- **Cost**: depends on the model selected per request — check current
  per-token pricing at https://ai.google.dev/pricing and
  https://openrouter.ai/models

### Scaling

#### Horizontal Scaling
```bash
# Run multiple instances on different ports
python start_server.py --port 8080 &
python start_server.py --port 8081 &
python start_server.py --port 8082 &

# Use nginx/HAProxy as load balancer
```

#### Vertical Scaling
- Increase server RAM/CPU
- Use faster embedding GPU
- Optimize ChromaDB settings

---

## 🔐 Security Best Practices

1. **API Keys**: Always use environment variables, never hardcode
2. **HTTPS**: Use reverse proxy (nginx) with SSL certificates
3. **Rate Limiting**: Implement in production
4. **Firewall**: Only expose necessary ports
5. **Updates**: Keep dependencies up to date

---

## 📈 Monitoring

### Health Checks

```bash
# Check server health
curl http://localhost:8080/health

# Expected response
{
  "status": "healthy",
  "timestamp": "...",
  "version": "...",
  "gemini_configured": true
}

# Component-level checks (ChromaDB, embeddings, reranker)
curl "http://localhost:8080/health?deep=true"

# Deep ingest-path health (store, embeddings, disk)
curl http://localhost:8080/ingest/health
```

### Logs

```bash
# View real-time logs
tail -f server.log  # Linux/Mac
Get-Content server.log -Wait  # Windows PowerShell
```

### Statistics

```bash
# Get vector store stats
curl http://localhost:8080/stats
```

---

## 🆘 Troubleshooting

### Port Already in Use

```bash
# Find process using port
netstat -ano | findstr :8080  # Windows
lsof -i :8080                 # Linux/Mac

# Kill process
taskkill /F /PID <PID>        # Windows
kill -9 <PID>                 # Linux/Mac

# Or use different port
python start_server.py --port 9000
```

### "Module not found" Errors

```bash
pip install -r requirements.txt
```

### "API key not configured"

```bash
# Check .env file exists and has key
cat .env  # Linux/Mac
type .env  # Windows
```

### Out of Memory

The LLM runs remotely, so RAM goes to the local models. Trim them in `.env`:

```bash
USE_COLBERT_RERANK=false   # drops the ColBERT MaxSim pass
USE_RERANKER=false         # drops the cross-encoder (~2GB)
```

Or increase server memory.

### ChromaDB Errors

```bash
# Reset database (WARNING: deletes all data)
rm -rf chroma_db/*  # Linux/Mac
Remove-Item -Recurse -Force chroma_db\*  # Windows

# Restart server
python start_server.py
```

---

## 🔄 Updates & Maintenance

### Update System

```bash
# Pull latest code
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart server
# (stop current server with Ctrl+C, then)
python start_server.py
```

### Backup Data

Three things need backing up. `sessions.db` is the one people forget — it holds
**all** user state (sessions, chat history, feedback, watches, saved reports,
job records). Losing it loses everything except the corpus.

```bash
# Stop the server first, or at least accept a slightly stale snapshot: the DB
# runs in WAL mode, so the -wal/-shm sidecars matter.
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz chroma_db/     # vector store
tar -czf papers_backup_$(date +%Y%m%d).tar.gz papers/        # source PDFs
sqlite3 sessions.db ".backup 'sessions_$(date +%Y%m%d).db'"  # user state
```

`sqlite3 .backup` is safe against a running server; a plain `cp` of
`sessions.db` without its `-wal` file can produce a torn snapshot.

---

## ⏪ Rollback

Take the backups above **before** deploying. Then:

```bash
# 1. Put the previous release back
git checkout v2.3.0        # or the tag/commit you were running
pip install -r requirements.txt

# 2. Restore state captured before the upgrade
tar -xzf chroma_backup_YYYYMMDD.tar.gz
cp sessions_YYYYMMDD.db sessions.db

# 3. Restart
python start_server.py
```

**What rolls back cleanly and what doesn't:**

- **Vector store and papers** — restore from the tarballs above, no caveats.
- **`sessions.db`** — schema changes are additive (`CREATE TABLE IF NOT EXISTS`,
  no `ALTER`), so a newer database keeps working on an older release; the older
  code simply ignores tables it doesn't know about. Rows written by the newer
  release into new tables (e.g. `reports`, `query_log`) are invisible to it.
  There is no schema version marker, so verify this yourself before relying on
  it across a release that adds columns rather than tables.
- **Configuration** — `.env` is not versioned. Keep a copy alongside the
  backups; a rollback that keeps a newer `.env` can start the old code with
  variables it doesn't understand.

### Upgrading to 2.4.0

One breaking change: **`/purge/*` is now fail-closed.** It previously accepted
any key in `API_KEYS` when `ADMIN_API_KEY` was unset, and accepted anonymous
requests when `API_KEYS` was empty. Both are gone.

Before deploying 2.4.0, set a dedicated admin key:

```bash
ADMIN_API_KEY=a-separate-long-random-string
```

Without it, every `/purge/*` request returns
`403 ADMIN_KEY_NOT_CONFIGURED` — including automation that used to purge with an
ordinary user key.

---

## ✅ Production Checklist

- [ ] Python 3.11+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Gemini API key configured in `.env`
- [ ] Documents ingested (`python examples/example_ingest.py`)
- [ ] Server starts successfully
- [ ] Health check passes (`curl localhost:8080/health`)
- [ ] Can query via web UI (http://localhost:8080)
- [ ] (Optional) API authentication configured
- [ ] (Optional) HTTPS/SSL configured
- [ ] (Optional) Monitoring setup

---

**Your multilingual RAG system is ready for production deployment!**

For quick start, see [QUICKSTART.md](QUICKSTART.md)  
For technical details, see [ARCHITECTURE.md](ARCHITECTURE.md)
