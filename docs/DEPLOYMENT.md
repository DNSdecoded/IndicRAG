# Deployment Guide

## 🚀 Quick Deploy (3 Steps)

### 1. Install Dependencies
```bash
cd d:/RAG
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy example
copy .env.example .env

# Edit .env and add your Gemini API key
# Get key from: https://makersuite.google.com/app/apikey
```

### 3. Start Server

```bash
python start_server.py
```

**Done!** Access at http://localhost:8080

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
LLM_API_KEY=your-gemini-api-key-here

# Optional
LLM_MODEL_NAME=gemini-2.5-flash    # or gemini-2.5-pro
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR
```

### API Authentication (Optional)

Enable API key authentication:

```bash
# In .env, add:
API_KEYS=key1,key2,key3
```

Then use in requests:
```bash
curl -H "X-API-Key: key1" http://localhost:8080/query ...
```

---

## 📊 Performance

### Expected Performance
- **Query Time**: 2-5 seconds
- **Concurrent Users**: 10-20 (single instance)
- **Cost**: ~$0.0001 per query (Gemini 2.5 Flash)

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
  "gemini_configured": true
}
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

```bash
# Use smaller model
LLM_MODEL_NAME=gemini-2.5-flash

# Or increase server memory
```

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

```bash
# Backup vector database
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz chroma_db/  # Linux/Mac

# Backup papers
tar -czf papers_backup_$(date +%Y%m%d).tar.gz papers/
```

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
