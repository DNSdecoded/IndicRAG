# Multilingual Scientific RAG System - Production Deployment

This guide supplements `DEPLOY.md` to provide comprehensive details on deploying IndicRAG in a production environment.

---

## 🔐 1. API Key Authentication

API Key authentication secures your endpoints. It is enabled by default in production.

1. Open your `.env` file.
2. Define `API_KEYS` with a comma-separated list of your secure keys:
   ```env
   API_KEYS=your_secure_key_1,your_secure_key_2
   ```
3. All clients must include the following header when making API requests:
   ```http
   X-API-Key: your_secure_key_1
   ```

*(Note: If `API_KEYS` is left empty or commented out, the API will be completely open. This is strictly meant for local development.)*

---

## 📈 2. Monitoring (Prometheus Metrics)

The FastAPI server is instrumented with **Prometheus FastAPI Instrumentator**.

- Metrics are automatically tracked for all endpoints (latency, request counts, errors).
- Access the metrics endpoint at: `http://localhost:8000/metrics`
- Configure your centralized Prometheus scraper to pull from this `/metrics` endpoint.

---

## 🌐 3. HTTPS Reverse Proxy (Nginx)

To secure traffic to your FastAPI server, use a reverse proxy like Nginx to handle SSL/TLS termination. 

We have provided a sample configuration file at `deploy/nginx.example.conf`.

1. Copy the example configuration to your Nginx sites directory:
   ```bash
   sudo cp deploy/nginx.example.conf /etc/nginx/sites-available/indicrag
   sudo ln -s /etc/nginx/sites-available/indicrag /etc/nginx/sites-enabled/
   ```
2. Obtain an SSL certificate using Let's Encrypt (Certbot):
   ```bash
   sudo certbot --nginx -d api.indicrag.com
   ```
3. Restart Nginx:
   ```bash
   sudo systemctl restart nginx
   ```

---

## 🖥️ 4. Deploying as a Windows Service

Since Windows does not natively use `systemd`, `nssm` (Non-Sucking Service Manager) is the easiest way to run IndicRAG as a background Windows Service.

1. **Download NSSM**: 
   Get the latest executable from [nssm.cc](http://nssm.cc/).
2. **Open Command Prompt as Administrator** and run:
   ```cmd
   nssm install IndicRAG
   ```
3. **Configure the Service**:
   - **Path**: Path to your Python executable (e.g., `C:\Python311\python.exe` or the path in your virtual environment `d:\IndicRAG\venv\Scripts\python.exe`).
   - **Arguments**: `start_server.py --port 8000`
   - **Details tab**: Set a description and startup type (Automatic).
   - **AppDirectory**: `d:\IndicRAG`
   - **Environment tab**: (Optional) define local environment variables, though `.env` will be picked up automatically.
4. **Click "Install Service"**.
5. **Start the Service**:
   ```cmd
   nssm start IndicRAG
   ```
   
You can now manage the API via the standard Windows Services interface (`services.msc`).

---

## ☁️ 5. Cloud Deployment (AWS / Azure)

### AWS EC2 or Azure VM
1. Provision a VM (Ubuntu 22.04 LTS recommended).
2. Install Python 3.11+, Git, and Nginx.
3. Clone the repository and install dependencies (`pip install -r requirements.txt`).
4. Follow the **Nginx** instructions above.
5. Setup a `systemd` service:

   ```ini
   [Unit]
   Description=IndicRAG API Server
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/IndicRAG
   ExecStart=/usr/bin/python3 start_server.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```
   Save to `/etc/systemd/system/indicrag.service`, then `sudo systemctl enable --now indicrag`.
