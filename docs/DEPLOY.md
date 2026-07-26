# Deploy — see DEPLOYMENT.md

This file used to duplicate the deployment guide. It drifted out of date (port
8000, `/docs`, root-level example scripts) while the real guide moved on, so it
is now a pointer instead of a second copy.

- **Setup, run, cloud deploy, troubleshooting** → [DEPLOYMENT.md](DEPLOYMENT.md)
- **Production hardening (auth, HTTPS, backups)** → [PRODUCTION.md](PRODUCTION.md)
- **5-minute first run** → [QUICKSTART.md](QUICKSTART.md)

Quick reminder of the three steps:

```bash
pip install -r requirements.txt
cp .env.example .env        # add LLM_API_KEYS + API_KEYS
python start_server.py      # http://localhost:8080
```
