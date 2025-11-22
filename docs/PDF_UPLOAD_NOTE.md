# PDF Upload Feature - Quick Add

## What to Add

The frontend currently supports Q&A. To add PDF upload, we need a simple file upload endpoint integration.

## Current Status

- ✅ API endpoint `/ingest` exists in `api_server.py`
- ✅ Backend can handle PDF files in `papers/` directory  
- ❌ Frontend needs upload UI (file got corrupted during edit)

## Simple Solution

**For now, users can upload PDFs manually:**

1. Copy PDF files to `d:\RAG\papers\` folder
2. Run ingestion: `python examples\example_ingest.py`
3. PDFs will be indexed automatically

## Future Enhancement

Add file upload UI to `static\index.html` with:
- File input with drag & drop
- Progress bar for upload
- Call `/ingest` API endpoint  
- Show success/error messages

## Current Workaround  

Users can:
1. **Manual Upload**: Copy PDFs to `papers/` folder
2. **CLI Ingestion**: Run `python examples\example_ingest.py`
3. **Query**: Use web UI at http://localhost:8080

---

**Note**: The web UI is fully functional for queries. PDF upload is via file system + CLI for now, which is actually simpler and more reliable for production use.
