#!/usr/bin/env python3
"""
Production startup script for Multilingual RAG System.
No Docker required - just Python!
"""

import sys
import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Check Python version is 3.11+"""
    if sys.version_info < (3, 11):
        logger.error("Python 3.11+ required. Current: %s", sys.version)
        return False
    logger.info("✓ Python version: %s", sys.version.split()[0])
    return True


def check_env_file():
    """Check .env file exists"""
    if not os.path.exists('.env'):
        logger.error("✗ .env file not found!")
        logger.info("  Copy .env.example to .env and add your Gemini API key")
        return False
    logger.info("✓ .env file found")
    return True


def check_api_key():
    """Check API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    raw_keys = os.getenv('LLM_API_KEYS', '')
    api_key = os.getenv('LLM_API_KEY', '')
    keys = [k.strip() for k in raw_keys.split(',') if k.strip()] if raw_keys else []
    if not keys and api_key and api_key != 'your-gemini-api-key-here':
        keys = [api_key]
    if not keys:
        logger.error("✗ Gemini API key not configured!")
        logger.info("  Edit .env and set LLM_API_KEY or LLM_API_KEYS")
        return False
    if len(keys) > 1:
        logger.info(f"✓ Gemini API keys configured ({len(keys)} keys, load balanced)")
    else:
        logger.info("✓ Gemini API key configured")
    return True


def check_dependencies():
    """Check required packages are installed"""
    import importlib.util
    
    required = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('chromadb', 'chromadb'),
        ('sentence-transformers', 'sentence_transformers'),
        ('google-genai', 'google.genai')
    ]
    
    missing = []
    for pip_name, import_name in required:
        if importlib.util.find_spec(import_name) is None:
            missing.append(pip_name)
    
    if missing:
        logger.error("✗ Missing dependencies: %s", ', '.join(missing))
        logger.info("  Run: pip install -r requirements.txt")
        return False
    
    logger.info("✓ All dependencies installed")
    return True


def check_documents():
    """Check if documents are ingested"""
    try:
        import vector_store
        collection = vector_store.get_or_create_collection()
        stats = vector_store.get_collection_stats(collection)
        
        if stats['count'] == 0:
            logger.warning("⚠ No documents ingested yet")
            logger.info("  Add PDFs to papers/ and run: python example_ingest.py")
        else:
            logger.info("✓ %d document chunks indexed", stats['count'])
        return True
    except Exception as e:
        logger.warning("⚠ Could not check documents: %s", e)
        return True


def start_server(mode='production', port=8080):
    """Start the API server"""
    logger.info("\n" + "="*60)
    logger.info("Starting Multilingual RAG API Server")
    logger.info("="*60)
    
    if mode == 'development':
        logger.info("Mode: Development (auto-reload enabled)")
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api_server:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--reload",
            "--log-level", "info"
        ]
    else:
        logger.info("Mode: Production")
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api_server:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--workers", "1",
            "--log-level", "info"
        ]
    
    logger.info(f"\nAPI will be available at:")
    logger.info(f"  → http://localhost:{port}")
    logger.info(f"  → http://localhost:{port}/api/docs (interactive docs)")
    logger.info("\nPress Ctrl+C to stop\n")
    logger.info("="*60 + "\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Server exited with error: {e}")
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Start the Multilingual RAG API Server')
    parser.add_argument('--dev', action='store_true', help='Run in development mode with auto-reload')
    parser.add_argument('--skip-checks', action='store_true', help='Skip pre-flight checks')
    parser.add_argument('--port', type=int, default=8080, help='Port to run server on (default: 8080)')
    args = parser.parse_args()
    
    logger.info("Multilingual Scientific RAG System")
    logger.info("="*60)
    
    if not args.skip_checks:
        logger.info("\nRunning pre-flight checks...")
        logger.info("-"*60)
        
        checks = [
            check_python_version(),
            check_env_file(),
            check_api_key(),
            check_dependencies(),
            check_documents()
        ]
        
        if not all(checks[:4]):  # First 4 are critical
            logger.error("\n✗ Pre-flight checks failed!")
            logger.info("Fix the issues above and try again\n")
            sys.exit(1)
        
        logger.info("-"*60)
        logger.info("✓ All checks passed!\n")
    
    # Ensure directories exist before starting
    try:
        import config
        config.ensure_directories()
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)
    
    mode = 'development' if args.dev else 'production'
    start_server(mode, port=args.port)


if __name__ == '__main__':
    main()
