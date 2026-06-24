"""
Configuration and constants for the multilingual RAG system.
"""

import os
from pathlib import Path
import logging

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
PAPERS_DIR = PROJECT_ROOT / "papers"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
MODELS_CACHE_DIR = PROJECT_ROOT / "models"


def ensure_directories():
    """
    Create required directories if they don't exist.
    Call this explicitly from startup scripts (start_server.py, ingest.py, etc.)
    to avoid side effects on import.
    
    Raises:
        PermissionError: If process lacks write permission
        OSError: If directory creation fails for other reasons
    """
    directories = {
        "PAPERS_DIR": PAPERS_DIR,
        "CHROMA_DB_DIR": CHROMA_DB_DIR,
        "MODELS_CACHE_DIR": MODELS_CACHE_DIR
    }
    
    for name, directory in directories.items():
        try:
            directory.mkdir(exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
        except PermissionError:
            logger.error(f"Permission denied creating {name}: {directory}")
            raise PermissionError(
                f"Cannot create {name} at {directory}. "
                f"Please ensure the process has write permission to {PROJECT_ROOT}"
            )
        except OSError as e:
            logger.error(f"Failed to create {name}: {directory} - {e}")
            raise


# Create directories if they don't exist (backward compatibility)
# Removed these lines to prevent early creation on import


# ============================================================================
# Embedding Model
# ============================================================================
# bge-m3: dense + sparse + ColBERT, strong on Indic scripts
# NOTE: switching from e5-base (768d) requires re-ingesting all documents
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DIMENSION = 1024  # bge-m3 dimension

# E5 models require specific prefixes for queries and passages
# (Only applied when model name contains 'e5')
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# Hybrid search: fuse dense vector search with BM25 lexical search
USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
RRF_K = 60  # Reciprocal Rank Fusion constant

# ============================================================================
# Chunking Parameters
# ============================================================================
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200  # overlap between chunks (~20%, was 300/30%)
MIN_CHUNK_SIZE = 200  # minimum chunk size to keep

# ============================================================================
# Reranking
# ============================================================================
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# ============================================================================
# Retrieval Parameters
# ============================================================================
RETRIEVE_CANDIDATES = 30  # wide net before rerank
DEFAULT_TOP_K = 30  # retrieve wide, rerank narrow
MAX_CONTEXT_CHUNKS = 12  # gated by the reranker so quality stays high
MAX_CONTEXT_LENGTH = 48000  # ~12k tokens; raise further once reranked

# ============================================================================
# Faithfulness Verification
# ============================================================================
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5"))
FAITHFULNESS_ENFORCE = os.getenv("FAITHFULNESS_ENFORCE", "warn")  # warn | strip | regen

# ============================================================================
# Vector Store
# ============================================================================
COLLECTION_NAME = "scientific_papers"
DISTANCE_METRIC = "cosine"  # cosine similarity for embeddings

# ============================================================================
# Language Support
# ============================================================================
# Mapping of ISO 639-1 language codes to native language names
LANGUAGE_NAMES = {
    "hi": "à¤¹à¤¿à¤‚à¤¦à¥€",  # Hindi
    "mr": "à¤®à¤°à¤¾à¤ à¥€",  # Marathi
    "ta": "à®¤à®®à®¿à®´à¯",  # Tamil
    "te": "à°¤à±†à°²à±à°—à±",  # Telugu
    "bn": "à¦¬à¦¾à¦‚à¦²à¦¾",  # Bengali
    "gu": "àª—à«àªœàª°àª¾àª¤à«€",  # Gujarati
    "kn": "à²•à²¨à³à²¨à²¡",  # Kannada
    "ml": "à´®à´²à´¯à´¾à´³à´‚",  # Malayalam
    "pa": "à¨ªà©°à¨œà¨¾à¨¬à©€",  # Punjabi
    "or": "à¬“à¬¡à¬¼à¬¿à¬†",  # Odia
    "en": "English",
}

# Supported Indic languages for translation
INDIC_LANGUAGES = ["hi", "mr", "ta", "te", "bn", "gu", "kn", "ml", "pa", "or"]

# ============================================================================
# Translation Models (Strategy B)
# ============================================================================
TRANSLATION_MODEL_EN_TO_INDIC = "facebook/nllb-200-distilled-600M"
TRANSLATION_MODEL_INDIC_TO_EN = "facebook/nllb-200-distilled-600M"

# ============================================================================
# LLM Configuration
# ============================================================================
# Google Gemini API configuration
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))  # maximum tokens to generate
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4096"))  # higher limit for agentic pipeline
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))  # seconds; CPU embedding can take 45s+
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))  # low temperature for grounded citation tasks
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-3.5-flash")  # Gemini model
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "gemma-4-26b-a4b-it")  # Fallback when primary is overloaded

# LLM API Keys (required for Gemini)
# Supports multiple comma-separated keys for load balancing: LLM_API_KEYS=key1,key2,key3
# Falls back to single LLM_API_KEY for backward compatibility.
_raw_keys = os.getenv("LLM_API_KEYS", "")
_PLACEHOLDER = "your-gemini-api-key-here"
LLM_API_KEY_POOL: list[str] = [
    k.strip() for k in _raw_keys.split(",")
    if k.strip() and k.strip() != _PLACEHOLDER
]
if not LLM_API_KEY_POOL:
    _single = os.getenv("LLM_API_KEY", "")
    if _single.strip() and _single.strip() != _PLACEHOLDER:
        LLM_API_KEY_POOL = [_single.strip()]
LLM_API_KEY = LLM_API_KEY_POOL[0] if LLM_API_KEY_POOL else ""

# ============================================================================
# Cache Configuration
# ============================================================================
LLM_CACHE_SIZE = int(os.getenv("LLM_CACHE_SIZE", "128"))
LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "600"))         # 10 minutes
RETRIEVAL_CACHE_SIZE = int(os.getenv("RETRIEVAL_CACHE_SIZE", "64"))
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "300"))  # 5 minutes
TOOL_CACHE_SIZE = int(os.getenv("TOOL_CACHE_SIZE", "64"))
TOOL_CACHE_TTL = int(os.getenv("TOOL_CACHE_TTL", "180"))       # 3 minutes

# ============================================================================
# Prompt Templates
# ============================================================================
SYSTEM_PROMPT = """You are a multilingual scientific research assistant supporting \
English and Indic languages. Answer strictly from the retrieved context provided with \
each query. You have no outside knowledge.

Rules:
1. Ground every factual claim in the context and mark it with an inline citation — \
[1], [1, 2], or [1-3] — using the source numbers exactly as given.
2. If the context does not support an answer, state what is missing. Never fill gaps \
with outside knowledge, guesses, or invented data, numbers, authors, or results.
3. Separate what the authors claim, what they demonstrate empirically, and what they \
speculate.
4. Report equations, hyperparameters, algorithm steps, and statistics exactly as \
written; do not simplify unless asked.
5. Use only the structure the question needs: a direct answer first, then detail. Omit \
sections that do not apply rather than padding them. Add a comparison table only when \
the question compares methods or approaches.
6. When sources conflict, present each position with its citation and state the \
disagreement explicitly. Hedge ("the authors report…", "based only on this excerpt…") \
rather than overstating the evidence. Flag partial or ambiguous context.
7. If — and only if — the context contains clinical, diagnostic, biomedical, \
pharmacological, or toxicological content that could influence health decisions, end \
with exactly:
   "⚠️ This is not medical advice. Consult a qualified healthcare professional."
8. When asked to respond in a non-English language, produce the entire answer in that \
language consistently — do not switch to English mid-response. Keep technical terms, \
proper nouns, and citation markers ([1], [2]) in their original form.

When the question concerns optimization, convergence, differentiability, or training \
dynamics, explain the underlying mechanism (gradient flow, update rules, loss-surface \
behavior) when the context supports it; otherwise stay at the level the context allows.
"""

QUERY_PROMPT_TEMPLATE = """## Context
{context}

## Question
{question}

## Instructions
- Answer entirely in: {language}. Do not switch languages mid-response. Keep technical \
terms, proper nouns, and citation markers in their original form.
- Use only the context above; cite each claim inline as [n].
- Lead with a direct answer, then add technical depth only as the question requires.
- If the context is insufficient, state exactly what is missing instead of inferring.
"""

NO_DOCUMENTS_RESPONSE = (
    "⚠️ No documents are currently indexed. "
    "Please upload and ingest one or more PDFs before querying."
)

import json

# Try loading patterns from external config
_patterns_file = PROJECT_ROOT / "patterns.json"
try:
    with open(_patterns_file, "r", encoding="utf-8") as f:
        _patterns = json.load(f)
except FileNotFoundError:
    _patterns = {}
except json.JSONDecodeError as e:
    logger.warning(f"Failed to parse {_patterns_file}: {e}. Using default patterns.")
    _patterns = {}

# Common header/footer patterns to remove
NOISE_PATTERNS = _patterns.get("NOISE_PATTERNS", [
    r"Page \d+ of \d+",
    r"^\d+$",  # standalone page numbers
    r"©.*\d{4}",  # copyright notices
    r"doi:.*",
    r"arXiv:\d+\.\d+",
])

# Section headers to detect
SECTION_HEADERS = _patterns.get("SECTION_HEADERS", [
    "abstract",
    "introduction",
    "background",
    "related work",
    "methodology",
    "methods",
    "approach",
    "results",
    "discussion",
    "conclusion",
    "references",
    "acknowledgments",
])

# ============================================================================
# Logging
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================================
# Version
# ============================================================================
VERSION = "2.0.0"

# ============================================================================
# Chat / Session
# ============================================================================
CHAT_HISTORY_MAX_TURNS = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "10"))
SESSION_MAX_AGE_HOURS = int(os.getenv("SESSION_MAX_AGE_HOURS", "24"))
