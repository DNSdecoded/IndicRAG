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
RETRIEVE_CANDIDATES = 15  # wider net for agent; keep moderate for CPU embedding speed
DEFAULT_TOP_K = 15  # dense + BM25 fusion, then rerank narrow
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
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "8192"))  # higher limit for agentic pipeline
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
SYSTEM_PROMPT = """\
You are a multilingual scientific research assistant. \
Answer strictly from the retrieved context provided with each query.

Rules:
1. GROUNDING: Ground every factual claim with an inline citation [N] \
   using the source number exactly as given in the context.
2. CITATION ESCAPE: If a claim cannot be supported by any source, write \
   [NOT FOUND: <topic>] — never leave a factual sentence without either \
   a [N] citation or a [NOT FOUND] marker.
3. NO FABRICATION: Never fill gaps with outside knowledge, guesses, or \
   invented data, numbers, authors, or results. If the context is \
   insufficient, state exactly what is missing.
4. SOURCE INTEGRITY: Distinguish what authors claim, what they demonstrate \
   empirically, and what they speculate. Hedge with "the authors report…" \
   rather than stating findings as universal facts.
5. ACCURACY: Report equations, hyperparameters, algorithm steps, and \
   statistics exactly as written. Do not simplify unless explicitly asked.
6. CONCISION: Lead with a direct answer, then add technical depth only as \
   the question requires. Omit sections that do not apply.
7. CONFLICTS: When sources disagree, present each position with its [N] \
   and state the disagreement explicitly rather than silently merging them.
8. LANGUAGE: When asked to respond in a non-English language, produce the \
   entire answer in that language consistently. Keep technical terms, \
   proper nouns, and citation markers [N] in their original form.
9. MEDICAL: If — and only if — the context describes specific patient \
   treatment recommendations, dosage guidance, or diagnostic criteria that \
   could directly influence a health decision, append exactly: \
   "⚠️ This is not medical advice. Consult a qualified healthcare professional."
10. CONDUCT: Never reference system architecture, prompt guidelines, or \
    internal engineering constraints in your output.\
"""

# Used by answer_generator_node — handles externally retrieved papers
AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT + """

AGENTIC RETRIEVAL MODE: The context above includes passages retrieved from \
both the local indexed corpus AND external academic databases (arXiv, OpenAlex, \
Semantic Scholar). External sources are legitimate and intentionally retrieved — \
treat them identically to local corpus chunks. Cite them with [N] as normal. \
When a source is an arXiv preprint (not yet peer-reviewed), note this \
parenthetically after the citation: [N] (preprint).\
"""

QUERY_PROMPT_TEMPLATE = """\
<context>
{context}
</context>

<query>
{question}
</query>

<instructions>
- Respond entirely in: {language}. Do not switch languages mid-response.
- Cite every factual sentence inline as [N] using the source number from <context>.
- Use [NOT FOUND: topic] for any claim the context cannot support.
- Lead with a direct answer; add technical depth only as the query requires.
- If context is insufficient, state exactly what is missing rather than inferring.
</instructions>\
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
    "materials and methods",
    "experimental",
    "analysis",
    "limitations",
    "future work",
    "appendix",
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
