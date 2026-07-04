"""
Configuration and constants for the multilingual RAG system.
"""

import json
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
SESSIONS_DB_PATH = Path(os.getenv("SESSIONS_DB_PATH", PROJECT_ROOT / "sessions.db"))


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

# Per-section chunk-size overrides (chars). Dense sections (abstract, methods,
# conclusion) chunk smaller for retrieval precision; narrative sections
# (results, discussion) chunk larger to preserve context. Falls back to CHUNK_SIZE.
# Method sections carry the equation blocks (reward funcs, constraints, algorithm
# steps). 500 chars split a labelled block like (2a)…(2c) across chunks, orphaning
# formulas from their variable definitions — so method sections chunk at 900 to
# keep an equation group with its surrounding text. Abstract/conclusion stay tight
# (short prose, no math).
SECTION_CHUNK_SIZES = {
    "abstract": 500,
    "methods": 900,
    "methodology": 900,
    "materials and methods": 900,
    "approach": 900,
    "conclusion": 500,
    "results": 1500,
    "discussion": 1500,
}

# ============================================================================
# Reranking
# ============================================================================
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# ColBERT MaxSim reranking (query-time, no persistent index — see colbert_rerank.py)
USE_COLBERT_RERANK = os.getenv("USE_COLBERT_RERANK", "false").lower() == "true"
COLBERT_WEIGHT = float(os.getenv("COLBERT_WEIGHT", "0.5"))  # dense-vs-colbert fusion weight

# HyDE: embed a drafted hypothetical answer instead of the bare query
USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"

# Ingest-time metadata enrichment (arXiv) and cross-ingestion title dedup
ENRICH_METADATA = os.getenv("ENRICH_METADATA", "true").lower() == "true"
DEDUP_PAPERS = os.getenv("DEDUP_PAPERS", "true").lower() == "true"
DEDUP_TITLE_THRESHOLD = float(os.getenv("DEDUP_TITLE_THRESHOLD", "0.9"))

# Opt-in per-user preference storage (GET/PUT /prefs/{user_id}) — caller
# supplies user_id; not wired into agent prompts (no user-identity system
# exists elsewhere in this app to link it to yet).
ENABLE_USER_PREFS = os.getenv("ENABLE_USER_PREFS", "false").lower() == "true"

# ============================================================================
# Retrieval Parameters
# ============================================================================
RETRIEVE_CANDIDATES = 15  # wider net for agent; keep moderate for CPU embedding speed
DEFAULT_TOP_K = 15  # dense + BM25 fusion, then rerank narrow
MAX_CONTEXT_CHUNKS = 12  # gated by the reranker so quality stays high
MAX_CONTEXT_LENGTH = 48000  # ~12k tokens; raise further once reranked

# Agentic mode gathers passages from up to 4 tool calls, so the 12-chunk cap can
# truncate formula/equation chunks off the tail before the LLM sees them. The
# agent answer generator uses a wider cap (Gemini's window has the room).
AGENT_MAX_CONTEXT_CHUNKS = int(os.getenv("AGENT_MAX_CONTEXT_CHUNKS", "20"))
AGENT_MAX_CONTEXT_LENGTH = int(os.getenv("AGENT_MAX_CONTEXT_LENGTH", "80000"))

# Paper-scoped ("only these papers") retrieval fetches ALL chunks of the selected
# papers in document order instead of top-k, so reconstruction-style queries see
# the whole paper. Caps keep a very large paper/book from blowing the context.
SCOPED_MAX_CHUNKS = int(os.getenv("SCOPED_MAX_CHUNKS", "50"))
SCOPED_MAX_CONTEXT_LENGTH = int(os.getenv("SCOPED_MAX_CONTEXT_LENGTH", "120000"))  # ~30k tokens; fits Gemini's window

# ============================================================================
# Faithfulness Verification
# ============================================================================
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5"))
FAITHFULNESS_ENFORCE = os.getenv("FAITHFULNESS_ENFORCE", "warn")  # warn | strip | regen

# NLI model for claim faithfulness. Default is MULTILINGUAL so Indic-language
# claims are scored against Indic chunks by a model that understands them; the
# old English-only nli-deberta-v3-base collapsed to noise on Hindi/Tamil/Bengali
# — the languages that are this project's differentiator.
NLI_MODEL_NAME = os.getenv(
    "NLI_MODEL_NAME",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
)
# Entailment class index in the model's output logits — models disagree:
#   mDeBERTa-xnli (default):     0=entailment, 1=neutral, 2=contradiction
#   cross-encoder/nli-deberta-v3-base: 0=contradiction, 1=entailment, 2=neutral
# Set this to match whatever NLI_MODEL_NAME you choose or scores invert silently.
NLI_ENTAILMENT_INDEX = int(os.getenv("NLI_ENTAILMENT_INDEX", "0"))

# ============================================================================
# Vector Store
# ============================================================================
COLLECTION_NAME = "scientific_papers"
DISTANCE_METRIC = "cosine"  # cosine similarity for embeddings

# HNSW index tuning. ef_construction and M (max_neighbors) are fixed at collection
# CREATE time (immutable after — change them and you must re-create the collection);
# ef_search is a query-time recall/latency knob. Defaults match ChromaDB's own, so
# behaviour is unchanged until you deliberately A/B them once the eval set exists.
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "100"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))
HNSW_M = int(os.getenv("HNSW_M", "16"))

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
# Thinking budget for ALL agentic-mode LLM calls (query planner, tool routing,
# query expansion, answer generation, reflexion judge). Gemini semantics:
#   0  = thinking OFF (cheapest — default; matches standard RAG)
#   -1 = DYNAMIC (model decides how much to think)
#   N  = cap thinking to N tokens (billed; higher = smarter routing/judging, pricier)
# Raise this only if agent answer/routing quality is the bottleneck, not the bill.
AGENT_THINKING_BUDGET = int(os.getenv("AGENT_THINKING_BUDGET", "0"))
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))  # seconds; CPU embedding can take 45s+
# Wall-clock budget for the reflexion loop. Once exceeded, the evaluator finalizes
# the current best draft instead of starting another retrieve→generate→verify cycle,
# so the user gets an answer rather than a hard AGENT_TIMEOUT 504 that discards all
# work. Keep below AGENT_TIMEOUT so it fires first.
AGENT_REFLEXION_BUDGET_S = float(os.getenv("AGENT_REFLEXION_BUDGET_S", "90"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))  # low temperature for grounded citation tasks
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-3.5-flash")  # Gemini model

# Explicit Gemini context caching of the (stable) system-instruction prefix.
# gemini-3.5-flash already does IMPLICIT caching for free; explicit caching adds
# guaranteed reuse but is billed per token-hour of storage — so it's OFF by default.
# Enable only if your system prompts clear the model's min-token cache floor and you
# want deterministic cache hits. Falls back to inline prompts on any create failure.
GEMINI_CACHE_ENABLED = os.getenv("GEMINI_CACHE_ENABLED", "false").lower() == "true"
GEMINI_CACHE_TTL = int(os.getenv("GEMINI_CACHE_TTL", "3600"))  # seconds cache lives
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
- Equations in <context> may appear as flattened multi-line plain text (PDF extraction). Reconstructing such an equation into standard notation or LaTeX is faithful quoting, not inference — do it when asked, using only the symbols and values present in the context.
- If context is insufficient, state exactly what is missing rather than inferring.
</instructions>\
"""

NO_DOCUMENTS_RESPONSE = (
    "⚠️ No documents are currently indexed. "
    "Please upload and ingest one or more PDFs before querying."
)

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

# Indic-script section headers (Hindi/Marathi + Tamil), appended so
# extract_sections detects sections in Indic-language papers, not just Latin
# headers. Appended unconditionally so it applies even when patterns.json
# overrides SECTION_HEADERS above.
INDIC_SECTION_HEADERS = [
    # Hindi / Marathi (Devanagari)
    "सारांश",      # abstract / summary
    "प्रस्तावना",   # introduction
    "परिचय",       # introduction
    "पृष्ठभूमि",     # background
    "कार्यप्रणाली",  # methodology
    "विधि",        # method
    "परिणाम",      # results
    "चर्चा",        # discussion
    "निष्कर्ष",      # conclusion
    "संदर्भ",       # references
    # Tamil
    "முன்னுரை",    # introduction
    "முறை",        # method
    "முடிவுகள்",    # results
    "முடிவு",       # conclusion
    "மேற்கோள்கள்",  # references
]
SECTION_HEADERS = list(SECTION_HEADERS) + INDIC_SECTION_HEADERS

# ============================================================================
# Logging
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================================
# Version
# ============================================================================
VERSION = "2.2.0"

# ============================================================================
# Chat / Session
# ============================================================================
CHAT_HISTORY_MAX_TURNS = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "10"))
SESSION_MAX_AGE_HOURS = int(os.getenv("SESSION_MAX_AGE_HOURS", "24"))

# ============================================================================
# Safety Settings (shared across rag.py and agent nodes)
# ============================================================================
from google.genai import types as _genai_types  # noqa: E402 — lazy-ish, underscore keeps it private
_dev_mode = os.getenv("DEV_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
_safety_threshold = "BLOCK_NONE" if _dev_mode else "BLOCK_MEDIUM_AND_ABOVE"
SAFETY_SETTINGS = [
    _genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=_safety_threshold),
    _genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=_safety_threshold),
    _genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=_safety_threshold),
    _genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=_safety_threshold),
]
