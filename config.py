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
# Thread budget
# ============================================================================
# The process already runs a 32-worker ChromaDB timeout pool, FastAPI's ~40-thread
# default pool, a ProcessPoolExecutor for PDF parsing, and a ThreadPoolExecutor for
# parallel agent tools. On top of that, torch and ONNX Runtime each default to one
# intra-op thread PER CORE, so a single reranker pass can fan out across every core
# while dozens of request threads are already runnable. The result is a ready queue
# far deeper than the core count, where every stage slows down at once.
#
# Pin it: 0 means "leave the library default alone" (opt out); anything else caps
# intra-op parallelism. Must be set BEFORE torch/onnxruntime are imported, which is
# why it lives at the top of config.py — the first module everything else imports.
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "4"))
if TORCH_NUM_THREADS > 0:
    os.environ.setdefault("OMP_NUM_THREADS", str(TORCH_NUM_THREADS))
    os.environ.setdefault("MKL_NUM_THREADS", str(TORCH_NUM_THREADS))

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
PAPERS_DIR = PROJECT_ROOT / "papers"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
MODELS_CACHE_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"  # Phase 3: saved figure/table crops
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
        "MODELS_CACHE_DIR": MODELS_CACHE_DIR,
        "FIGURES_DIR": FIGURES_DIR,
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
# int8 ONNX for the embedding model on CPU. Works, but OFF by default — measured,
# the trade is worse than it is for the cross-encoders:
#
#   speed              1.38x faster (the reranker/NLI get ~3-11x)
#   cosine vs fp32     mean 0.987, min 0.983 across a 24-chunk sample
#
# Every vector moves. 0.983 is enough for near-neighbours to reorder, so this can
# change what retrieval returns — and there is currently no working way to measure
# that (docs/Eval/run_live.py is blocked on the judgments/corpus mismatch). Trading
# unmeasurable retrieval quality for 1.38x on an operation that runs once per
# corpus is a bad deal; the reranker's 3-11x on a per-QUERY path is a good one.
#
# Turn it on deliberately, and only with a working eval gate to confirm quality
# held. Switching it means re-embedding the whole corpus — mixing backends in one
# collection is the incomparable-vectors problem. Every chunk records which
# backend produced it (vector_store._provenance_stamp) and a mixed collection is
# reported at startup; `reindex.py` replays the ingest log to re-embed without
# re-parsing PDFs.
#
# CPU-only regardless: on GPU, fp16 is both faster and more accurate than int8.
EMBED_ONNX_INT8 = os.getenv("EMBED_ONNX_INT8", "false").lower() == "true"
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

# Phase 3 — multimodal figure/table indexing.
# On: ingest extracts figure/table crops + nearby captions, has the Gemini VLM
# describe each, and indexes "caption + description" as chunks in the SAME
# collection (retrieval/citation unchanged). Off by default — adds one VLM call
# per figure at ingest, bounded by MULTIMODAL_MAX_FIGS_PER_DOC.
ENABLE_MULTIMODAL_INGEST = os.getenv("ENABLE_MULTIMODAL_INGEST", "false").lower() == "true"
MULTIMODAL_MAX_FIGS_PER_DOC = int(os.getenv("MULTIMODAL_MAX_FIGS_PER_DOC", "12"))
# Concurrent VLM caption calls per paper; 1 restores the old sequential behavior.
# Keep this small — the cap bounds requests in flight at the provider, it is not
# there to saturate the box. Too high just converts a slow ingest into 429s that
# trip the LLM circuit breaker for every other caller.
FIGURE_CAPTION_WORKERS = int(os.getenv("FIGURE_CAPTION_WORKERS", "4"))

# Persist the BM25 index between runs. Rebuilding reads every document out of
# ChromaDB, so without this a restart pays a full corpus scan on the first query
# and the lifespan warm-up pays it on every deploy. The cache is validated
# against the live document count on load and ignored when it disagrees, so a
# stale file costs a rebuild rather than wrong results.
BM25_PERSIST = os.getenv("BM25_PERSIST", "true").lower() == "true"
BM25_CACHE_DIR = Path(os.getenv("BM25_CACHE_DIR", PROJECT_ROOT / "chroma_db"))

# Phase 5 — contradiction/consensus detection. When on, the answer generator
# runs pairwise NLI (the same faithfulness model) over the top retrieved passages
# and, if any pair contradicts above the threshold, instructs the model to
# present both positions with citations instead of silently picking one. Off by
# default — adds O(n^2) NLI passes over a capped passage set at answer time.
CONTRADICTION_DETECT_ENABLE = os.getenv("CONTRADICTION_DETECT_ENABLE", "false").lower() == "true"
CONTRADICTION_NLI_THRESHOLD = float(os.getenv("CONTRADICTION_NLI_THRESHOLD", "0.6"))

# Ingest-time metadata enrichment (arXiv) and cross-ingestion title dedup
ENRICH_METADATA = os.getenv("ENRICH_METADATA", "true").lower() == "true"
DEDUP_PAPERS = os.getenv("DEDUP_PAPERS", "true").lower() == "true"
DEDUP_TITLE_THRESHOLD = float(os.getenv("DEDUP_TITLE_THRESHOLD", "0.9"))

# Opt-in per-user preference storage (GET/PUT /prefs/{user_id}) — caller
# supplies user_id; not wired into agent prompts (no user-identity system
# exists elsewhere in this app to link it to yet).
ENABLE_USER_PREFS = os.getenv("ENABLE_USER_PREFS", "false").lower() == "true"

# Phase 6 — "watch a topic" scheduled ingest + digests. When on, users can
# register topic watches (POST /watch); each run does external search (arXiv /
# open-access), dedups against the corpus, ingests genuinely new papers, and
# stores a cited digest. Off by default — it makes outbound API calls and grows
# the corpus. Cadence controls the default re-run interval for new watches.
WATCH_ENABLE = os.getenv("WATCH_ENABLE", "false").lower() == "true"
WATCH_DEFAULT_CADENCE = os.getenv("WATCH_DEFAULT_CADENCE", "weekly")  # daily|weekly|monthly
WATCH_MAX_RESULTS = int(os.getenv("WATCH_MAX_RESULTS", "10"))  # papers fetched per watch run
WATCH_POLL_INTERVAL = int(os.getenv("WATCH_POLL_INTERVAL", "3600"))  # seconds between schedule-loop sweeps
# How long a claimed watch is parked before it becomes due again. Must comfortably
# exceed a real run (arXiv search + ingest + digest generation), or a slow run
# could be picked up a second time while the first is still going. It also bounds
# how long a watch stays stuck if the claimer dies mid-run.
WATCH_LEASE_SECONDS = int(os.getenv("WATCH_LEASE_SECONDS", "3600"))

# How long a job's lease stays valid without a heartbeat. Every progress update
# renews it, so this only has to exceed the longest gap BETWEEN updates — not the
# job's total runtime. Too low and a slow step gets its job reaped out from under
# it; too high and a crashed job takes that long to be marked failed.
JOB_LEASE_SECONDS = int(os.getenv("JOB_LEASE_SECONDS", "900"))

# Phase 7 — literature-review report workflow. POST /report decomposes a topic
# into sections, synthesizes a cited section per part from the corpus, and
# stores a downloadable Markdown artifact as an async job. Off by default: a
# multi-section report is several LLM calls per request.
REPORT_ENABLE = os.getenv("REPORT_ENABLE", "false").lower() == "true"
REPORT_MAX_SECTIONS = int(os.getenv("REPORT_MAX_SECTIONS", "6"))  # cap sections to bound cost/latency

# ============================================================================
# Retrieval Parameters
# ============================================================================
RETRIEVE_CANDIDATES = 15  # wider net for agent; keep moderate for CPU embedding speed
DEFAULT_TOP_K = 15  # dense + BM25 fusion, then rerank narrow
MAX_CONTEXT_CHUNKS = 12  # gated by the reranker so quality stays high
MAX_CONTEXT_LENGTH = 48000  # ~12k tokens; raise further once reranked

# Tags are applied as a Python-side post-filter (ChromaDB can't match one tag
# inside the stored comma-joined string), so a plain top-k fetch returns zero
# results whenever the tagged papers rank below the cut. Over-fetch first, then
# filter, then narrow back to top_k. TAGS_OVERFETCH_MAX bounds the widened fetch.
TAGS_OVERFETCH = int(os.getenv("TAGS_OVERFETCH", "10"))
TAGS_OVERFETCH_MAX = int(os.getenv("TAGS_OVERFETCH_MAX", "300"))

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
# Per-claim entailment probability above which a claim counts as grounded.
# Calibrated on this corpus with the shipped int8 mDeBERTa-xnli model: a sentence
# copied VERBATIM out of its own chunk scores median 0.226 / max 0.428, while a
# sentence from an unrelated paper scores p90 0.158. The old 0.5 was unreachable —
# recall 0.00 on guaranteed-grounded claims, so faithfulness read ~0 for every
# answer ever produced. 0.15 gives recall 0.70 at false-positive 0.10.
# Recalibrate (positives vs cross-paper negatives) if NLI_MODEL_NAME changes.
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.15"))
# Fraction of an answer's claims that must be grounded for the reflexion loop to
# accept it, and for the finalizer to trust the answer enough to abstain on
# completeness alone. Sits below 1.0 by design: per-claim recall is 0.70, so even a
# fully grounded answer lands near 0.70 — the old hardcoded 0.75 could never fire.
AGENT_FAITHFULNESS_ACCEPT = float(os.getenv("AGENT_FAITHFULNESS_ACCEPT", "0.6"))
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
# Contradiction class index in the SAME model's logits (Phase 5 contradiction
# detection). Mirrors NLI_ENTAILMENT_INDEX and follows the same per-model order:
#   mDeBERTa-xnli (default):     2=contradiction
#   cross-encoder/nli-deberta-v3-base: 0=contradiction
NLI_CONTRADICTION_INDEX = int(os.getenv("NLI_CONTRADICTION_INDEX", "2"))
# NLI cost is linear in pairs and in premise length — measured on this CPU box with
# the int8 ONNX model: 1.15s/pair at 512 tokens, 0.4s/pair at 256. A 30-sentence
# answer citing multi-chunk papers hit ~275 pairs = 318s in one reflexion pass.
# These two knobs bound that: shorter premise, and at most N chunks per cited paper
# (chunks arrive rerank-ordered, so the first ones are the best support anyway).
NLI_MAX_SEQ_LENGTH = int(os.getenv("NLI_MAX_SEQ_LENGTH", "256"))
# Clamped to >=1 because this value FAILS OPEN: a 0 (or negative) cap slices away
# every cited chunk, check_claims() then returns no claims, and the reflexion
# evaluator reads an empty claim list as "no citable claims != hallucination" —
# faithfulness 1.0, answer accepted with zero grounding. A typo in .env would
# silently disable verification while reporting perfect scores.
NLI_MAX_CHUNKS_PER_CITATION = max(1, int(os.getenv("NLI_MAX_CHUNKS_PER_CITATION", "2")))

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
# Caps thinking + answer together, not just the answer. Gemini 3.x Flash models
# reject thinking_budget=0 and spend a variable number of thought tokens on
# identical prompts (measured on 3.6-flash: 0-4856), so a 2048 cap left as little
# as 80 tokens for the answer and truncated mid-sentence.
# Measured on 3.6-flash: answer <=2100, worst thinking+answer 6926. Keep the
# headroom for newer Flash models rather than re-tuning per release.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))  # maximum tokens to generate
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "8192"))  # higher limit for agentic pipeline
# Thinking budget for ALL agentic-mode LLM calls (query planner, tool routing,
# query expansion, answer generation, reflexion judge). Gemini semantics:
#   0  = thinking OFF (cheapest — default; matches standard RAG)
#   -1 = DYNAMIC (model decides how much to think)
#   N  = cap thinking to N tokens (billed; higher = smarter routing/judging, pricier)
# Raise this only if agent answer/routing quality is the bottleneck, not the bill.
# LEGACY on Gemini 3.x: those models reject thinking_budget outright (see the level
# knobs below, which supersede it). Still honoured by models that accept budgets.
AGENT_THINKING_BUDGET = int(os.getenv("AGENT_THINKING_BUDGET", "0"))
# Thinking LEVEL — the Gemini 3.x control, replacing thinking_budget. Google's docs
# list minimal | low | medium | high for Gemini 3.x Flash, defaulting to MEDIUM when
# nothing is sent. That default is the trap: sending the legacy thinking_budget=0 gets
# a 400, the backend used to drop the field entirely, and the model then thought at
# MEDIUM — the opposite of the "thinking off" that was asked for, with those thought
# tokens coming out of LLM_MAX_TOKENS and squeezing the answer.
#   minimal = least thinking (default here; closest to the old budget=0 intent)
#   low | medium | high = progressively more (slower, pricier, sometimes better)
#   ""  = send nothing, i.e. accept the model's own default
LLM_THINKING_LEVEL = os.getenv("LLM_THINKING_LEVEL", "minimal").strip().lower()
AGENT_THINKING_LEVEL = os.getenv("AGENT_THINKING_LEVEL", "minimal").strip().lower()
# Seconds. Measured end-to-end on a CPU-only box: ~45s retrieval + ~50s generation
# + ~25s evaluation. The old 120s default left under 30s of room once
# AGENT_EVAL_RESERVE_S was set aside, so the evaluator skipped verification on
# every stock-config run — the answer shipped with no faithfulness score at all.
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300"))
# Per-request HTTP timeout for non-streaming LLM calls. Without it the SDK defaults
# apply (OpenAI: 600s x 2 retries) and ONE stalled request outlasts the whole agent
# budget. 60s is sized for a legitimate call, not for the failover chain: agent
# answer generation is unary and measured 20-50s on CPU, so a materially lower
# value would abort real generations rather than stalled ones.
#
# The chain is therefore NOT bounded by AGENT_REFLEXION_BUDGET_S: generate_with_failover
# walks up to 3 (provider, model) attempts sequentially, so a fully-stalled chain can
# reach ~180s — past the 90s reflexion budget. What actually bounds it is AGENT_TIMEOUT
# plus AGENT_EVAL_RESERVE_S, which finalise the draft rather than 504. In practice the
# per-(provider, model) circuit breaker in llm_client skips recently-dead paths, so
# three consecutive full stalls are rare. If that worst case ever matters more than
# generation headroom, make the timeout deadline-aware (remaining budget / attempts
# left) instead of just lowering it.
LLM_REQUEST_TIMEOUT_S = int(os.getenv("LLM_REQUEST_TIMEOUT_S", "60"))
# Streaming needs its own, much larger budget: for Gemini the HTTP timeout covers
# the WHOLE stream, not the gap between chunks, so reusing the 60s unary value tore
# down long answers mid-generation (WinError 10054, truncated text). This still
# bounds a genuinely stuck stream without capping legitimate long generations.
LLM_STREAM_TIMEOUT_S = int(os.getenv("LLM_STREAM_TIMEOUT_S", "300"))
# Wall-clock budget for the reflexion loop. Once exceeded, the evaluator finalizes
# the current best draft instead of starting another retrieve→generate→verify cycle,
# so the user gets an answer rather than a hard AGENT_TIMEOUT 504 that discards all
# work. Keep below AGENT_TIMEOUT so it fires first.
AGENT_REFLEXION_BUDGET_S = float(os.getenv("AGENT_REFLEXION_BUDGET_S", "90"))
# Wall-clock room that must remain under AGENT_TIMEOUT for the evaluator to attempt
# an evaluation at all. Unlike the budget above (which only stops FURTHER loops),
# this one can skip iteration 1 — but only when finishing would overrun the timeout
# and discard the draft entirely. Sized for one NLI pass plus one completeness LLM
# call: measured ~30s + ~15s on CPU, doubled for headroom.
AGENT_EVAL_RESERVE_S = float(os.getenv("AGENT_EVAL_RESERVE_S", "90"))
# Max sub-queries the planner emits (and tools run per cycle). Each sub-query does a
# retrieve + CPU reranker pass (~15 pairs); those passes are CPU-bound so N concurrent
# ones thrash rather than parallelize. Over a small corpus, 3 covers most queries at a
# fraction of the latency of 6. Raise for broad checklist queries if recall suffers.
AGENT_MAX_SUB_QUERIES = int(os.getenv("AGENT_MAX_SUB_QUERIES", "3"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))  # low temperature for grounded citation tasks
# gemini-3.7-flash is the current Flash generation: built for complex coding,
# agentic workflows and multi-step execution, which is what the agent pipeline
# does. 3.6-flash remains selectable as the previous generation.
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-3.7-flash")  # Gemini model

# Explicit Gemini context caching of the (stable) system-instruction prefix.
# Gemini 3.x Flash already does IMPLICIT caching for free; explicit caching adds
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
# Phase 8 — Secondary LLM provider (OpenRouter)
# ============================================================================
# Default backend for LLM calls and the cross-vendor fallback. When the chosen
# provider's models are all exhausted/circuit-open, failover crosses to the
# other provider's default model so a whole-vendor outage isn't a total failure.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")                    # gemini|openrouter
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "openrouter")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Curated allowlist offered to the user in the model dropdown (comma-separated).
# Bare name → Gemini; slug with "/" → OpenRouter. First entry is the default.
_raw_selectable = os.getenv(
    "LLM_SELECTABLE_MODELS",
    # Current Flash first (the default), then the previous generation, then a
    # cheap high-throughput option for routine calls, then cross-vendor entries.
    # The cross-vendor slugs matter beyond user choice: failover picks a "/"-shaped
    # slug from this list, so an all-Gemini list would leave nothing to fail over to.
    "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash-lite,"
    "anthropic/claude-haiku,openai/gpt-5.4-nano",
)
LLM_SELECTABLE_MODELS = [m.strip() for m in _raw_selectable.split(",") if m.strip()]
# How long the enriched OpenRouter /models catalog is cached (seconds).
MODELS_CACHE_TTL = int(os.getenv("MODELS_CACHE_TTL", "3600"))

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
# Phase 2 — surfaced confidence + abstention
# ============================================================================
# Aggregate a per-answer confidence from the final reflexion feedback and expose
# an explicit "insufficient evidence" path. Weights are uncalibrated until the
# Phase 1 eval reliability curve exists — treat the number as directional.
ANSWER_CONFIDENCE_ENABLE = os.getenv("ANSWER_CONFIDENCE_ENABLE", "true").lower() == "true"
# Abstain when the answer is grounded (high faithfulness) but the corpus does not
# cover the query (completeness below this floor) after the reflexion budget is spent.
ABSTAIN_COMPLETENESS_FLOOR = float(os.getenv("ABSTAIN_COMPLETENESS_FLOOR", "0.5"))

# ============================================================================
# Version
# ============================================================================
# Surfaced in the OpenAPI spec and /health, so it is what an operator reads when
# asking "which build is this?" — it had drifted two releases behind the README.
VERSION = "2.5.0-dev"

# ============================================================================
# Chat / Session
# ============================================================================
CHAT_HISTORY_MAX_TURNS = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "20"))
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
