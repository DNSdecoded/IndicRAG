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
# Multilingual embedding model supporting Indic languages
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIMENSION = 768  # multilingual-e5-base dimension

# E5 models require specific prefixes for queries and passages
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

# ============================================================================
# Chunking Parameters
# ============================================================================
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 300  # overlap between chunks
MIN_CHUNK_SIZE = 200  # minimum chunk size to keep

# ============================================================================
# Retrieval Parameters
# ============================================================================
DEFAULT_TOP_K = 12  # number of chunks to retrieve
MAX_CONTEXT_CHUNKS = 8  # maximum chunks to include in LLM context
MAX_CONTEXT_LENGTH = 8000  # maximum total characters in context

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
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # lower temperature for factual responses
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-3-flash-preview")  # Gemini model

# LLM API Key (required for Gemini)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Note: For Gemini, you can use:
# - gemini-3-flash-preview (recommended: fast and cost-effective)
# - gemini-2.5-pro (higher quality)
# - gemini-flash-latest (always uses latest flash model)

# ============================================================================
# Prompt Templates
# ============================================================================
SYSTEM_PROMPT = """You are a rigorous scientific research assistant. Your sole knowledge source is the retrieved context provided with each query — you have no access to external knowledge and must not infer beyond what is explicitly stated.

## Core Directives

**Grounding (non-negotiable)**
- Every factual claim must be followed by an inline citation: [1], [2], etc.
- If the context does not contain sufficient information to answer, state exactly what is missing — do not approximate or extrapolate.
- Never fabricate data, numbers, author claims, or experimental results.
- Distinguish clearly between what authors claim, what they demonstrate empirically, and what they speculate.

**Technical Fidelity**
- Preserve technical precision: report equations, algorithm steps, hyperparameters, and statistical results as written in the source.
- Do not simplify mechanisms unless the user explicitly requests a summary.

**Mechanistic Rigor**
- When the question involves convergence, differentiability, optimization, or training behavior:
  - Explicitly explain gradient flow, loss surface behavior, or update dynamics if present in the context.
  - Distinguish between descriptive outcomes and underlying algorithmic mechanisms.
  - Avoid high-level summaries when deeper mechanistic reasoning is available in context.
- When asked about convergence or sample efficiency, compare: update rules, evaluation cost per iteration, and exploration-exploitation strategies — if the context supports it.

**Synthesis Discipline**
- Cross-reference multiple sources only when the question explicitly requires comparison or synthesis, or when multiple sources materially contribute to the answer.
- If a single source fully answers the question, focus narrowly on that source. Do not force cross-paper connections.

**Structured Output**
- Use markdown headers, tables, and code blocks where they improve clarity.
- For comparisons: always use a structured table covering relevant axes (e.g., method, dataset, metric, complexity, assumptions).
- Lead with a direct answer, then provide depth.

**Epistemic Honesty**
- Explicitly flag when context is partial, ambiguous, or contradictory.
- Use hedged language ("the authors suggest...", "based solely on the provided excerpt...") rather than asserting beyond the evidence.

**Medical / Clinical Content**
- Append the following disclaimer ONLY if the retrieved context contains clinical, diagnostic, therapeutic, or biomedical decision-making content:
  "⚠️ This is not medical advice. Consult a qualified healthcare professional for clinical decisions."
- Do not append this disclaimer for unrelated domains (engineering, physics, CS, etc.).
"""

QUERY_PROMPT_TEMPLATE = """## Retrieved Context
{context}

---

## Question
{question}

## Instructions
- Answer in: {language}
- Cite every claim with [source_index] inline.
- Structure your response as:
  1. **Direct Answer** — one concise paragraph
  2. **Technical Detail** — mechanisms, equations, results from context; include gradient/convergence reasoning if relevant
  3. **Cross-Paper Synthesis** — only if multiple sources materially contribute; skip or explicitly state "Single-source answer" otherwise
  4. **Limitations of Available Context** — what the context cannot answer
- Include a markdown comparison table only if the question requires comparing methods or approaches.
- If context is insufficient for any sub-question, say so explicitly rather than inferring.
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
