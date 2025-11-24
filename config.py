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
PAPERS_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)
MODELS_CACHE_DIR.mkdir(exist_ok=True)


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
CHUNK_OVERLAP = 200  # overlap between chunks
MIN_CHUNK_SIZE = 100  # minimum chunk size to keep

# ============================================================================
# Retrieval Parameters
# ============================================================================
DEFAULT_TOP_K = 8  # number of chunks to retrieve
MAX_CONTEXT_CHUNKS = 5  # maximum chunks to include in LLM context
MAX_CONTEXT_LENGTH = 4000  # maximum total characters in context

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
LLM_MAX_TOKENS = 2048  # maximum tokens to generate
LLM_TEMPERATURE = 0.3  # lower temperature for factual responses
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")  # Gemini model

# LLM API Key (required for Gemini)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Note: For Gemini, you can use:
# - gemini-2.5-flash (recommended: fast and cost-effective)
# - gemini-2.5-pro (higher quality)
# - gemini-flash-latest (always uses latest flash model)

# ============================================================================
# Prompt Templates
# ============================================================================
SYSTEM_PROMPT = """You are a scientific assistant helping users understand research papers. 
Your task is to answer questions based ONLY on the provided context from scientific literature.

Guidelines:
- Use only information from the provided context
- If the context doesn't contain enough information, say so clearly
- Provide clear, simple explanations suitable for a general audience
- Include brief citations using [1], [2], etc. when referencing specific papers
- Do not make up information or use external knowledge
- If discussing medical topics, add: "This is not medical advice; consult a healthcare professional"
"""

QUERY_PROMPT_TEMPLATE = """Context from scientific papers:
{context}

Question: {question}

Please answer the question in {language} using only the information from the context above. 
Use clear, simple language that a non-expert can understand.
"""

# ============================================================================
# PDF Processing
# ============================================================================
# Common header/footer patterns to remove
NOISE_PATTERNS = [
    r"Page \d+ of \d+",
    r"^\d+$",  # standalone page numbers
    r"Â©.*\d{4}",  # copyright notices
    r"doi:.*",
    r"arXiv:\d+\.\d+",
]

# Section headers to detect
SECTION_HEADERS = [
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
]

# ============================================================================
# Logging
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
