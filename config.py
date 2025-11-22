"""
Configuration and constants for the multilingual RAG system.
"""

import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
PAPERS_DIR = PROJECT_ROOT / "papers"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
MODELS_CACHE_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
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
    "hi": "हिंदी",  # Hindi
    "mr": "मराठी",  # Marathi
    "ta": "தமிழ்",  # Tamil
    "te": "తెలుగు",  # Telugu
    "bn": "বাংলা",  # Bengali
    "gu": "ગુજરાતી",  # Gujarati
    "kn": "ಕನ್ನಡ",  # Kannada
    "ml": "മലയാളം",  # Malayalam
    "pa": "ਪੰਜਾਬੀ",  # Punjabi
    "or": "ଓଡ଼ିଆ",  # Odia
    "en": "English",
}

# Supported Indic languages for translation
INDIC_LANGUAGES = ["hi", "mr", "ta", "te", "bn", "gu", "kn", "ml", "pa", "or"]

# ============================================================================
# Translation Models (Strategy B)
# ============================================================================
TRANSLATION_MODEL_EN_TO_INDIC = "ai4bharat/indictrans2-en-indic-1B"
TRANSLATION_MODEL_INDIC_TO_EN = "ai4bharat/indictrans2-indic-en-1B"

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
    r"©.*\d{4}",  # copyright notices
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
