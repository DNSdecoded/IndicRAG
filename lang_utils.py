"""
Language detection and mapping utilities.
"""

import regex as re
import langdetect
from langdetect import DetectorFactory
from typing import Optional
import logging
import config

DetectorFactory.seed = 0  # Make detection deterministic

logger = logging.getLogger(__name__)

_SCRIPT_RANGES = {
    'hi': r'\p{Devanagari}', 'ta': r'\p{Tamil}',
    'te': r'\p{Telugu}', 'bn': r'\p{Bengali}', 'gu': r'\p{Gujarati}',
    'kn': r'\p{Kannada}', 'ml': r'\p{Malayalam}', 'pa': r'\p{Gurmukhi}',
    'or': r'\p{Oriya}', 'mr': r'\p{Devanagari}',
}


def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of the input text.

    Uses Unicode script detection first (unambiguous for most Indic scripts),
    then falls back to langdetect for Latin-script text.
    """
    for code, rng in _SCRIPT_RANGES.items():
        if re.search(rng, text):
            return code
    if len(text.strip()) < 15:
        return 'en'
    try:
        lang_code = langdetect.detect(text)
        return lang_code
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return None


def get_language_name(lang_code: str) -> str:
    """
    Get the native name of a language from its ISO code.
    
    Args:
        lang_code: ISO 639-1 language code (e.g., 'hi', 'en')
        
    Returns:
        Native language name (e.g., 'हिंदी', 'English')
        Returns the code itself if not found in mapping
    """
    return config.LANGUAGE_NAMES.get(lang_code, lang_code)


def is_indic_language(lang_code: str) -> bool:
    """
    Check if a language code corresponds to an Indian language.
    
    Args:
        lang_code: ISO 639-1 language code
        
    Returns:
        True if the language is an Indian language, False otherwise
    """
    return lang_code in config.INDIC_LANGUAGES


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and cleaning up.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


if __name__ == "__main__":
    # Test language detection
    test_texts = {
        "hi": "क्या यह दवा डायबिटीज़ के इलाज में मदद करती है?",
        "ta": "இந்த மருந்து நீரிழிவு நோய்க்கு உதவுமா?",
        "en": "How does this drug help with diabetes treatment?",
        "mr": "हे औषध मधुमेहाच्या उपचारात कसे मदत करते?",
        "te": "ఈ మందు మధుమేహ చికిత్సలో ఎలా సహాయపడుతుంది?",
    }
    
    print("Language Detection Tests:")
    print("-" * 60)
    for expected_lang, text in test_texts.items():
        detected = detect_language(text)
        lang_name = get_language_name(detected) if detected else "Unknown"
        is_indic = is_indic_language(detected) if detected else False
        
        print(f"Text: {text[:50]}...")
        print(f"Expected: {expected_lang}, Detected: {detected}")
        print(f"Language Name: {lang_name}")
        print(f"Is Indic: {is_indic}")
        print("-" * 60)
