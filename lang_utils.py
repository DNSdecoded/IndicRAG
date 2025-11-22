"""
Language detection and mapping utilities.
"""

import langdetect
from typing import Optional
import config


def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of the input text.
    
    Args:
        text: Input text to detect language for
        
    Returns:
        ISO 639-1 language code (e.g., 'hi', 'en', 'ta') or None if detection fails
    """
    try:
        # langdetect returns ISO 639-1 codes
        lang_code = langdetect.detect(text)
        return lang_code
    except Exception as e:
        print(f"Language detection failed: {e}")
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
