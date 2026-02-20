"""
Translation utilities for Strategy B (English reasoning + translation).
Uses IndicTrans2 models for high-quality Indic language translation.
"""

from typing import Optional
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import config

logger = logging.getLogger(__name__)


# Global model caches
_translation_model = None
_translation_tokenizer = None

def load_translation_model():
    """
    Load translation model.
    """
    global _translation_model, _translation_tokenizer
    
    if _translation_model is not None:
        return _translation_model, _translation_tokenizer
        
    model_name = config.TRANSLATION_MODEL_EN_TO_INDIC
    logger.info(f"Loading translation model: {model_name}")
    logger.info("This may take several minutes on first run (model is ~2.5GB)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _translation_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_CACHE_DIR),
        trust_remote_code=True
    )
    
    _translation_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        cache_dir=str(config.MODELS_CACHE_DIR),
        trust_remote_code=True
    ).to(device).eval()  # Set to eval mode for inference
    
    logger.info(f"Model loaded on device: {device}")
    return _translation_model, _translation_tokenizer


NLLB_LANG_MAP = {
    "hi": "hin_Deva", "ta": "tam_Taml", "te": "tel_Telu",
    "bn": "ben_Beng", "mr": "mar_Deva", "gu": "guj_Gujr",
    "kn": "kan_Knda", "ml": "mal_Mlym", "pa": "pan_Guru",
    "or": "ory_Orya", "en": "eng_Latn"
}

def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    max_length: int = 512
) -> str:
    """
    Translate text between English and Indic languages.
    
    Args:
        text: Text to translate
        source_lang: Source language code (e.g., 'en', 'hi', 'ta')
        target_lang: Target language code (e.g., 'en', 'hi', 'ta')
        max_length: Maximum length of generated translation
        
    Returns:
        Translated text
    """
    if source_lang == target_lang:
        return text
        
    if source_lang not in NLLB_LANG_MAP or target_lang not in NLLB_LANG_MAP:
        raise ValueError(
            f"Unsupported translation: {source_lang} -> {target_lang}. "
            f"Only English and supported Indic languages are allowed."
        )
    
    # Load model
    model, tokenizer = load_translation_model()
    
    # Set source language for the tokenizer
    tokenizer.src_lang = NLLB_LANG_MAP[source_lang]
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Target language ID
    target_id = tokenizer.convert_tokens_to_ids(NLLB_LANG_MAP[target_lang])
    
    # Generate translation
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            max_length=max_length,
            num_beams=5,
            early_stopping=True
        )
    
    # Decode
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return translation


def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate Indic language text to English.
    
    Args:
        text: Text in Indic language
        source_lang: Source language code (e.g., 'hi', 'ta')
        
    Returns:
        English translation
    """
    return translate_text(text, source_lang, "en")


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate English text to Indic language.
    
    Args:
        text: English text
        target_lang: Target language code (e.g., 'hi', 'ta')
        
    Returns:
        Translation in target language
    """
    return translate_text(text, "en", target_lang)


if __name__ == "__main__":
    # Test translation
    print("Testing IndicTrans2 Translation")
    print("=" * 60)
    
    # Note: This test requires downloading large models (~2.5GB each)
    # Uncomment to test
    
    # Test English to Hindi
    print("\n1. Testing English -> Hindi...")
    en_text = "Diabetes is a chronic disease that affects blood sugar levels."
    
    try:
        hi_translation = translate_from_english(en_text, "hi")
        print(f"English: {en_text}")
        print(f"Hindi: {hi_translation}")
    except Exception as e:
        print(f"Translation failed: {e}")
        print("Note: IndicTrans2 models are large (~2.5GB). Ensure you have enough disk space.")
    
    # Test Hindi to English
    print("\n2. Testing Hindi -> English...")
    hi_text = "मधुमेह एक पुरानी बीमारी है जो रक्त शर्करा के स्तर को प्रभावित करती है।"
    
    try:
        en_translation = translate_to_english(hi_text, "hi")
        print(f"Hindi: {hi_text}")
        print(f"English: {en_translation}")
    except Exception as e:
        print(f"Translation failed: {e}")
    
    print("\nNote: Translation quality may vary. IndicTrans2 is optimized for Indic languages.")
