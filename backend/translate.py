# backend/translate.py

from deep_translator import GoogleTranslator
from langdetect import detect

# Load all supported languages from deep_translator
SUPPORTED_LANGS = GoogleTranslator().get_supported_languages(as_dict=True)
SUPPORTED_CODES = set(SUPPORTED_LANGS.values())


# ----------------------------
# Language Detection
# ----------------------------
def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en"   # fallback


# ----------------------------
# Language Normalization
# ----------------------------
def safe_lang(code: str) -> str:
    """
    Convert detected language code to one supported by GoogleTranslator.
    Example: zh-cn → zh-CN, he → iw, etc.
    """
    code = code.lower().strip()

    # special fixes
    SPECIAL = {
        "zh-cn": "zh-CN",
        "zh-tw": "zh-TW",
        "he": "iw",         # Hebrew uses iw in Google Translate
        "pt-br": "pt",
    }
    if code in SPECIAL:
        return SPECIAL[code]

    # if already supported
    if code in SUPPORTED_CODES:
        return code

    # if language name is provided (rare)
    if code in SUPPORTED_LANGS:
        return SUPPORTED_LANGS[code]

    return "en"  # fallback if unrecognized


# ----------------------------
# Translate ANY → English
# ----------------------------
def translate_to_english(text: str) -> str:
    try:
        src = detect_lang(text)
        src = safe_lang(src)

        if src == "en":
            return text

        return GoogleTranslator(source=src, target="en").translate(text)

    except Exception:
        return text  # fallback


# ----------------------------
# Translate English → TARGET LANG
# ----------------------------
def translate_from_english(text: str, target_lang: str = "en") -> str:
    try:
        target = safe_lang(target_lang)

        if target == "en":
            return text

        return GoogleTranslator(source="en", target=target).translate(text)

    except Exception:
        return text  # fallback
