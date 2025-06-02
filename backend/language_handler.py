import logging
import json
import os
import re
import httpx
from langdetect import detect, DetectorFactory, LangDetectException

log = logging.getLogger(__name__)
DetectorFactory.seed = 0

# --- Configuration ---
KEYWORDS_FILE = os.path.join(os.path.dirname(__file__), "language_keywords.json")

DZIRIBERT_DETECTION_URL = os.getenv("DZIRIBERT_DETECTION_SERVICE_URL", "http://localhost:8001/process_darija")
ENG_TO_DARIJA_TRANSLATION_URL = os.getenv("ENG_TO_DARIJA_TRANSLATION_SERVICE_URL", "http://localhost:8001/translate_en_to_darija")

DZIRIBERT_TIMEOUT = 5.0
TRANSLATION_TIMEOUT = 20.0
DZIRIBERT_CONFIDENCE_THRESHOLD = 0.7

SUPPORTED_LANGUAGES_MAP = {"en": "English", "fr": "French", "ar": "Arabic"}
INTERNAL_LANG_CODES = list(SUPPORTED_LANGUAGES_MAP.keys())
DEFAULT_LANGUAGE_CODE = "en"
DEFAULT_LANGUAGE_NAME = SUPPORTED_LANGUAGES_MAP[DEFAULT_LANGUAGE_CODE]

_keywords_data = {}
try:
    if os.path.exists(KEYWORDS_FILE):
        with open(KEYWORDS_FILE, 'r', encoding='utf-8') as f:
            _keywords_data = json.load(f)
        log.info(f"Successfully loaded keywords from {KEYWORDS_FILE}")
    else:
        log.error(f"CRITICAL: Keywords file '{KEYWORDS_FILE}' not found.")
        _keywords_data = {
            "explicit_requests": {}, 
            "darija_indicators_latin": [], 
            "darija_indicators_arabic": [],
            "problem_solved_keywords": {},
            "session_reset_keywords": {},
            "simple_closing_remarks": {},
            "image_component_keywords": {},
            "list_all_issues_keywords": {} # Added for completeness
        }
except Exception as e:
    log.error(f"CRITICAL: Error loading keywords: {e}", exc_info=True)
    _keywords_data = {
        "explicit_requests": {}, 
        "darija_indicators_latin": [], 
        "darija_indicators_arabic": [],
        "problem_solved_keywords": {},
        "session_reset_keywords": {},
        "simple_closing_remarks": {},
        "image_component_keywords": {},
        "list_all_issues_keywords": {}
    }

DARIJA_EXPLICIT_REQUEST_KEYWORDS = _keywords_data.get("explicit_requests", {}).get("darija", [])
FRENCH_REQUEST_KEYWORDS = _keywords_data.get("explicit_requests", {}).get("french", [])
ARABIC_MSA_REQUEST_KEYWORDS = _keywords_data.get("explicit_requests", {}).get("arabic_msa", [])
ENGLISH_REQUEST_KEYWORDS = _keywords_data.get("explicit_requests", {}).get("english", [])
DARIJA_LATIN_INDICATORS = _keywords_data.get("darija_indicators_latin", [])
DARIJA_ARABIC_INDICATORS = _keywords_data.get("darija_indicators_arabic", [])
DARIJA_INDICATOR_THRESHOLD = _keywords_data.get("darija_indicator_threshold", 2)


async def _call_darija_detection_service(text: str) -> dict | None:
    if not DZIRIBERT_DETECTION_URL:
        log.debug("DziriBERT detection service URL not configured.")
        return None
    if not text or not text.strip(): return None
    payload = {"text": text}
    try:
        async with httpx.AsyncClient(timeout=DZIRIBERT_TIMEOUT) as client:
            response = await client.post(DZIRIBERT_DETECTION_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            log.info(f"Darija Detection Service responded for '{text[:30]}...': {data}")
            return data
    except httpx.RequestError as e:
        log.error(f"Network error calling Darija Detection Service at {DZIRIBERT_DETECTION_URL}: {e}")
    except httpx.HTTPStatusError as e:
        log.error(f"HTTP error {e.response.status_code} from Darija Detection Service: {e.response.text[:200]}")
    except json.JSONDecodeError as e:
        log.error(f"JSON decode error from Darija Detection Service: {e}")
    except Exception as e:
        log.error(f"Unexpected error calling Darija Detection Service: {e}", exc_info=False)
    return None

async def translate_english_to_darija_via_service(english_text: str) -> str | None:
    if not ENG_TO_DARIJA_TRANSLATION_URL:
        log.warning("English-to-Darija translation service URL not configured. Cannot translate.")
        return None
    if not english_text or not english_text.strip():
        log.warning("Empty text provided for Eng-to-Darija translation.")
        return english_text

    payload = {"text_to_translate": english_text}
    log.info(f"Calling Eng-to-Darija translation service for: '{english_text[:50]}...'")
    try:
        async with httpx.AsyncClient(timeout=TRANSLATION_TIMEOUT) as client:
            response = await client.post(ENG_TO_DARIJA_TRANSLATION_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            translated = data.get("translated_text")
            if translated:
                log.info(f"Eng-to-Darija translation result: '{translated[:50]}...'")
                return translated.strip()
            else:
                log.warning(f"Eng-to-Darija translation service returned no 'translated_text'. Response: {data}")
                return None
    except httpx.RequestError as e:
        log.error(f"Network error calling Eng-to-Darija translation service: {e}")
    except httpx.HTTPStatusError as e:
        log.error(f"HTTP error {e.response.status_code} from Eng-to-Darija translation service: {e.response.text[:200]}")
    except json.JSONDecodeError as e:
        log.error(f"JSON decode error from Eng-to-Darija translation service: {e}")
    except Exception as e:
        log.error(f"Unexpected error calling Eng-to-Darija translation service: {e}", exc_info=False)
    return None

async def detect_language_and_intent(text: str) -> tuple[str, str | None, dict | None]:
    text_lower_for_latin_keywords = text.lower()
    final_detected_lang_code = DEFAULT_LANGUAGE_CODE
    specific_dialect_or_request_type = None
    detection_analysis_result = None

    if any(k.lower() in text_lower_for_latin_keywords for k in DARIJA_EXPLICIT_REQUEST_KEYWORDS if k.isascii()) or \
       any(k in text for k in DARIJA_EXPLICIT_REQUEST_KEYWORDS if not k.isascii()):
        log.info(f"Explicit Darija request in '{text[:70]}...'")
        final_detected_lang_code = "ar"
        specific_dialect_or_request_type = "darija_explicit_request"
        detection_analysis_result = await _call_darija_detection_service(text)
        if detection_analysis_result and detection_analysis_result.get("is_darija"):
            specific_dialect_or_request_type = "darija_confirmed_dziribert_explicit_request"
    elif any(k.lower() in text_lower_for_latin_keywords for k in FRENCH_REQUEST_KEYWORDS):
        return "fr", "french_request", None
    elif any(k.lower() in text_lower_for_latin_keywords for k in ARABIC_MSA_REQUEST_KEYWORDS if k.isascii()) or \
         any(k in text for k in ARABIC_MSA_REQUEST_KEYWORDS if not k.isascii()):
        return "ar", "arabic_msa_request", None
    elif any(k.lower() in text_lower_for_latin_keywords for k in ENGLISH_REQUEST_KEYWORDS):
        return "en", "english_request", None

    if specific_dialect_or_request_type: 
        return final_detected_lang_code, specific_dialect_or_request_type, detection_analysis_result

    initial_lib_detection = None
    tentative_lang_code = DEFAULT_LANGUAGE_CODE
    try:
        if len(text.strip()) >= 3: 
            # Use detect_langs for more robustness if langdetect is used
            # detections = DetectorFactory().get_detector().detect_langs(text)
            # For simplicity and consistency with original, using detect()
            initial_lib_detection = detect(text) # Simpler, but can be less robust for mixed scripts
            if initial_lib_detection in INTERNAL_LANG_CODES: tentative_lang_code = initial_lib_detection
            elif initial_lib_detection.startswith("ar"): tentative_lang_code = "ar" 
            elif initial_lib_detection == 'fra': tentative_lang_code = 'fr'
            elif initial_lib_detection == 'eng': tentative_lang_code = 'en'
            # else: log.debug(f"langdetect result '{initial_lib_detection}' not in primary map, using default or previous.")
        else: 
            if DZIRIBERT_DETECTION_URL:
                dz_short_text_analysis = await _call_darija_detection_service(text)
                if dz_short_text_analysis and dz_short_text_analysis.get("is_darija") and \
                   dz_short_text_analysis.get("confidence", 0) >= DZIRIBERT_CONFIDENCE_THRESHOLD:
                    return "ar", "darija_confirmed_dziribert_short_text", dz_short_text_analysis
    except LangDetectException:
        log.warning(f"langdetect failed for: '{text[:70]}...'")
    except Exception as e: 
        log.error(f"Unexpected error during langdetect processing: {e}", exc_info=True)
    final_detected_lang_code = tentative_lang_code

    should_call_detection_service = False
    if final_detected_lang_code == "ar": should_call_detection_service = True

    darija_indicator_score = 0
    if DARIJA_LATIN_INDICATORS:
        for keyword in DARIJA_LATIN_INDICATORS:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower_for_latin_keywords):
                darija_indicator_score += 1
    if DARIJA_ARABIC_INDICATORS:
        for keyword in DARIJA_ARABIC_INDICATORS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                darija_indicator_score += 1
    
    if darija_indicator_score >= DARIJA_INDICATOR_THRESHOLD:
        should_call_detection_service = True
        final_detected_lang_code = "ar"

    if should_call_detection_service and not detection_analysis_result:
        detection_analysis_result = await _call_darija_detection_service(text)
        
    if detection_analysis_result:
        is_darija_confirmed_by_service = detection_analysis_result.get("is_darija")
        confidence_from_service = detection_analysis_result.get("confidence", 0)
        if is_darija_confirmed_by_service and confidence_from_service >= DZIRIBERT_CONFIDENCE_THRESHOLD:
            final_detected_lang_code = "ar"
            specific_dialect_or_request_type = "darija_confirmed_dziribert"
        elif is_darija_confirmed_by_service is False: 
            if final_detected_lang_code == "ar": 
                specific_dialect_or_request_type = "arabic_generic_not_darija_dzb"
        else: 
            if final_detected_lang_code == "ar":
                if darija_indicator_score >= DARIJA_INDICATOR_THRESHOLD:
                    specific_dialect_or_request_type = "darija_heuristic_keywords_dzb_inconclusive"
                else:
                    specific_dialect_or_request_type = "arabic_generic_dzb_inconclusive"
    
    if final_detected_lang_code == "ar" and not specific_dialect_or_request_type:
        if darija_indicator_score >= DARIJA_INDICATOR_THRESHOLD:
            specific_dialect_or_request_type = "darija_heuristic_keywords_only"
        else:
            specific_dialect_or_request_type = "arabic_generic_content"

    log.info(f"FINAL Lang Decision: Code='{final_detected_lang_code}', SpecificType='{specific_dialect_or_request_type}' for: '{text[:70]}...'")
    return final_detected_lang_code, specific_dialect_or_request_type, detection_analysis_result

def get_language_name(lang_code: str) -> str:
    return SUPPORTED_LANGUAGES_MAP.get(lang_code, DEFAULT_LANGUAGE_NAME)

def get_localized_keywords(key_group: str, lang_code: str) -> list:
    if not _keywords_data or key_group not in _keywords_data:
        log.warning(f"Keyword group '{key_group}' not found in keywords data.")
        return []
    
    specific_lang_group = _keywords_data.get(key_group, {})
    keywords_for_lang = specific_lang_group.get(lang_code, [])
    
    if not keywords_for_lang and lang_code != DEFAULT_LANGUAGE_CODE:
        log.debug(f"No keywords for lang '{lang_code}' in group '{key_group}'. Trying default '{DEFAULT_LANGUAGE_CODE}'.")
        keywords_for_lang = specific_lang_group.get(DEFAULT_LANGUAGE_CODE, [])
        
    if not keywords_for_lang:
        log.debug(f"No keywords found for lang '{lang_code}' or default '{DEFAULT_LANGUAGE_CODE}' in group '{key_group}'.")
        return []
        
    return keywords_for_lang