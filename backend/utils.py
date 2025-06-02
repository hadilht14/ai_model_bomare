# backend/utils.py
import re
import logging

log = logging.getLogger(__name__)

# Words that are unlikely to be model names on their own, especially if short
# (You can refine this list based on your data and observation)
AFFIRMATIVE_WORDS_UTIL = ["yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "affirmative", "indeed", "certainly", "please", "go", "do"]
NEGATIVE_WORDS_UTIL = ["no", "nope", "nah", "negative", "don't", "do not", "cancel", "stop", "not", "never"]
COMMON_TV_TERMS_UTIL = [
    "TV","MODEL","ISSUE","HELP","FIX","PROBLEM","POWER","SOUND","VIDEO","REMOTE","INPUT", "SCREEN",
    "DISPLAY", "IMAGE", "AUDIO", "CHANNEL", "PICTURE", "HDMI", "USB", "MENU", "THANKS", "THANK", 
    "YOU", "HELLO", "HI", "GOOD", "DAY", "WHAT", "HOW", "IS", "MY", "THE", "A", "AN", "FOR", "WITH", 
    "ON", "IT", "ITS", "SHOW", "ME", "VIEW", "OF", "LIST", "COMPONENTS", "DIAGRAM", "SCHEMATIC", 
    "PHOTO", "LAYOUT", "SORRY", "ERROR", "INVALID", "GENERAL", "SPECIFIC", "STANDARD"
]
COMMON_BRAND_NAMES_UTIL = ["SONY", "LG", "SAMSUNG", "VIZIO", "TCL", "HISENSE", "PHILIPS", "PANASONIC"] # Add more as needed
NUMBERS_AS_WORDS_UTIL = [str(n) for n in range(1000)] # Avoid short numbers being models

ALL_COMMON_WORDS_UTIL = list(set(
    AFFIRMATIVE_WORDS_UTIL + NEGATIVE_WORDS_UTIL + COMMON_TV_TERMS_UTIL + 
    COMMON_BRAND_NAMES_UTIL + NUMBERS_AS_WORDS_UTIL
))


def extract_tv_model_from_query(query: str) -> str | None:
    """
    Extracts a potential TV model name from a user query using regex and heuristics.
    """
    if not query or not query.strip():
        return None
    
    query_upper = query.upper()
    
    # Pattern 1: Keywords like "model [is/number/no] XXXXX"
    # e.g., "model is UA55C300", "tv model number EL.RT2864-FG48"
    keyword_pattern_full_sentence = r"(?:MODEL\s*(?:IS|:|NUMBER\s*(?:IS|NO\.?)?)?|TV\s*MODEL\s*(?:IS|:|NUMBER\s*(?:IS|NO\.?)?)?)\s+([A-Z0-9_.\-]+[A-Z0-9])"
    match_full = re.search(keyword_pattern_full_sentence, query_upper)
    if match_full:
        model_candidate = match_full.group(1).strip()
        # Basic validation: length, not just "TV" or "MODEL"
        if len(model_candidate) >= 3 and model_candidate not in ["TV", "MODEL", "TELEVISION"]:
            log.debug(f"UTILS_MODEL_EXTRACT (Keyword Full Sentence): Extracted model '{model_candidate}'.")
            return model_candidate

    # Pattern 2: More general "TV XXXXX" or "MODEL XXXXX" where XXXXX is alphanumeric with possible specials
    # e.g., "tv UA55C300 problem", "model EL.RT2864-FG48 manual"
    keyword_pattern_phrase = r"\b(?:TV|TELEVISION|MODEL)\s+([A-Z0-9_.\-]{3,25})\b" # Model length 3-25
    match_phrase = re.search(keyword_pattern_phrase, query_upper)
    if match_phrase:
        model_candidate = match_phrase.group(1).strip()
        has_letter = bool(re.search(r"[A-Z]", model_candidate))
        has_digit = bool(re.search(r"[0-9]", model_candidate))
        # Require mix or longer length, and not a common word
        if ((has_letter and has_digit) or len(model_candidate) > 4) and \
           model_candidate not in ["TV", "MODEL", "TELEVISION"] and \
           model_candidate not in COMMON_BRAND_NAMES_UTIL: # Avoid just "SONY" if it's "TV SONY"
            log.debug(f"UTILS_MODEL_EXTRACT (Keyword Phrase): Extracted model '{model_candidate}'.")
            return model_candidate

    # Heuristic: Look for standalone alphanumeric strings with typical model characteristics.
    # This is important for inputs that are *just* the model name or embedded within a sentence.
    potential_model_tokens = re.findall(r"([A-Z0-9_.\-]+)", query_upper) 

    for token in potential_model_tokens:
        token_upper = token.upper() # Already upper from query_upper but good practice if tokenizing differently
        if token_upper in ALL_COMMON_WORDS_UTIL:
            continue
        if not (3 <= len(token_upper) <= 25): # Typical model length
            continue

        has_letter = bool(re.search(r"[A-Z]", token_upper))
        has_digit = bool(re.search(r"[0-9]", token_upper))
        has_special_char = bool(re.search(r"[.\-_]", token_upper)) 

        # Score based heuristics (can be tuned)
        score = 0
        if has_letter: score += 2 # Letters are strong indicators
        if has_digit: score += 2  # Digits are strong indicators
        if has_special_char: score += 1 # Special chars like '-' or '.' are common
        if len(token_upper) > 5: score +=1 
        if len(token_upper) > 8: score +=1 # Longer tokens are more likely models
        
        # Check for common TV model prefixes
        common_prefixes = ("EL", "UA", "QN", "OLED", "KD", "KDL", "XR", "UN", "LN", "UE", "LC", "TH", "TC")
        if token_upper.startswith(common_prefixes): score += 2
        # Check for common TV model suffixes (less common but can help)
        # common_suffixes = ("US", "UK", "EU", "ZA", "FXZA")
        # if any(token_upper.endswith(s) for s in common_suffixes): score +=1

        # Require a minimum score and a mix of char types or specific structures
        if score >= 3 and (has_letter and has_digit): 
            # Avoid things like "V1.0" or "HDMI1" if they are too short without other strong indicators
            if len(token_upper) < 4 and not has_special_char and (token_upper.isalpha() or token_upper.isdigit()):
                 continue 

            # Check for too many special characters or leading/trailing ones that are not typical
            if token_upper.count('.') > 3 or token_upper.count('-') > 4 or token_upper.count('_') > 3 or \
               token_upper.startswith(("_")) or token_upper.endswith(("-",".", "_")): # Allow leading '.' or '-'
                continue
            
            # Further check: ensure it doesn't consist *only* of digits if it's short (e.g. "2024")
            if token_upper.isdigit() and len(token_upper) < 5:
                continue

            log.debug(f"UTILS_MODEL_EXTRACT (Heuristic Token): Potential model '{token_upper}'. Score: {score}")
            return token_upper 

    log.debug(f"UTILS_MODEL_EXTRACT: No model reliably found in query: '{query[:70]}...'")
    return None