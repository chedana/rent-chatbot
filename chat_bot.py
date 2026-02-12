import os
import sys
import re
import json
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
try:
    from qdrant_client import QdrantClient, models
except Exception:
    QdrantClient = None
    models = None


from openai import OpenAI

QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL    = os.environ.get("QWEN_MODEL", "./Qwen3-14B")  # 你启动时的 model 名称
QWEN_API_KEY  = os.environ.get("OPENAI_API_KEY", "dummy")

qwen_client = OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)

EXTRACT_SYSTEM = """You output STRICT JSON only (no markdown, no explanation).
Schema:
{
  "max_rent_pcm": number|null,
  "bedrooms": int|null,
  "bedrooms_op": string|null,
  "bathrooms": number|null,
  "bathrooms_op": string|null,
  "available_from": string|null,
  "furnish_type": string|null,
  "let_type": string|null,
  "property_type": string|null,
  "min_tenancy_months": number|null,
  "min_size_sqm": number|null,
  "min_size_sqft": number|null,
  "location_keywords": string[],
  "must_have_keywords": string[],
  "k": int|null
}
Rules:
- location_keywords are place names/areas/postcodes (e.g., "Canary Wharf", "E14", "Shoreditch").
- must_have_keywords are requirements/features (e.g., "balcony", "pet friendly", "near tube").
- bedrooms_op must be one of: "eq", "gte", or null.
- Set bedrooms/bedrooms_op only for hard constraints:
  - "at least/minimum/>= X bedrooms" -> {"bedrooms": X, "bedrooms_op": "gte"}
  - "exactly/only X bedrooms" -> {"bedrooms": X, "bedrooms_op": "eq"}
  - soft wording (prefer/ideally/nice to have) -> bedrooms = null, bedrooms_op = null
- bathrooms_op must be one of: "eq", "gte", or null.
- Set bathrooms/bathrooms_op only for hard constraints:
  - "at least/minimum/>= X bathrooms" -> {"bathrooms": X, "bathrooms_op": "gte"}
  - "exactly/only X bathrooms" -> {"bathrooms": X, "bathrooms_op": "eq"}
  - soft wording (prefer/ideally/nice to have) -> bathrooms = null, bathrooms_op = null
- available_from should be an ISO date string "YYYY-MM-DD" when possible.
- available_from means user's latest move-in date.
- Do not output available_from_op.
- furnish_type should be one of: "furnished", "unfurnished", "part-furnished", or null.
- let_type examples: "long term", "short term", or null.
- property_type examples: "flat", "apartment", "studio", "house", or null.
- min_tenancy_months is numeric months (e.g., 6, 12) when user specifies tenancy term.
- size constraints:
  - "at least X sqm/sq m/m2" -> min_size_sqm = X
  - "at least X sqft/sq ft/ft2" -> min_size_sqft = X
- If unknown use null or [].
"""

SEMANTIC_EXTRACT_SYSTEM = """You output STRICT JSON only (no markdown, no explanation).
Schema:
{
  "transit_terms": string[],
  "school_terms": string[],
  "general_semantic_phrases": string[]
}
Rules:
- Extract intent-bearing phrases from user request, not single random words.
- Keep named entities as full phrases (e.g., "Seven Mills Primary School", "Heron Quays Station").
- Put transport-specific preferences into transit_terms.
- Put school/education-specific preferences into school_terms.
- Put remaining soft preferences into general_semantic_phrases.
- Do NOT copy hard constraints into semantic terms (budget, bedroom count, property type, strict location filters).
- Do NOT split one entity into many words (bad: "seven", "mills", "primary", "school"; good: "Seven Mills Primary School").
- Avoid generic filler terms like "school" when a concrete entity/phrase exists.
- Return [] when not present.
"""

EXTRACT_ALL_SYSTEM = """You output STRICT JSON only (no markdown, no explanation).
Schema:
{
  "constraints": {
    "max_rent_pcm": number|null,
    "bedrooms": int|null,
    "bedrooms_op": string|null,
    "bathrooms": number|null,
    "bathrooms_op": string|null,
    "available_from": string|null,
    "furnish_type": string|null,
    "let_type": string|null,
    "property_type": string|null,
    "min_tenancy_months": number|null,
    "min_size_sqm": number|null,
    "min_size_sqft": number|null,
    "location_keywords": string[],
    "must_have_keywords": string[],
    "k": int|null
  },
  "semantic_terms": {
    "transit_terms": string[],
    "school_terms": string[],
    "general_semantic_phrases": string[]
  }
}
Rules:
- constraints: extract hard constraints only.
- semantic_terms: extract phrase-level semantic intents.
- Keep named entities as full phrases (e.g., "Seven Mills Primary School", "Heron Quays Station").
- Do NOT put hard constraints into semantic_terms (budget, bedroom count, property type, strict location filters).
- Do NOT split one entity into component words.
- Avoid generic filler terms like "school" when a concrete entity/phrase exists.
- If unknown use null or [].
"""

GROUNDED_EXPLAIN_SYSTEM = """You are a grounded rental explanation engine.
You MUST use only the provided candidate evidence JSON.
Do not invent facts, do not re-rank candidates, do not add external knowledge.

Output format:
1) A short overall summary (2-4 sentences) that reflects user preferences.
2) Then bullet points for each candidate in given rank order:
   - Why it matches preference terms.
   - What is uncertain/missing.
   - Include url.

Rules:
- If evidence is missing, explicitly say unknown.
- Keep concise and factual.
- Preserve the given rank order exactly.
"""

NEAR_WORDS = {
    "near subway","near station","near tube","tube","subway","station","close to station","near metro",
    "near underground","near tube station","close to tube","walk to station"
}

# Let-type rule patterns (high-precision, deterministic).
# Covered forms:
# - short term, short-term, shortterm
# - long term, long-term, longterm
# - short let, short-let, shortlet
# - long let, long-let, longlet
# - st let, st-let, stlet
# - lt let, lt-let, ltlet
# - short stay, short-stay, shortstay
# - long stay, long-stay, longstay
# - temporary let
LET_TYPE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bshort\s*[- ]?\s*term\b", re.I), "short term"),
    (re.compile(r"\blong\s*[- ]?\s*term\b", re.I), "long term"),
    (re.compile(r"\bshort\s*[- ]?\s*let\b", re.I), "short term"),
    (re.compile(r"\blong\s*[- ]?\s*let\b", re.I), "long term"),
    (re.compile(r"\bst\s*[- ]?\s*let\b", re.I), "short term"),
    (re.compile(r"\blt\s*[- ]?\s*let\b", re.I), "long term"),
    (re.compile(r"\bshort\s*[- ]?\s*stay\b", re.I), "short term"),
    (re.compile(r"\blong\s*[- ]?\s*stay\b", re.I), "long term"),
    (re.compile(r"\btemporary\s+let\b", re.I), "short term"),
]

TENANCY_MONTH_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:minimum|min)\s+tenancy\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*months?\b", re.I),
    re.compile(r"\b(?:minimum|min)\s*tenancy\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*months?\b", re.I),
    re.compile(r"\btenancy\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*months?\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*months?\s*(?:minimum|min)\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*months?(?:minimum|min)\b", re.I),
    re.compile(r"\bfor\s+at\s+least\s+(\d+(?:\.\d+)?)\s*months?\b", re.I),
    re.compile(r"\bfor\s*at\s*least\s*(\d+(?:\.\d+)?)\s*months?\b", re.I),
    re.compile(r"\b(?:at\s+least|minimum|min)\s+(\d+(?:\.\d+)?)\s*months?\b", re.I),
    re.compile(r"\b(?:at\s*least|minimum|min)\s*(\d+(?:\.\d+)?)\s*months?\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*mo\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*mos\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*mon(?:th)?s?\b", re.I),
]

# Tenancy "years" forms mapped to months.
# Covered forms:
# - half year, half a year, half yr, half a yr
# - a year, one year, a yr, one yr
# - two years, three years, two yrs, three yrs
# - N years, N yrs (e.g. 1.5 years -> 18)
TENANCY_YEAR_FIXED_RULES: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\bhalf\s*(?:a\s*)?year\b", re.I), 6.0),
    (re.compile(r"\bhalf\s*(?:a\s*)?yr\b", re.I), 6.0),
    (re.compile(r"\b(?:one|a)\s*year\b", re.I), 12.0),
    (re.compile(r"\b(?:one|a)\s*yr\b", re.I), 12.0),
    (re.compile(r"\btwo\s*years?\b", re.I), 24.0),
    (re.compile(r"\btwo\s*yrs?\b", re.I), 24.0),
    (re.compile(r"\bthree\s*years?\b", re.I), 36.0),
    (re.compile(r"\bthree\s*yrs?\b", re.I), 36.0),
]
TENANCY_YEAR_NUMERIC_RULES: List[re.Pattern] = [
    re.compile(r"\b(\d+(?:\.\d+)?)\s*years?\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*yrs?\b", re.I),
]

BEDROOM_EQ_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bexactly\s*(\d+(?:\.\d+)?)\s*[- ]?bed(?:room)?s?\b", re.I),
    re.compile(r"\bonly\s*(\d+(?:\.\d+)?)\s*[- ]?bed(?:room)?s?\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*[- ]?bed(?:room)?s?\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*bd\b", re.I),  # e.g. 1bd -> 1 bedroom
    re.compile(r"\b(\d+(?:\.\d+)?)\s*br\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*b\b", re.I),  # e.g. 1b -> 1 bedroom
]
BATHROOM_EQ_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bexactly\s*(\d+(?:\.\d+)?)\s*[- ]?bath(?:room)?s?\b", re.I),
    re.compile(r"\bonly\s*(\d+(?:\.\d+)?)\s*[- ]?bath(?:room)?s?\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*[- ]?bath(?:room)?s?\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*ba\b", re.I),
]
BED_BATH_COMPACT_PATTERNS: List[re.Pattern] = [
    # Compact shorthand: 1b1b, 2b1b, 2b/1b, 2b-1b, 2bed1bath, 2bd1ba, 2br1ba, 2 bed 1.5 bath
    re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(?:bed(?:room)?s?|bd|br|b)\s*[/,-]?\s*(\d+(?:\.\d+)?)\s*(?:bath(?:room)?s?|ba|b)\b",
        re.I,
    ),
]
NUM_WORDS: Dict[str, str] = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
}
FURNISH_QUERY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Query-side patterns only cover explicit user intents.
    # Do not include "ask agent" here; that's listing-side metadata handling.
    (re.compile(r"\bunfurnish(?:ed)?\b", re.I), "unfurnished"),
    (re.compile(r"\bpart\s*[- ]?\s*furnish(?:ed)?\b", re.I), "part-furnished"),
    (re.compile(r"\bfully\s*[- ]?\s*furnish(?:ed)?\b", re.I), "furnished"),
    (re.compile(r"\bfurnish(?:ed)?\b", re.I), "furnished"),
]
PROPERTY_TYPE_QUERY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bstudio\b", re.I), "studio"),
    (re.compile(r"\bapartment(?:s)?\b", re.I), "apartment"),
    (re.compile(r"\bapt(?:s)?\b", re.I), "apartment"),
    (re.compile(r"\bappartment(?:s)?\b", re.I), "apartment"),
    (re.compile(r"\bground\s*flat\b", re.I), "flat"),
    (re.compile(r"\bflat(?:s)?\b", re.I), "flat"),
    (re.compile(r"\bsemi\s*[- ]?\s*detached\b", re.I), "house"),
    (re.compile(r"\bdetached\b", re.I), "house"),
    (re.compile(r"\btown\s*house\b", re.I), "house"),
    (re.compile(r"\bterraced\b", re.I), "house"),
    (re.compile(r"\bmews\b", re.I), "house"),
    (re.compile(r"\bcottage\b", re.I), "house"),
    (re.compile(r"\bbungalow\b", re.I), "house"),
    (re.compile(r"\bhouse\b", re.I), "house"),
]
PROPERTY_TYPE_HOUSE_LIKE = {
    "terraced",
    "detached",
    "semi detached",
    "semi-detached",
    "town house",
    "mews",
    "cottage",
    "bungalow",
    "end of terrace",
    "link detached house",
    "detached bungalow",
    "country house",
}
PROPERTY_TYPE_FLAT_LIKE = {
    "ground flat",
    "maisonette",
    "duplex",
    "penthouse",
    "serviced apartments",
    "serviced apartment",
    "flat share",
    "block of apartments",
    "block of apartment",
}
PROPERTY_TYPE_SPECIAL_OR_UNKNOWN = {
    "ask agent",
    "house share",
    "house of multiple occupation",
    "retirement property",
    "parking",
    "land",
    "private halls",
    "barn conversion",
    "barn",
    "garages",
    "hotel room",
    "off-plan",
    "park home",
    "retail property (high street)",
    "office",
}
RENT_PCM_PATTERNS: List[re.Pattern] = [
    # If unit is omitted after a budget cue word, default to pcm.
    re.compile(
        r"\b(?:budget|under|below|max(?:imum)?|up\s*to|within|around|about|roughly)\s*£?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(?:pcm|per\s*month|p/?m|pm)?\b",
        re.I,
    ),
    re.compile(r"\b£\s*([0-9][0-9,]*(?:\.\d+)?)\s*(?:pcm|per\s*month|p/?m|pm)\b", re.I),
    re.compile(r"\b([0-9][0-9,]*(?:\.\d+)?)\s*(?:pcm|per\s*month|p/?m|pm)\b", re.I),
]
RENT_PCW_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"\b(?:budget|under|below|max(?:imum)?|up\s*to|within)\s*£?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(?:pcw|per\s*week|p/?w|pw)\b",
        re.I,
    ),
    re.compile(r"\b£\s*([0-9][0-9,]*(?:\.\d+)?)\s*(?:pcw|per\s*week|p/?w|pw)\b", re.I),
    re.compile(r"\b([0-9][0-9,]*(?:\.\d+)?)\s*(?:pcw|per\s*week|p/?w|pw)\b", re.I),
]

DATE_TOKEN_RE = (
    r"(?:\d{4}[/-]\d{1,2}[/-]\d{1,2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s,.-]+\d{1,2}(?:[\s,.-]+\d{2,4})?"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{2,4})?)"
)
AVAILABLE_FROM_PREFIX_PATTERNS: List[re.Pattern] = [
    re.compile(
        rf"\b(?:by|before|no\s+later\s+than|latest(?:\s+move[- ]?in|\s+start)?(?:\s+date)?|"
        rf"available\s*from|starting\s*from|start(?:ing)?\s*from|starting|start\s*date|from)\s*[:=]?\s*({DATE_TOKEN_RE})\b",
        re.I,
    ),
]
AVAILABLE_FROM_BARE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b", re.I),
    re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", re.I),
    re.compile(
        r"\b((?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s,.-]+\d{1,2}(?:[\s,.-]+\d{2,4})?)\b",
        re.I,
    ),
    re.compile(r"\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{2,4})?)\b", re.I),
]

def _parse_user_date_uk_first(value: Any) -> Optional[str]:
    s = _safe_text(value)
    if not s:
        return None
    s = s.strip()

    m_iso = re.fullmatch(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
    if m_iso:
        y, mth, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        try:
            return datetime(y, mth, d).date().isoformat()
        except Exception:
            return None

    m_num = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s)
    if m_num:
        p1, p2, y = int(m_num.group(1)), int(m_num.group(2)), int(m_num.group(3))
        if y < 100:
            y += 2000
        # UK first (DD/MM), but if MM/DD is obvious (2nd part > 12), switch.
        if p2 > 12 and p1 <= 12:
            month, day = p1, p2
        else:
            day, month = p1, p2
        try:
            return datetime(y, month, day).date().isoformat()
        except Exception:
            return None

    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if pd.notna(dt):
        return dt.date().isoformat()
    return None

def _infer_available_from_from_text(text: Any) -> Optional[str]:
    src = _safe_text(text)
    if not src:
        return None
    for pattern in AVAILABLE_FROM_PREFIX_PATTERNS:
        m = pattern.search(src)
        if not m:
            continue
        parsed = _parse_user_date_uk_first(m.group(1))
        if parsed:
            return parsed
    for pattern in AVAILABLE_FROM_BARE_PATTERNS:
        m = pattern.search(src)
        if not m:
            continue
        parsed = _parse_user_date_uk_first(m.group(1))
        if parsed:
            return parsed
    return None

def _infer_numeric_eq_from_patterns(text: Any, patterns: List[re.Pattern]) -> Optional[int]:
    src = _safe_text(text).lower()
    if not src:
        return None
    for w, d in NUM_WORDS.items():
        src = re.sub(rf"\b{w}\b", d, src)
    for pattern in patterns:
        m = pattern.search(src)
        if not m:
            continue
        try:
            v = int(float(m.group(1)))
            if v >= 0:
                return v
        except Exception:
            continue
    return None

def _infer_float_eq_from_patterns(text: Any, patterns: List[re.Pattern]) -> Optional[float]:
    src = _safe_text(text).lower()
    if not src:
        return None
    for w, d in NUM_WORDS.items():
        src = re.sub(rf"\b{w}\b", d, src)
    for pattern in patterns:
        m = pattern.search(src)
        if not m:
            continue
        try:
            v = float(m.group(1))
            if v >= 0:
                return v
        except Exception:
            continue
    return None

def _infer_bed_bath_compact_from_query(text: Any) -> Tuple[Optional[int], Optional[float]]:
    src = _safe_text(text).lower()
    if not src:
        return None, None
    for w, d in NUM_WORDS.items():
        src = re.sub(rf"\b{w}\b", d, src)
    for pattern in BED_BATH_COMPACT_PATTERNS:
        m = pattern.search(src)
        if not m:
            continue
        try:
            bed = int(float(m.group(1)))
            bath = float(m.group(2))
            if bed >= 0 and bath >= 0:
                return bed, bath
        except Exception:
            continue
    return None, None

def _norm_furnish_value(v: Any) -> str:
    s = str(v).strip().lower() if v is not None else ""
    if not s:
        return ""
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    if s in {"ask agent", "ask the agent", "unknown", "not provided", "not known", "n/a", "na"}:
        return "ask agent"
    if "furnished or unfurnished" in s or ("landlord" in s and "flexible" in s):
        return "flexible"
    if "unfurn" in s:
        return "unfurnished"
    if "part" in s and "furnish" in s:
        return "part-furnished"
    if "furnish" in s:
        return "furnished"
    return s

def _infer_furnish_type_from_query(text: Any) -> Optional[str]:
    src = _safe_text(text).lower()
    if not src:
        return None
    # Ambiguous request: do not force hard furnish filter.
    if "furnished or unfurnished" in src or ("landlord" in src and "flexible" in src):
        return None
    for pattern, mapped in FURNISH_QUERY_PATTERNS:
        if pattern.search(src):
            return mapped
    return None

def _norm_property_type_value(v: Any) -> str:
    s = str(v).strip().lower() if v is not None else ""
    if not s:
        return ""
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    if s in {"ask agent", "ask the agent", "unknown", "not provided", "not known", "n/a", "na"}:
        return "ask agent"
    if s == "studio":
        return "studio"
    if s in {"apartment", "apartments"}:
        return "apartment"
    if s in {"flat", "flats"}:
        return "flat"
    if s == "house":
        return "house"
    if s in PROPERTY_TYPE_HOUSE_LIKE:
        return "house"
    if s in PROPERTY_TYPE_FLAT_LIKE:
        return "flat"
    if s in PROPERTY_TYPE_SPECIAL_OR_UNKNOWN:
        return "ask agent"
    return s

def _infer_property_type_from_query(text: Any) -> Optional[str]:
    src = _safe_text(text).lower()
    if not src:
        return None
    for pattern, mapped in PROPERTY_TYPE_QUERY_PATTERNS:
        if pattern.search(src):
            return mapped
    return None

def _infer_max_rent_pcm_from_query(text: Any) -> Optional[float]:
    src = _safe_text(text)
    if not src:
        return None

    def _to_amount(raw: str) -> Optional[float]:
        try:
            val = float(str(raw).replace(",", ""))
            return val if val > 0 else None
        except Exception:
            return None

    for pattern in RENT_PCW_PATTERNS:
        m = pattern.search(src)
        if not m:
            continue
        amt = _to_amount(m.group(1))
        if amt is not None:
            return amt * 52.0 / 12.0

    for pattern in RENT_PCM_PATTERNS:
        m = pattern.search(src)
        if not m:
            continue
        amt = _to_amount(m.group(1))
        if amt is not None:
            return amt

    return None

def _infer_let_type_from_text(text: Any) -> Optional[str]:
    src = _safe_text(text).lower()
    if not src:
        return None
    for pattern, mapped in LET_TYPE_RULES:
        if pattern.search(src):
            return mapped
    return None

def _infer_min_tenancy_months_from_text(text: Any) -> Optional[float]:
    src = _safe_text(text).lower()
    if not src:
        return None

    for pattern in TENANCY_MONTH_PATTERNS:
        m = pattern.search(src)
        if not m:
            continue
        try:
            months = float(m.group(1))
            if months > 0:
                return months
        except Exception:
            continue

    for pattern, months in TENANCY_YEAR_FIXED_RULES:
        if pattern.search(src):
            return months

    for pattern in TENANCY_YEAR_NUMERIC_RULES:
        m = pattern.search(src)
        if not m:
            continue
        try:
            years = float(m.group(1))
            months = years * 12.0
            if months > 0:
                return months
        except Exception:
            continue
    return None

def repair_extracted_constraints(extracted: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    out = dict(extracted or {})
    inferred_from_query = _infer_let_type_from_text(user_text)
    inferred_tenancy_months = _infer_min_tenancy_months_from_text(user_text)
    inferred_available_from = _infer_available_from_from_text(user_text)
    inferred_bed_compact, inferred_bath_compact = _infer_bed_bath_compact_from_query(user_text)
    inferred_bedrooms_eq = _infer_numeric_eq_from_patterns(user_text, BEDROOM_EQ_PATTERNS)
    inferred_bathrooms_eq = _infer_float_eq_from_patterns(user_text, BATHROOM_EQ_PATTERNS)
    inferred_furnish = _infer_furnish_type_from_query(user_text)
    inferred_property_type = _infer_property_type_from_query(user_text)
    inferred_max_rent_pcm = _infer_max_rent_pcm_from_query(user_text)

    # Rescue common slot-mapping error:
    # available_from_op gets "short/long term" text by mistake.
    inferred_from_avail_op = _infer_let_type_from_text(out.get("available_from_op"))
    if inferred_from_avail_op:
        out["available_from_op"] = None
        if not _safe_text(out.get("let_type")):
            out["let_type"] = inferred_from_avail_op

    # Query text has highest confidence for these explicit phrases.
    if inferred_from_query:
        out["let_type"] = inferred_from_query

    # Keep min_tenancy_months only when explicit month/year evidence exists in query text.
    # This prevents model drift like "short term" -> min_tenancy_months = 1.
    if inferred_tenancy_months is not None:
        out["min_tenancy_months"] = inferred_tenancy_months
    else:
        out["min_tenancy_months"] = None

    try:
        if out.get("min_tenancy_months") is not None and float(out["min_tenancy_months"]) <= 0:
            out["min_tenancy_months"] = None
    except Exception:
        out["min_tenancy_months"] = None

    if inferred_available_from:
        out["available_from"] = inferred_available_from
    else:
        out["available_from"] = _parse_user_date_uk_first(out.get("available_from"))

    if inferred_bed_compact is not None:
        out["bedrooms"] = inferred_bed_compact
        out["bedrooms_op"] = "eq"
    elif inferred_bedrooms_eq is not None:
        out["bedrooms"] = inferred_bedrooms_eq
        out["bedrooms_op"] = "eq"

    if inferred_bath_compact is not None:
        out["bathrooms"] = float(inferred_bath_compact)
        out["bathrooms_op"] = "eq"
    elif inferred_bathrooms_eq is not None:
        out["bathrooms"] = float(inferred_bathrooms_eq)
        out["bathrooms_op"] = "eq"

    if inferred_furnish is not None:
        out["furnish_type"] = inferred_furnish

    if inferred_property_type is not None:
        out["property_type"] = inferred_property_type

    if inferred_max_rent_pcm is not None:
        out["max_rent_pcm"] = inferred_max_rent_pcm

    # Deprecated field: always ignore op and use latest move-in semantics.
    out["available_from_op"] = None

    return out

def qwen_chat(messages, temperature=0.0) -> str:
    r = qwen_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        temperature=temperature
    )
    return r.choices[0].message.content.strip()

def _extract_json_obj(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError("No JSON found. Got:\n" + txt)
    return json.loads(m.group(0))

def _normalize_constraint_extract(obj: dict) -> dict:
    obj = obj or {}
    obj.setdefault("k", None)
    obj.setdefault("bedrooms_op", None)
    obj.setdefault("bathrooms", None)
    obj.setdefault("bathrooms_op", None)
    obj.setdefault("available_from", None)
    obj.setdefault("available_from_op", None)
    obj.setdefault("furnish_type", None)
    obj.setdefault("let_type", None)
    obj.setdefault("property_type", None)
    obj.setdefault("min_tenancy_months", None)
    obj.setdefault("min_size_sqm", None)
    obj.setdefault("min_size_sqft", None)
    obj.setdefault("location_keywords", [])
    obj.setdefault("must_have_keywords", [])
    return obj

def _normalize_semantic_extract(obj: dict) -> dict:
    obj = obj or {}
    obj.setdefault("transit_terms", [])
    obj.setdefault("school_terms", [])
    obj.setdefault("general_semantic_phrases", [])

    def _norm_list(v: Any) -> List[str]:
        out = []
        seen = set()
        for x in (v or []):
            s = str(x).strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out

    def _drop_redundant_short_terms(items: List[str]) -> List[str]:
        cleaned = [str(x).strip() for x in (items or []) if str(x).strip()]
        out: List[str] = []
        for s in cleaned:
            sl = s.lower()
            redundant = False
            for t in cleaned:
                tl = t.lower()
                if tl == sl:
                    continue
                # Drop generic/short terms when fully covered by a longer phrase.
                if len(sl) <= len(tl) and re.search(rf"\b{re.escape(sl)}\b", tl):
                    redundant = True
                    break
            if not redundant:
                out.append(s)
        return out

    def _drop_hard_like_general_terms(items: List[str]) -> List[str]:
        out: List[str] = []
        for s in (items or []):
            sl = s.lower()
            # Remove hard-constraint-like fragments from semantic phrases.
            if re.search(r"\bunder\s+\d+\b", sl):
                continue
            if re.search(r"\b\d+\s*bed(room)?s?\b", sl):
                continue
            if re.search(r"\b(flat|apartment|studio|house)\b", sl):
                continue
            out.append(s)
        return out

    school_terms = _drop_redundant_short_terms(_norm_list(obj.get("school_terms")))
    transit_terms = _drop_redundant_short_terms(_norm_list(obj.get("transit_terms")))
    general_terms = _drop_hard_like_general_terms(_norm_list(obj.get("general_semantic_phrases")))
    general_terms = _drop_redundant_short_terms(general_terms)

    return {
        "transit_terms": transit_terms,
        "school_terms": school_terms,
        "general_semantic_phrases": general_terms,
    }

def llm_extract(user_text: str, existing_constraints: Optional[dict]) -> dict:
    prefix = ""
    if existing_constraints:
        prefix = "Existing constraints (JSON):\n" + json.dumps(existing_constraints, ensure_ascii=False) + "\n\n"

    txt = qwen_chat(
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": prefix + "User says:\n" + user_text}
        ],
        temperature=0.0
    )
    obj = _extract_json_obj(txt)
    return _normalize_constraint_extract(obj)

def llm_extract_semantic_terms(user_text: str, existing_constraints: Optional[dict]) -> dict:
    prefix = ""
    if existing_constraints:
        prefix = "Structured constraints (JSON):\n" + json.dumps(existing_constraints, ensure_ascii=False) + "\n\n"

    txt = qwen_chat(
        [
            {"role": "system", "content": SEMANTIC_EXTRACT_SYSTEM},
            {"role": "user", "content": prefix + "User says:\n" + user_text},
        ],
        temperature=0.0,
    )
    obj = _extract_json_obj(txt)
    return _normalize_semantic_extract(obj)

def llm_extract_all_signals(user_text: str, existing_constraints: Optional[dict]) -> Dict[str, Any]:
    prefix = ""
    if existing_constraints:
        prefix = "Existing constraints (JSON):\n" + json.dumps(existing_constraints, ensure_ascii=False) + "\n\n"
    txt = qwen_chat(
        [
            {"role": "system", "content": EXTRACT_ALL_SYSTEM},
            {"role": "user", "content": prefix + "User says:\n" + user_text},
        ],
        temperature=0.0,
    )
    obj = _extract_json_obj(txt)
    constraints = _normalize_constraint_extract(obj.get("constraints") or {})
    semantic_terms = _normalize_semantic_extract(obj.get("semantic_terms") or {})
    return {
        "constraints": constraints,
        "semantic_terms": semantic_terms,
    }

def build_grounded_candidates_payload(
    df: pd.DataFrame,
    c: Dict[str, Any],
    signals: Dict[str, Any],
    user_query: str,
    max_items: int = 5,
) -> Dict[str, Any]:
    rows = []
    if df is not None and len(df) > 0:
        for i, row in df.head(max_items).iterrows():
            r = row.to_dict()
            ev = r.get("evidence") if isinstance(r.get("evidence"), dict) else {}
            pref_ctx = ev.get("preference_context", {}) if isinstance(ev, dict) else {}
            rows.append(
                {
                    "rank": int(i + 1),
                    "title": _safe_text(r.get("title")),
                    "address": _safe_text(r.get("address")),
                    "url": _safe_text(r.get("url")),
                    "price_pcm": _to_float(r.get("price_pcm")),
                    "bedrooms": _to_float(r.get("bedrooms")),
                    "bathrooms": _to_float(r.get("bathrooms")),
                    "scores": {
                        "final_score": _to_float(r.get("final_score")),
                        "transit_score": _to_float(r.get("transit_score")),
                        "school_score": _to_float(r.get("school_score")),
                        "preference_score": _to_float(r.get("preference_score")),
                        "penalty_score": _to_float(r.get("penalty_score")),
                    },
                    "hits": {
                        "transit_hits": _safe_text(r.get("transit_hits")),
                        "school_hits": _safe_text(r.get("school_hits")),
                        "preference_hits": _safe_text(r.get("preference_hits")),
                        "penalty_reasons": _safe_text(r.get("penalty_reasons")),
                    },
                    "preference_context": pref_ctx,
                    "fields": ev.get("fields", {}) if isinstance(ev, dict) else {},
                }
            )
    return {
        "user_query": user_query,
        "hard_constraints": signals.get("hard_constraints", {}) if isinstance(signals, dict) else {},
        "semantic_preferences": {
            "location_intent": signals.get("location_intent", []) if isinstance(signals, dict) else [],
            "topic_preferences": signals.get("topic_preferences", {}) if isinstance(signals, dict) else {},
            "general_semantic": signals.get("general_semantic", []) if isinstance(signals, dict) else [],
        },
        "candidates": rows,
        "k": int(c.get("k", DEFAULT_K) or DEFAULT_K),
    }

def llm_grounded_explain(
    user_query: str,
    c: Dict[str, Any],
    signals: Dict[str, Any],
    df: pd.DataFrame,
) -> Tuple[str, Dict[str, Any], str]:
    payload = build_grounded_candidates_payload(
        df=df,
        c=c or {},
        signals=signals or {},
        user_query=user_query,
        max_items=min(8, int(c.get("k", DEFAULT_K) or DEFAULT_K)),
    )
    txt = qwen_chat(
        [
            {"role": "system", "content": GROUNDED_EXPLAIN_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Generate grounded explanation from this JSON only:\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ],
        temperature=0.1,
    )
    out = txt.strip()
    return out, payload, txt

def format_grounded_evidence(df: pd.DataFrame, max_items: int = 8) -> str:
    if df is None or len(df) == 0:
        return ""
    lines: List[str] = []
    for i, row in df.head(max_items).iterrows():
        r = row.to_dict()
        ev = r.get("evidence") if isinstance(r.get("evidence"), dict) else {}
        pref = ev.get("preference_context", {}) if isinstance(ev, dict) else {}
        merged: List[Dict[str, Any]] = []
        for key in ("transit_evidence", "school_evidence", "preference_evidence"):
            vals = pref.get(key, [])
            if isinstance(vals, list):
                for x in vals:
                    if isinstance(x, dict):
                        merged.append(x)
        if not merged:
            lines.append(f"- #{i+1}: no preference evidence")
            continue
        lines.append(f"- #{i+1}:")
        for x in merged[:4]:
            intent = _safe_text(x.get("intent"))
            field = _safe_text(x.get("field"))
            text = _safe_text(x.get("text"))
            sim = _to_float(x.get("sim"))
            sim_txt = f"{sim:.3f}" if sim is not None else "na"
            # Show explicit pairwise comparison: user preference phrase vs listing evidence text.
            lines.append(f"  compare: '{intent}' vs '{text[:140]}'")
            lines.append(f"  score: sim={sim_txt} | field={field}")
    return "\n".join(lines)
def normalize_budget_to_pcm(c: dict) -> dict:
    """
    Normalize budget constraints to pcm.
    Supports:
      - max_rent_pcm
      - max_rent_pcw
    Priority:
      - If both provided, pcm wins.
    """
    if c is None:
        return c

    # if user gave pcw, convert to pcm
    if c.get("max_rent_pcm") is None and c.get("max_rent_pcw") is not None:
        try:
            pcw = float(c["max_rent_pcw"])
            c["max_rent_pcm"] = pcw * 52.0 / 12.0
        except:
            pass

    return c
def normalize_constraints(c: dict) -> dict:
    # normalize bedrooms hard-constraint operator
    bed_op = c.get("bedrooms_op")
    if bed_op is not None:
        bed_op = str(bed_op).strip().lower()
        if bed_op in ("==", "=", "exact", "exactly", "eq"):
            bed_op = "eq"
        elif bed_op in (">=", "min", "minimum", "at_least", "at least", "gte"):
            bed_op = "gte"
        else:
            bed_op = None
    c["bedrooms_op"] = bed_op

    if c.get("bedrooms") is not None:
        try:
            c["bedrooms"] = int(float(c["bedrooms"]))
        except:
            c["bedrooms"] = None

    # default to strict equality when bedrooms is set but operator is absent
    if c.get("bedrooms") is not None and c.get("bedrooms_op") is None:
        c["bedrooms_op"] = "eq"

    # normalize bathrooms hard-constraint operator
    op = c.get("bathrooms_op")
    if op is not None:
        op = str(op).strip().lower()
        if op in ("==", "=", "exact", "exactly", "eq"):
            op = "eq"
        elif op in (">=", "min", "minimum", "at_least", "at least", "gte"):
            op = "gte"
        else:
            op = None
    c["bathrooms_op"] = op

    if c.get("bathrooms") is not None:
        try:
            c["bathrooms"] = float(c["bathrooms"])
        except:
            c["bathrooms"] = None

    # default to strict equality when bathrooms is set but operator is absent
    if c.get("bathrooms") is not None and c.get("bathrooms_op") is None:
        c["bathrooms_op"] = "eq"

    if c.get("available_from") is not None:
        c["available_from"] = _parse_user_date_uk_first(c.get("available_from"))
    c["available_from_op"] = None

    def _norm_cat_text(v: Any) -> Optional[str]:
        s = _safe_text(v).lower()
        if not s:
            return None
        s = s.replace("_", " ").replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s or None

    furn = _norm_furnish_value(c.get("furnish_type"))
    if furn not in {"furnished", "unfurnished", "part-furnished"}:
        furn = None
    c["furnish_type"] = furn
    c["let_type"] = _norm_cat_text(c.get("let_type"))
    ptype = _norm_property_type_value(c.get("property_type"))
    if ptype not in {"studio", "apartment", "flat", "house"}:
        ptype = None
    c["property_type"] = ptype

    if c.get("min_tenancy_months") is not None:
        try:
            c["min_tenancy_months"] = float(c.get("min_tenancy_months"))
        except Exception:
            c["min_tenancy_months"] = None

    if c.get("min_size_sqm") is not None:
        try:
            c["min_size_sqm"] = float(c.get("min_size_sqm"))
        except Exception:
            c["min_size_sqm"] = None
    if c.get("min_size_sqft") is not None:
        try:
            c["min_size_sqft"] = float(c.get("min_size_sqft"))
        except Exception:
            c["min_size_sqft"] = None
    # merge size constraints into a single canonical hard filter in sqm
    if c.get("min_size_sqm") is None and c.get("min_size_sqft") is not None:
        c["min_size_sqm"] = float(c["min_size_sqft"]) * 0.092903
    c.pop("min_size_sqft", None)

    # Disabled on purpose:
    # Do not auto-move location_keywords entries (e.g. "near tube/station")
    # into must_have_keywords. Keep LLM-structured fields as-is.
    #
    # locs = []
    # must = set([str(x).strip() for x in (c.get("must_have_keywords") or []) if str(x).strip()])
    # for x in (c.get("location_keywords") or []):
    #     s = str(x).strip()
    #     if not s:
    #         continue
    #     if s.lower() in NEAR_WORDS:
    #         must.add(s)
    #     else:
    #         locs.append(s)
    # c["location_keywords"] = locs
    # c["must_have_keywords"] = list(must)
    return c

def merge_constraints(old: Optional[dict], new: dict) -> dict:
    if old is None:
        old = {}
    out = dict(old)

    # scalar fields: new overrides if not null
    for key in [
        "max_rent_pcm",
        "bedrooms",
        "bedrooms_op",
        "bathrooms",
        "bathrooms_op",
        "available_from",
        "furnish_type",
        "let_type",
        "property_type",
        "min_tenancy_months",
        "min_size_sqm",
        "min_size_sqft",
        "k",
    ]:
        if new.get(key) is not None:
            out[key] = new.get(key)

    def merge_list(a, b):
        a = a or []
        b = b or []
        seen = set()
        res = []
        for x in a + b:
            s = str(x).strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            res.append(s)
        return res

    out["location_keywords"] = merge_list(old.get("location_keywords"), new.get("location_keywords"))
    out["must_have_keywords"] = merge_list(old.get("must_have_keywords"), new.get("must_have_keywords"))

    # default k
    if out.get("k") is None:
        out["k"] = DEFAULT_K

    return normalize_constraints(out)

def _norm_scalar_for_diff(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if np.isnan(v):
            return None
        return float(v)
    if isinstance(v, (int, bool)):
        return v
    s = str(v).strip()
    return s if s != "" else None


def summarize_constraint_changes(old_c: Optional[dict], new_c: dict) -> str:
    old_c = old_c or {}
    new_c = new_c or {}
    keys = [
        "max_rent_pcm",
        "bedrooms", "bedrooms_op",
        "bathrooms", "bathrooms_op",
        "available_from",
        "furnish_type", "let_type", "property_type",
        "min_tenancy_months", "min_size_sqm",
        "k",
    ]
    changes = []
    for k in keys:
        old_v = _norm_scalar_for_diff(old_c.get(k))
        new_v = _norm_scalar_for_diff(new_c.get(k))
        if old_v == new_v:
            continue
        if old_v is None and new_v is not None:
            changes.append(f"added {k}={new_v}")
        elif old_v is not None and new_v is None:
            changes.append(f"removed {k}")
        else:
            changes.append(f"updated {k}: {old_v} -> {new_v}")

    def _norm_list(x):
        out = []
        for i in (x or []):
            s = str(i).strip()
            if s:
                out.append(s)
        return out

    for k in ["location_keywords", "must_have_keywords"]:
        old_list = _norm_list(old_c.get(k))
        new_list = _norm_list(new_c.get(k))
        old_set = set([x.lower() for x in old_list])
        new_set = set([x.lower() for x in new_list])
        added = [x for x in new_list if x.lower() not in old_set]
        removed = [x for x in old_list if x.lower() not in new_set]
        if added:
            changes.append(f"added {k}: {added}")
        if removed:
            changes.append(f"removed {k}: {removed}")

    return "; ".join(changes) if changes else "no constraint changes"


def compact_constraints_view(c: Optional[dict]) -> dict:
    c = c or {}
    out = {}
    keep_keys = [
        "max_rent_pcm",
        "bedrooms", "bedrooms_op",
        "bathrooms", "bathrooms_op",
        "available_from",
        "furnish_type", "let_type", "property_type",
        "min_tenancy_months", "min_size_sqm",
        "location_keywords", "must_have_keywords",
        "k",
    ]
    for k in keep_keys:
        v = c.get(k)
        if v is None:
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        out[k] = v
    return out

def constraints_to_query_hint(c: dict) -> str:
    parts = []
    if c.get("bedrooms") is not None:
        bed_op = c.get("bedrooms_op", "eq")
        if bed_op == "gte":
            parts.append(f"at least {int(c['bedrooms'])} bedroom")
        else:
            parts.append(f"{int(c['bedrooms'])} bedroom")
    if c.get("bathrooms") is not None:
        op = c.get("bathrooms_op", "eq")
        if op == "gte":
            parts.append(f"at least {float(c['bathrooms']):g} bathroom")
        else:
            parts.append(f"{float(c['bathrooms']):g} bathroom")
    if c.get("max_rent_pcm") is not None:
        parts.append(f"budget {float(c['max_rent_pcm'])} pcm")
    if c.get("available_from") is not None:
        parts.append(f"available by {c['available_from']}")
    if c.get("furnish_type"):
        parts.append(str(c.get("furnish_type")))
    if c.get("let_type"):
        parts.append(str(c.get("let_type")))
    if c.get("property_type"):
        parts.append(str(c.get("property_type")))
    if c.get("min_tenancy_months") is not None:
        parts.append(f"min tenancy {float(c['min_tenancy_months']):g} months")
    if c.get("min_size_sqm") is not None:
        parts.append(f"at least {float(c['min_size_sqm']):g} sqm")
    for x in (c.get("location_keywords") or [])[:5]:
        parts.append(x)
    for x in (c.get("must_have_keywords") or [])[:5]:
        parts.append(x)
    return " | ".join(parts)


def infer_soft_keywords_from_query(user_text: str) -> List[str]:
    if not user_text:
        return []
    stop_words = {
        "the", "and", "for", "with", "near", "from", "rent", "flat", "apartment",
        "bed", "bath", "budget", "pcm", "in", "to", "of", "a", "an", "at",
        "need", "want", "looking", "around", "about", "please", "london",
    }
    toks = re.findall(r"[A-Za-z0-9]{3,}", user_text.lower())
    out = []
    seen = set()
    for t in toks:
        if t in stop_words:
            continue
        if t.isdigit():
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# project root = this file's directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_INDEX_DIR = os.path.join(ROOT_DIR, "artifacts", "hnsw", "index")
LEGACY_INDEX_DIR = os.path.join(ROOT_DIR, "data", "HNSW", "index")
DEFAULT_INDEX_DIR = "/workspace/rent-chatbot/artifacts/hnsw/zone1_global"

OUT_DIR = os.environ.get("RENT_INDEX_DIR", DEFAULT_INDEX_DIR)

LIST_INDEX_PATH = os.path.join(OUT_DIR, "listings_hnsw.faiss")
LIST_META_PATH  = os.path.join(OUT_DIR, "listings_meta.parquet")
QDRANT_LOCAL_PATH = os.environ.get(
    "RENT_QDRANT_PATH",
    os.path.join(ROOT_DIR, "artifacts", "qdrant_local"),
)
QDRANT_COLLECTION = os.environ.get("RENT_QDRANT_COLLECTION", "rent_listings")
STAGEA_BACKEND = os.environ.get("RENT_STAGEA_BACKEND", "faiss").strip().lower()
if STAGEA_BACKEND not in {"faiss", "qdrant"}:
    STAGEA_BACKEND = "faiss"
QDRANT_ENABLE_PREFILTER = os.environ.get("RENT_QDRANT_ENABLE_PREFILTER", "1") != "0"


EMBED_MODEL = os.environ.get("RENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH = int(os.environ.get("RENT_EMBED_BATCH", "256"))

DEFAULT_K = int(os.environ.get("RENT_K", "5"))
DEFAULT_RECALL = int(os.environ.get("RENT_RECALL", "200"))  # pull more then slice to k
UNKNOWN_PENALTY_WEIGHTS = {
    "price": 0.35,
    "bedrooms": 0.30,
    "bathrooms": 0.20,
    "available_from": 0.15,
}
UNKNOWN_PENALTY_CAP = 0.60
SOFT_MUST_HIT_BONUS = 0.04
SOFT_MUST_MISS_PENALTY = 0.06
SOFT_MUST_MAX_BONUS = 0.20
SOFT_MUST_MAX_PENALTY = 0.30
FURNISH_ASK_AGENT_PENALTY = 0.06
FAISS_SCORE_WEIGHT = float(os.environ.get("RENT_FAISS_WEIGHT", "0.35"))
PREFERENCE_SCORE_WEIGHT = float(os.environ.get("RENT_PREF_WEIGHT", "1.0"))
UNKNOWN_PENALTY_WEIGHT = float(os.environ.get("RENT_UNKNOWN_PENALTY_WEIGHT", "1.0"))
SOFT_PENALTY_WEIGHT = float(os.environ.get("RENT_SOFT_PENALTY_WEIGHT", "1.0"))
RANKING_LOG_PATH = os.environ.get(
    "RENT_RANKING_LOG_PATH",
    os.path.join(ROOT_DIR, "artifacts", "debug", "ranking_log.jsonl"),
)
STRUCTURED_POLICY = os.environ.get("RENT_STRUCTURED_POLICY", "RULE_FIRST").strip().upper()
if STRUCTURED_POLICY not in {"RULE_FIRST", "HYBRID", "LLM_FIRST"}:
    STRUCTURED_POLICY = "RULE_FIRST"
STRUCTURED_CONFLICT_LOG_PATH = os.environ.get(
    "RENT_STRUCTURED_CONFLICT_LOG_PATH",
    os.path.join(ROOT_DIR, "artifacts", "debug", "structured_conflicts.jsonl"),
)
STRUCTURED_TRAINING_LOG_PATH = os.environ.get(
    "RENT_STRUCTURED_TRAINING_LOG_PATH",
    os.path.join(ROOT_DIR, "artifacts", "debug", "structured_training_samples.jsonl"),
)
ENABLE_STRUCTURED_CONFLICT_LOG = os.environ.get("RENT_STRUCTURED_CONFLICT_LOG", "1") != "0"
ENABLE_STRUCTURED_TRAINING_LOG = os.environ.get("RENT_STRUCTURED_TRAINING_LOG", "1") != "0"
SEMANTIC_TOP_K = int(os.environ.get("RENT_SEMANTIC_TOPK", "4"))
SEMANTIC_FIELD_WEIGHTS = {
    "schools": 1.00,
    "stations": 1.00,
    "features": 0.80,
    "description": 0.60,
}
INTENT_HIT_THRESHOLD = 0.45
INTENT_EVIDENCE_TOP_N = 2

TRANSIT_KEYWORDS = [
    "tube", "station", "underground", "metro", "rail", "dlr", "overground",
    "line", "jubilee", "elizabeth", "central", "northern", "victoria",
    "piccadilly", "district", "circle", "bakerloo", "waterloo", "city",
    "crossrail", "commute",
]
SCHOOL_KEYWORDS = [
    "school", "schools", "university", "college", "campus", "ucl",
    "imperial", "kcl", "kings", "lse", "qmul", "queen mary", "student",
]

ROUTE_WEIGHT_TEMPLATES = {
    "HARD_DOMINANT": {
        "w_faiss": 0.15,
        "w_pref": 0.80,
        "w_unknown": 1.20,
        "w_soft": 0.80,
    },
    "HYBRID": {
        "w_faiss": 0.25,
        "w_pref": 1.00,
        "w_unknown": 1.00,
        "w_soft": 1.00,
    },
    "SEMANTIC_DOMINANT": {
        "w_faiss": 0.40,
        "w_pref": 1.10,
        "w_unknown": 0.80,
        "w_soft": 1.10,
    },
}

def decide_route(c: dict) -> str:
    if c is None:
        return "HYBRID"

    hard_fields = [
        "max_rent_pcm",
        "bedrooms",
        "bathrooms",
        "available_from",
        "furnish_type",
        "let_type",
        "property_type",
        "min_tenancy_months",
        "min_size_sqm",
    ]
    hard_count = sum(1 for k in hard_fields if c.get(k) is not None)

    semantic_count = 0
    semantic_count += len([x for x in (c.get("must_have_keywords") or []) if str(x).strip()])
    semantic_count += len([x for x in (c.get("location_keywords") or []) if str(x).strip()])

    if hard_count >= 3 and semantic_count <= 1:
        return "HARD_DOMINANT"
    if hard_count <= 1 and semantic_count >= 2:
        return "SEMANTIC_DOMINANT"
    return "HYBRID"


# ----------------------------
# Load resources
# ----------------------------
def load_index_and_meta():
    if not os.path.exists(LIST_INDEX_PATH):
        raise FileNotFoundError(f"Missing FAISS index: {LIST_INDEX_PATH}")
    if not os.path.exists(LIST_META_PATH):
        raise FileNotFoundError(f"Missing meta parquet: {LIST_META_PATH}")

    index = faiss.read_index(LIST_INDEX_PATH)
    meta = pd.read_parquet(LIST_META_PATH).copy()

    # 关键：保证 0..N-1 行号对齐（用于 iloc）
    meta = meta.reset_index(drop=True)

    print(f"[boot] faiss ntotal={index.ntotal}, meta rows={len(meta)}")
    return index, meta

def load_qdrant_client() -> QdrantClient:
    if QdrantClient is None or models is None:
        raise ImportError("qdrant-client is not installed. Please run: pip install qdrant-client")
    client = QdrantClient(path=QDRANT_LOCAL_PATH)
    if not client.collection_exists(QDRANT_COLLECTION):
        raise FileNotFoundError(
            f"Missing Qdrant collection: {QDRANT_COLLECTION} (path={QDRANT_LOCAL_PATH})"
        )
    info = client.get_collection(QDRANT_COLLECTION)
    print(f"[boot] qdrant collection={QDRANT_COLLECTION}, points={info.points_count}")
    return client

def load_stage_a_resources():
    if STAGEA_BACKEND == "qdrant":
        return load_qdrant_client(), None
    return load_index_and_meta()

def embed_query(embedder: SentenceTransformer, q: str) -> np.ndarray:
    x = embedder.encode(
        [q],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,  # match your builder, then normalize with faiss
    ).astype("float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms
    return x


# ----------------------------
# Retrieval
# ----------------------------
def faiss_search(index, embedder, meta: pd.DataFrame, query: str, recall: int) -> pd.DataFrame:
    qx = embed_query(embedder, query)
    scores, ids = index.search(qx, recall)

    ids = ids[0].tolist()
    scores = scores[0].tolist()

    rows = []
    n = len(meta)

    for lid, sc in zip(ids, scores):
        if lid is None:
            continue
        lid = int(lid)
        if lid < 0 or lid >= n:
            continue

        r = meta.iloc[lid].to_dict()
        r["faiss_score"] = float(sc)
        r["_faiss_id"] = lid
        rows.append(r)

    if not rows:
        return meta.head(0).copy()

    # IMPORTANT:
    # Do not truncate to k before hard filters. We need a larger recall pool first.
    return pd.DataFrame(rows).reset_index(drop=True)


def qdrant_search(
    client: QdrantClient,
    embedder,
    query: str,
    recall: int,
    c: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if models is None:
        raise ImportError("qdrant-client models are unavailable. Please run: pip install qdrant-client")

    def _norm_cat(v: Any) -> str:
        s = _safe_text(v).lower()
        if not s:
            return ""
        s = s.replace("_", " ").replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _build_qdrant_filter(c: Optional[Dict[str, Any]]) -> Optional["models.Filter"]:
        c = c or {}
        must: List[Any] = []

        rent_req = c.get("max_rent_pcm")
        if rent_req is not None:
            try:
                must.append(
                    models.FieldCondition(
                        key="price_pcm_num",
                        range=models.Range(lte=float(rent_req)),
                    )
                )
            except Exception:
                pass

        bed_req = c.get("bedrooms")
        if bed_req is not None:
            op = str(c.get("bedrooms_op") or "eq").lower()
            try:
                req = float(bed_req)
                if op == "gte":
                    must.append(
                        models.FieldCondition(
                            key="bedrooms_num",
                            range=models.Range(gte=req),
                        )
                    )
                else:
                    must.append(
                        models.FieldCondition(
                            key="bedrooms_num",
                            range=models.Range(gte=req, lte=req),
                        )
                    )
            except Exception:
                pass

        bath_req = c.get("bathrooms")
        if bath_req is not None:
            op = str(c.get("bathrooms_op") or "eq").lower()
            try:
                req = float(bath_req)
                if op == "gte":
                    must.append(
                        models.FieldCondition(
                            key="bathrooms_num",
                            range=models.Range(gte=req),
                        )
                    )
                else:
                    must.append(
                        models.FieldCondition(
                            key="bathrooms_num",
                            range=models.Range(gte=req, lte=req),
                        )
                    )
            except Exception:
                pass

        let_req = _norm_cat(c.get("let_type"))
        if let_req:
            must.append(
                models.FieldCondition(
                    key="let_type_norm",
                    match=models.MatchValue(value=let_req),
                )
            )

        prop_req = _norm_property_type_value(c.get("property_type"))
        if prop_req:
            must.append(
                models.FieldCondition(
                    key="property_type_norm",
                    match=models.MatchValue(value=prop_req),
                )
            )

        furn_req = _norm_furnish_value(c.get("furnish_type"))
        if furn_req and furn_req not in {"ask agent", "flexible"}:
            must.append(
                models.FieldCondition(
                    key="furnish_type_norm",
                    match=models.MatchValue(value=furn_req),
                )
            )

        loc_values: List[str] = []
        for term in (c.get("location_keywords") or []):
            raw = _safe_text(term).lower()
            if not raw:
                continue
            raw = re.sub(r"\s+", " ", raw).strip()
            slug = re.sub(r"[^a-z0-9]+", "_", raw)
            slug = re.sub(r"_+", "_", slug).strip("_")
            if raw:
                loc_values.append(raw)
                if " " in raw:
                    loc_values.append(raw.replace(" ", ""))
            if slug:
                loc_values.append(slug)
                loc_values.append(f"{slug}_london")
                if not slug.endswith("_station"):
                    loc_values.append(f"{slug}_station")

        loc_values = list(dict.fromkeys([x for x in loc_values if x]))
        if loc_values:
            must.append(
                models.FieldCondition(
                    key="location_tokens",
                    match=models.MatchAny(any=loc_values),
                )
            )

        if not must:
            return None
        return models.Filter(must=must)

    qx = embed_query(embedder, query)[0].tolist()
    qfilter = _build_qdrant_filter(c) if QDRANT_ENABLE_PREFILTER else None
    hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=qx,
        query_filter=qfilter,
        limit=recall,
        with_payload=True,
        with_vectors=False,
    )

    rows = []
    for h in hits:
        payload = dict(h.payload or {})
        score = float(h.score)
        # Keep column compatibility for downstream ranking/logging code.
        payload["faiss_score"] = score
        payload["qdrant_score"] = score
        payload["_faiss_id"] = None
        payload["_qdrant_id"] = h.id
        rows.append(payload)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def stage_a_search(
    index_or_client,
    embedder,
    meta: Optional[pd.DataFrame],
    query: str,
    recall: int,
    c: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if STAGEA_BACKEND == "qdrant":
        return qdrant_search(index_or_client, embedder, query=query, recall=recall, c=c)
    if meta is None:
        raise ValueError("meta is required when STAGEA_BACKEND=faiss")
    return faiss_search(index_or_client, embedder, meta, query=query, recall=recall)


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in ("", "nan", "none", "ask agent"):
        return ""
    return s


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        s = re.sub(r"[^\d\.\-]", "", str(v))
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def parse_jsonish_items(v: Any) -> List[str]:
    s = _safe_text(v)
    if not s:
        return []
    if isinstance(v, list):
        out = []
        for it in v:
            t = _safe_text(it)
            if t:
                out.append(t)
        return out
    if s.startswith("[") or s.startswith("{"):
        try:
            parsed = json.loads(s)
            out: List[str] = []
            if isinstance(parsed, list):
                for it in parsed:
                    if isinstance(it, dict):
                        name = _safe_text(it.get("name"))
                        miles = _safe_text(it.get("miles"))
                        if name and miles:
                            out.append(f"{name} ({miles} miles)")
                        elif name:
                            out.append(name)
                    else:
                        t = _safe_text(it)
                        if t:
                            out.append(t)
                return out
            if isinstance(parsed, dict):
                name = _safe_text(parsed.get("name"))
                miles = _safe_text(parsed.get("miles"))
                if name and miles:
                    return [f"{name} ({miles} miles)"]
                if name:
                    return [name]
        except Exception:
            pass
    if "|" in s:
        return [_safe_text(x) for x in s.split("|") if _safe_text(x)]
    return [s]


def tokenize_query(s: str) -> List[str]:
    if not s:
        return []
    return re.findall(r"[a-z0-9]{2,}", s.lower())


def split_query_signals(
    user_in: str,
    c: Dict[str, Any],
    precomputed_semantic_terms: Optional[Dict[str, Any]] = None,
    semantic_parse_source: str = "llm",
) -> Dict[str, Any]:
    c = c or {}
    must_keywords = [str(x).strip() for x in (c.get("must_have_keywords") or []) if str(x).strip()]
    location_intent = [str(x).strip() for x in (c.get("location_keywords") or []) if str(x).strip()]

    model_terms = precomputed_semantic_terms or {
        "transit_terms": [],
        "school_terms": [],
        "general_semantic_phrases": [],
    }
    if precomputed_semantic_terms is None:
        semantic_parse_source = "llm"
        try:
            model_terms = llm_extract_semantic_terms(user_in, c)
        except Exception:
            # fall back to rule-based token extraction when LLM parsing fails
            semantic_parse_source = "fallback_rules"

    transit_terms: List[str] = []
    school_terms: List[str] = []
    for t in model_terms.get("transit_terms", []):
        s = str(t).strip().lower()
        if s:
            transit_terms.append(s)
    for t in model_terms.get("school_terms", []):
        s = str(t).strip().lower()
        if s:
            school_terms.append(s)

    general_semantic = [str(x).strip().lower() for x in model_terms.get("general_semantic_phrases", []) if str(x).strip()]
    keyword_fallback_used = {
        "transit_terms": False,
        "school_terms": False,
        "general_semantic": False,
    }
    # Keyword fallback is intentionally disabled for now.
    # Semantic intent comes from LLM structured output only.
    general_semantic = list(dict.fromkeys(general_semantic))
    transit_terms = list(dict.fromkeys([x for x in transit_terms if str(x).strip()]))
    school_terms = list(dict.fromkeys([x for x in school_terms if str(x).strip()]))

    return {
        "hard_constraints": {
            "max_rent_pcm": c.get("max_rent_pcm"),
            "bedrooms": c.get("bedrooms"),
            "bedrooms_op": c.get("bedrooms_op"),
            "bathrooms": c.get("bathrooms"),
            "bathrooms_op": c.get("bathrooms_op"),
            "available_from": c.get("available_from"),
            "furnish_type": c.get("furnish_type"),
            "let_type": c.get("let_type"),
            "property_type": c.get("property_type"),
            "min_tenancy_months": c.get("min_tenancy_months"),
            "min_size_sqm": c.get("min_size_sqm"),
        },
        "location_intent": location_intent,
        "topic_preferences": {
            "transit_terms": transit_terms,
            "school_terms": school_terms,
        },
        "general_semantic": general_semantic,
        "semantic_debug": {
            "parse_source": semantic_parse_source,
            "model_terms": model_terms,
            "keyword_fallback_used": keyword_fallback_used,
            "keyword_transit_candidates": [],
            "keyword_school_candidates": [],
            "fallback_tokens": [],
            "final_general_semantic": general_semantic,
        },
    }


def build_stage_a_query(signals: Dict[str, Any], user_in: str) -> str:
    parts: List[str] = []
    hard = signals.get("hard_constraints", {}) or {}

    # Keep hard hints in Stage A retrieval to improve recall coverage.
    if hard.get("bedrooms") is not None:
        op = str(hard.get("bedrooms_op") or "eq").lower()
        if op == "gte":
            parts.append(f"at least {int(float(hard.get('bedrooms')))} bedroom")
        else:
            parts.append(f"{int(float(hard.get('bedrooms')))} bedroom")
    if hard.get("bathrooms") is not None:
        op = str(hard.get("bathrooms_op") or "eq").lower()
        if op == "gte":
            parts.append(f"at least {float(hard.get('bathrooms')):g} bathroom")
        else:
            parts.append(f"{float(hard.get('bathrooms')):g} bathroom")
    if hard.get("max_rent_pcm") is not None:
        parts.append(f"under {int(float(hard.get('max_rent_pcm')))} pcm")
    if hard.get("furnish_type"):
        parts.append(str(hard.get("furnish_type")))
    if hard.get("let_type"):
        parts.append(str(hard.get("let_type")))
    if hard.get("property_type"):
        parts.append(str(hard.get("property_type")))
    if hard.get("min_tenancy_months") is not None:
        parts.append(f"{float(hard.get('min_tenancy_months')):g} months tenancy")
    if hard.get("min_size_sqm") is not None:
        parts.append(f"at least {float(hard.get('min_size_sqm')):g} sqm")

    parts.extend([x for x in signals.get("location_intent", []) if str(x).strip()])
    parts.extend([x for x in signals.get("general_semantic", []) if str(x).strip()])
    if not parts:
        return user_in
    return " | ".join(parts[:20])


def candidate_snapshot(r: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": _safe_text(r.get("url")),
        "title": _safe_text(r.get("title")),
        "address": _safe_text(r.get("address")),
        "price_pcm": _to_float(r.get("price_pcm")),
        "bedrooms": _to_float(r.get("bedrooms")),
        "bathrooms": _to_float(r.get("bathrooms")),
        "available_from": _safe_text(r.get("available_from")),
        "faiss_score": _to_float(r.get("faiss_score")),
        "_faiss_id": int(r.get("_faiss_id")) if r.get("_faiss_id") is not None else None,
    }


def apply_hard_filters_with_audit(df: pd.DataFrame, c: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    c = c or {}
    if df is None or len(df) == 0:
        return df, []

    keep_indices: List[int] = []
    audits: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        r = row.to_dict()
        reasons: List[str] = []
        checks: Dict[str, Any] = {}

        def _norm_cat_text(v: Any) -> str:
            s = _safe_text(v).lower()
            if not s:
                return ""
            s = s.replace("_", " ").replace("-", " ")
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _norm_furnish(v: Any) -> str:
            return _norm_furnish_value(v)

        def _parse_months(v: Any) -> Optional[float]:
            s = _safe_text(v).lower()
            if not s:
                return None
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            if not m:
                return None
            try:
                return float(m.group(1))
            except Exception:
                return None

        bed_req = c.get("bedrooms")
        if bed_req is not None:
            bed_val = _to_float(r.get("bedrooms"))
            op = str(c.get("bedrooms_op") or "eq").lower()
            checks["bedrooms"] = {"actual": bed_val, "required": int(bed_req), "op": op}
            if bed_val is not None:
                if op == "gte" and bed_val < float(bed_req):
                    reasons.append(f"bedrooms {bed_val:g} < {float(bed_req):g}")
                if op != "gte" and int(round(bed_val)) != int(bed_req):
                    reasons.append(f"bedrooms {bed_val:g} != {int(bed_req)}")

        bath_req = c.get("bathrooms")
        if bath_req is not None:
            bath_val = _to_float(r.get("bathrooms"))
            op = str(c.get("bathrooms_op") or "eq").lower()
            checks["bathrooms"] = {"actual": bath_val, "required": float(bath_req), "op": op}
            if bath_val is not None:
                if op == "gte" and bath_val < float(bath_req):
                    reasons.append(f"bathrooms {bath_val:g} < {float(bath_req):g}")
                if op != "gte" and bath_val != float(bath_req):
                    reasons.append(f"bathrooms {bath_val:g} != {float(bath_req):g}")

        rent_req = c.get("max_rent_pcm")
        if rent_req is not None:
            rent_val = _to_float(r.get("price_pcm"))
            checks["max_rent_pcm"] = {"actual": rent_val, "required": float(rent_req), "op": "lte"}
            if rent_val is not None and rent_val > float(rent_req):
                reasons.append(f"price {rent_val:g} > {float(rent_req):g}")

        avail_req = c.get("available_from")
        if avail_req is not None:
            listing_dt = pd.to_datetime(r.get("available_from"), errors="coerce")
            req_dt = pd.to_datetime(avail_req, errors="coerce")
            checks["available_from"] = {
                "actual": None if pd.isna(listing_dt) else listing_dt.date().isoformat(),
                "required": None if pd.isna(req_dt) else req_dt.date().isoformat(),
                "op": "lte",
            }
            if pd.notna(listing_dt) and pd.notna(req_dt) and listing_dt > req_dt:
                reasons.append(
                    f"available_from {listing_dt.date().isoformat()} > {req_dt.date().isoformat()}"
                )

        furnish_req = _norm_furnish(c.get("furnish_type"))
        if furnish_req:
            furnish_val = _norm_furnish(r.get("furnish_type"))
            checks["furnish_type"] = {"actual": furnish_val or None, "required": furnish_req, "op": "eq"}
            # "ask agent" and "flexible" should pass hard filter for furnish_type.
            if furnish_val and furnish_val not in {"ask agent", "flexible"} and furnish_val != furnish_req:
                reasons.append(f"furnish_type '{furnish_val}' != '{furnish_req}'")

        let_req = _norm_cat_text(c.get("let_type"))
        if let_req:
            let_val = _norm_cat_text(r.get("let_type"))
            checks["let_type"] = {"actual": let_val or None, "required": let_req, "op": "eq"}
            if let_val and let_val != let_req:
                reasons.append(f"let_type '{let_val}' != '{let_req}'")

        prop_req = _norm_property_type_value(c.get("property_type"))
        if prop_req:
            prop_val = _norm_property_type_value(r.get("property_type"))
            checks["property_type"] = {"actual": prop_val or None, "required": prop_req, "op": "eq"}
            # Unknown/special listing types should pass hard filter.
            if prop_val and prop_val not in {"ask agent"} and prop_val != prop_req:
                reasons.append(f"property_type '{prop_val}' != '{prop_req}'")

        tenancy_req = c.get("min_tenancy_months")
        if tenancy_req is not None:
            tenancy_val = _parse_months(r.get("min_tenancy"))
            checks["min_tenancy_months"] = {
                "actual": tenancy_val,
                "required": float(tenancy_req),
                "op": "eq",
            }
            if tenancy_val is not None and tenancy_val != float(tenancy_req):
                reasons.append(f"min_tenancy_months {tenancy_val:g} != {float(tenancy_req):g}")

        size_req = c.get("min_size_sqm")
        if size_req is not None:
            size_sqm = _to_float(r.get("size_sqm"))
            size_sqft = _to_float(r.get("size_sqft"))
            actual_sqm = size_sqm if size_sqm is not None else (size_sqft * 0.092903 if size_sqft is not None else None)
            checks["min_size_sqm"] = {
                "actual": actual_sqm,
                "required": float(size_req),
                "op": "gte",
            }
            if actual_sqm is not None and actual_sqm < float(size_req):
                reasons.append(f"size_sqm {actual_sqm:g} < {float(size_req):g}")

        hard_pass = len(reasons) == 0
        if hard_pass:
            keep_indices.append(idx)

        audits.append(
            {
                **candidate_snapshot(r),
                "hard_pass": hard_pass,
                "hard_fail_reasons": reasons,
                "hard_checks": checks,
                "score_formula": "hard_pass = all(active_hard_constraints_satisfied_or_unknown)",
                "score": 1.0 if hard_pass else 0.0,
            }
        )

    filtered = df.loc[keep_indices].copy().reset_index(drop=True)
    return filtered, audits


def _hit_ratio(text: str, terms: List[str]) -> Tuple[float, List[str]]:
    if not terms:
        return 0.0, []
    text_l = text.lower()
    hits = []
    for t in terms:
        tt = t.lower().strip()
        if not tt:
            continue
        if re.search(re.escape(tt), text_l):
            hits.append(tt)
    uniq_hits = list(dict.fromkeys(hits))
    return (len(uniq_hits) / max(1, len(list(dict.fromkeys([x.lower().strip() for x in terms if x.strip()]))))), uniq_hits

def _uniq_term_count(terms: List[str]) -> int:
    return len(list(dict.fromkeys([x.lower().strip() for x in (terms or []) if str(x).strip()])))

def _split_description_chunks(desc: str) -> List[str]:
    s = _safe_text(desc)
    if not s:
        return []
    s = s.replace("<PARA>", "\n")
    parts = []
    for p in re.split(r"[\n\r]+|(?<=[\.\!\?;])\s+", s):
        t = _safe_text(p)
        if len(t) >= 8:
            parts.append(t)
    return parts

def _embed_texts_cached(
    embedder: SentenceTransformer,
    texts: List[str],
    cache: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    missing = []
    for t in texts:
        if t not in cache:
            missing.append(t)
    if missing:
        embs = embedder.encode(
            missing,
            batch_size=min(BATCH, max(1, len(missing))),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        for t, e in zip(missing, embs):
            cache[t] = e
    return [cache[t] for t in texts]

def _sim_text(query: str, text: str, embedder: SentenceTransformer, cache: Dict[str, np.ndarray]) -> float:
    q = _safe_text(query).lower()
    t = _safe_text(text).lower()
    if not q or not t:
        return 0.0
    if q in t:
        return 1.0
    qv, tv = _embed_texts_cached(embedder, [q, t], cache)
    cos = float(np.dot(qv, tv))
    # map cosine from [-1, 1] to [0, 1] for compatibility with current scoring
    return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))

def _collect_value_candidates(r: Dict[str, Any]) -> List[Dict[str, str]]:
    cands: List[Dict[str, str]] = []
    for v in parse_jsonish_items(r.get("schools")):
        # Some sources store multiple schools in one string separated by ';'.
        # Split so each school gets an individual similarity score.
        parts = [_safe_text(x) for x in str(v).split(";")]
        for p in parts:
            if p:
                cands.append({"field": "schools", "text": p})
    for v in parse_jsonish_items(r.get("stations")):
        cands.append({"field": "stations", "text": v})
    for v in parse_jsonish_items(r.get("features")):
        cands.append({"field": "features", "text": v})
    for v in _split_description_chunks(_safe_text(r.get("description"))):
        cands.append({"field": "description", "text": v})
    return cands

def _score_single_intent(
    intent: str,
    candidates: List[Dict[str, str]],
    top_k: int,
    embedder: SentenceTransformer,
    sim_cache: Dict[str, np.ndarray],
) -> Tuple[float, str, List[Dict[str, Any]]]:
    scored = []
    school_rows = []
    for c in candidates:
        field = c.get("field", "")
        text = c.get("text", "")
        sim = _sim_text(intent, text, embedder=embedder, cache=sim_cache)
        w = float(SEMANTIC_FIELD_WEIGHTS.get(field, 0.60))
        weighted = w * sim
        scored.append((weighted, w, sim, field, text))
        if field == "schools":
            school_rows.append((weighted, sim, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max(1, top_k)]
    if not top:
        return 0.0, f"intent='{intent}' no_candidates", []
    num = sum(w * sim for _, w, sim, _, _ in top)
    den = sum(w for _, w, _, _, _ in top)
    score = (num / den) if den > 0 else 0.0
    top_show = []
    # Show all top_k matches (already sorted by weighted score desc) for transparent debugging.
    for rank, (weighted, w, sim, field, text) in enumerate(top, start=1):
        top_show.append(
            f"#{rank} {field}(weighted={weighted:.3f},w={w:.2f},sim={sim:.3f}):{text[:120]}"
        )
    top_struct = []
    for rank, (weighted, w, sim, field, text) in enumerate(top, start=1):
        top_struct.append(
            {
                "rank": int(rank),
                "field": str(field),
                "text": str(text),
                "sim": float(sim),
                "weight": float(w),
                "weighted": float(weighted),
            }
        )
    detail = (
        f"intent='{intent}' top_k={max(1, top_k)} "
        f"score={score:.4f} from weighted_mean; top_matches=[{'; '.join(top_show)}]"
    )
    if school_rows:
        school_rows.sort(key=lambda x: x[0], reverse=True)
        per_school = " ; ".join(
            f"{name[:100]} (sim={sim:.3f},weighted={weighted:.3f})"
            for weighted, sim, name in school_rows
        )
        detail += f"; school_field_scores=[{per_school}]"
    return float(score), detail, top_struct

def _score_intent_group(
    intents: List[str],
    candidates: List[Dict[str, str]],
    top_k: int,
    embedder: SentenceTransformer,
    sim_cache: Dict[str, np.ndarray],
) -> Tuple[float, List[str], str, List[Dict[str, Any]]]:
    cleaned = []
    seen = set()
    for i in intents or []:
        s = _safe_text(i).lower()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    if not cleaned:
        return 0.0, [], "no_intents", []

    scores = []
    hit_terms = []
    details = []
    selected_evidence: List[Dict[str, Any]] = []
    for it in cleaned:
        sc, dt, top_struct = _score_single_intent(
            it,
            candidates,
            top_k=top_k,
            embedder=embedder,
            sim_cache=sim_cache,
        )
        scores.append(sc)
        details.append(dt)
        if sc >= INTENT_HIT_THRESHOLD:
            hit_terms.append(it)
            for item in top_struct[:max(1, INTENT_EVIDENCE_TOP_N)]:
                selected_evidence.append(
                    {
                        "intent": str(it),
                        "intent_score": float(sc),
                        **item,
                    }
                )
    group_score = float(sum(scores) / max(1, len(scores)))
    return group_score, hit_terms, " | ".join(details), selected_evidence


def compute_stagec_weights(signals: Dict[str, Any]) -> Dict[str, float]:
    has_transit = len(signals.get("topic_preferences", {}).get("transit_terms", [])) > 0
    has_school = len(signals.get("topic_preferences", {}).get("school_terms", [])) > 0

    if has_transit and has_school:
        base = {"transit": 0.45, "school": 0.35, "preference": 0.20}
        penalty = 0.35
    elif has_transit:
        base = {"transit": 0.65, "school": 0.00, "preference": 0.35}
        penalty = 0.30
    elif has_school:
        base = {"transit": 0.00, "school": 0.65, "preference": 0.35}
        penalty = 0.30
    else:
        base = {"transit": 0.00, "school": 0.00, "preference": 1.00}
        penalty = 0.20
    return {**base, "penalty": penalty}


def rank_stage_c(
    filtered: pd.DataFrame,
    signals: Dict[str, Any],
    embedder: SentenceTransformer,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if filtered is None or len(filtered) == 0:
        return filtered, compute_stagec_weights(signals)

    out = filtered.copy()
    weights = compute_stagec_weights(signals)
    hard = signals.get("hard_constraints", {}) or {}
    transit_terms = signals.get("topic_preferences", {}).get("transit_terms", [])
    school_terms = signals.get("topic_preferences", {}).get("school_terms", [])
    pref_terms = signals.get("general_semantic", [])
    location_terms = [x.lower() for x in signals.get("location_intent", [])]

    out["transit_score"] = 0.0
    out["school_score"] = 0.0
    out["preference_score"] = 0.0
    out["penalty_score"] = 0.0
    out["location_hit_count"] = 0
    out["transit_hits"] = ""
    out["school_hits"] = ""
    out["preference_hits"] = ""
    out["penalty_reasons"] = ""
    out["transit_detail"] = ""
    out["school_detail"] = ""
    out["preference_detail"] = ""
    out["penalty_detail"] = ""
    out["transit_evidence"] = ""
    out["school_evidence"] = ""
    out["preference_evidence"] = ""
    sim_cache: Dict[str, np.ndarray] = {}

    for idx, row in out.iterrows():
        r = row.to_dict()
        stations_items = parse_jsonish_items(r.get("stations"))
        schools_items = parse_jsonish_items(r.get("schools"))

        title = _safe_text(r.get("title"))
        address = _safe_text(r.get("address"))
        desc = _safe_text(r.get("description"))
        feats = _safe_text(r.get("features"))
        stations_text = " ; ".join(stations_items)
        schools_text = " ; ".join(schools_items)

        transit_text = " ".join([stations_text, desc, feats, address]).strip()
        school_text = " ".join([schools_text, desc, feats, address]).strip()
        pref_text = " ".join([title, address, desc, feats]).strip()
        loc_text = " ".join([title, address, desc, feats, stations_text, schools_text]).lower()

        candidates = _collect_value_candidates(r)
        transit_score, transit_hits, transit_group_detail, transit_evidence = _score_intent_group(
            transit_terms,
            candidates,
            top_k=SEMANTIC_TOP_K,
            embedder=embedder,
            sim_cache=sim_cache,
        )
        school_score, school_hits, school_group_detail, school_evidence = _score_intent_group(
            school_terms,
            candidates,
            top_k=SEMANTIC_TOP_K,
            embedder=embedder,
            sim_cache=sim_cache,
        )
        pref_score, pref_hits, pref_group_detail, pref_evidence = _score_intent_group(
            pref_terms,
            candidates,
            top_k=SEMANTIC_TOP_K,
            embedder=embedder,
            sim_cache=sim_cache,
        )
        loc_hits = sum(1 for loc in location_terms if loc and loc in loc_text)

        penalties = []
        penalty_score = 0.0
        unknown_penalty_raw = 0.0
        unknown_items: List[str] = []

        # Penalize unknown values (e.g. "Ask agent") on active hard constraints.
        if hard.get("max_rent_pcm") is not None and _to_float(r.get("price_pcm")) is None:
            unknown_penalty_raw += float(UNKNOWN_PENALTY_WEIGHTS.get("price", 0.0))
            unknown_items.append("price")
        if hard.get("bedrooms") is not None and _to_float(r.get("bedrooms")) is None:
            unknown_penalty_raw += float(UNKNOWN_PENALTY_WEIGHTS.get("bedrooms", 0.0))
            unknown_items.append("bedrooms")
        if hard.get("bathrooms") is not None and _to_float(r.get("bathrooms")) is None:
            unknown_penalty_raw += float(UNKNOWN_PENALTY_WEIGHTS.get("bathrooms", 0.0))
            unknown_items.append("bathrooms")
        if hard.get("available_from") is not None and pd.isna(pd.to_datetime(r.get("available_from"), errors="coerce")):
            unknown_penalty_raw += float(UNKNOWN_PENALTY_WEIGHTS.get("available_from", 0.0))
            unknown_items.append("available_from")

        if unknown_penalty_raw > 0.0:
            unknown_penalty = min(float(UNKNOWN_PENALTY_CAP), float(unknown_penalty_raw))
            penalty_score += unknown_penalty
            penalties.append(
                f"unknown_hard({','.join(unknown_items)};+{unknown_penalty:.2f})"
            )

        # Furnish policy:
        # - ask agent: pass hard filter but apply small ranking penalty.
        # - flexible ("furnished or unfurnished, landlord is flexible"): no penalty.
        if hard.get("furnish_type"):
            furn_val = _norm_furnish_value(r.get("furnish_type"))
            if furn_val == "ask agent":
                penalty_score += float(FURNISH_ASK_AGENT_PENALTY)
                penalties.append(f"furnish_ask_agent(+{float(FURNISH_ASK_AGENT_PENALTY):.2f})")

        if transit_terms and not stations_items:
            penalty_score += 0.12
            penalties.append("missing_stations(+0.12)")
        if school_terms and not schools_items:
            penalty_score += 0.12
            penalties.append("missing_schools(+0.12)")
        if pref_terms and not pref_text:
            penalty_score += 0.08
            penalties.append("missing_text(+0.08)")

        out.at[idx, "transit_score"] = float(transit_score)
        out.at[idx, "school_score"] = float(school_score)
        out.at[idx, "preference_score"] = float(pref_score)
        out.at[idx, "penalty_score"] = float(penalty_score)
        out.at[idx, "location_hit_count"] = int(loc_hits)
        out.at[idx, "transit_hits"] = ", ".join(transit_hits)
        out.at[idx, "school_hits"] = ", ".join(school_hits)
        out.at[idx, "preference_hits"] = ", ".join(pref_hits)
        out.at[idx, "penalty_reasons"] = ", ".join(penalties)
        out.at[idx, "transit_detail"] = (
            f"group_score={transit_score:.4f}; "
            f"hits=[{', '.join(transit_hits)}]; "
            + transit_group_detail
        )
        out.at[idx, "school_detail"] = (
            f"group_score={school_score:.4f}; "
            f"hits=[{', '.join(school_hits)}]; "
            + school_group_detail
        )
        out.at[idx, "preference_detail"] = (
            f"group_score={pref_score:.4f}; "
            f"hits=[{', '.join(pref_hits)}]; "
            + pref_group_detail
        )
        out.at[idx, "penalty_detail"] = (
            f"sum(active_penalties)={penalty_score:.4f}; "
            f"triggers=[{', '.join(penalties)}]"
        )
        out.at[idx, "transit_evidence"] = json.dumps(transit_evidence, ensure_ascii=False)
        out.at[idx, "school_evidence"] = json.dumps(school_evidence, ensure_ascii=False)
        out.at[idx, "preference_evidence"] = json.dumps(pref_evidence, ensure_ascii=False)

    out["final_score"] = (
        weights["transit"] * out["transit_score"]
        + weights["school"] * out["school_score"]
        + weights["preference"] * out["preference_score"]
        - weights["penalty"] * out["penalty_score"]
    )
    out["score_formula"] = (
        f"final = {weights['transit']:.3f}*transit + "
        f"{weights['school']:.3f}*school + "
        f"{weights['preference']:.3f}*preference - "
        f"{weights['penalty']:.3f}*penalty"
    )
    out["w_transit"] = weights["transit"]
    out["w_school"] = weights["school"]
    out["w_preference"] = weights["preference"]
    out["w_penalty"] = weights["penalty"]

    out = out.sort_values(
        ["final_score", "location_hit_count", "faiss_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return out, weights


def append_ranking_log(obj: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(RANKING_LOG_PATH), exist_ok=True)
        with open(RANKING_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[warn] failed to write ranking log: {e}")


STRUCTURED_FIELDS = [
    "max_rent_pcm",
    "bedrooms",
    "bedrooms_op",
    "bathrooms",
    "bathrooms_op",
    "available_from",
    "furnish_type",
    "let_type",
    "property_type",
    "min_tenancy_months",
    "min_size_sqm",
    "location_keywords",
    "must_have_keywords",
    "k",
]

HIGH_RISK_STRUCTURED_FIELDS = {
    "max_rent_pcm",
    "bedrooms",
    "bedrooms_op",
    "bathrooms",
    "bathrooms_op",
    "available_from",
    "let_type",
    "property_type",
    "min_tenancy_months",
    "min_size_sqm",
}


def _append_jsonl(path: str, obj: Dict[str, Any], log_name: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[warn] failed to write {log_name}: {e}")


def _canon_for_structured_compare(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float):
        if np.isnan(v):
            return None
        return round(float(v), 6)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, list):
        out: List[str] = []
        seen = set()
        for x in v:
            s = str(x).strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return sorted(out, key=lambda x: x.lower())
    s = str(v).strip()
    return s if s else None


def _normalize_for_structured_policy(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = _normalize_constraint_extract(obj or {})
    out = normalize_budget_to_pcm(out)
    out = normalize_constraints(out)
    return out


def _choose_structured_value(policy: str, field: str, llm_v: Any, rule_v: Any) -> Tuple[Any, str]:
    same = _canon_for_structured_compare(llm_v) == _canon_for_structured_compare(rule_v)
    high_risk = field in HIGH_RISK_STRUCTURED_FIELDS

    if same:
        return llm_v, "agree"

    if policy == "RULE_FIRST":
        return rule_v, "override_with_rule"

    if policy == "LLM_FIRST":
        if high_risk and rule_v is not None:
            return rule_v, "guardrail_override_with_rule"
        return llm_v, "prefer_llm"

    # HYBRID: high-risk fields prefer rules, low-risk fields prefer llm.
    if high_risk:
        if rule_v is not None:
            return rule_v, "override_with_rule_high_risk"
        return llm_v, "fallback_llm_high_risk"

    if llm_v is not None:
        return llm_v, "prefer_llm_low_risk"
    if rule_v is not None:
        return rule_v, "fill_from_rule_low_risk"
    return None, "both_none"


def apply_structured_policy(
    user_text: str,
    llm_constraints: Dict[str, Any],
    rule_constraints: Dict[str, Any],
    policy: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    llm_n = _normalize_for_structured_policy(llm_constraints)
    rule_n = _normalize_for_structured_policy(rule_constraints)

    final_constraints: Dict[str, Any] = {}
    conflicts: List[Dict[str, Any]] = []
    agreements = 0

    for field in STRUCTURED_FIELDS:
        llm_v = llm_n.get(field)
        rule_v = rule_n.get(field)
        final_v, action = _choose_structured_value(policy, field, llm_v, rule_v)
        final_constraints[field] = final_v

        same = _canon_for_structured_compare(llm_v) == _canon_for_structured_compare(rule_v)
        if same:
            agreements += 1
            continue

        conflicts.append(
            {
                "field": field,
                "risk": "high" if field in HIGH_RISK_STRUCTURED_FIELDS else "low",
                "action": action,
                "llm_value": llm_v,
                "rule_value": rule_v,
                "final_value": final_v,
            }
        )

    final_constraints = _normalize_for_structured_policy(final_constraints)
    total = len(STRUCTURED_FIELDS)
    conflict_count = len(conflicts)
    agreement_rate = float(agreements) / float(total) if total > 0 else 1.0

    audit = {
        "policy": policy,
        "input_text": user_text,
        "llm_constraints": llm_n,
        "rule_constraints": rule_n,
        "final_constraints": final_constraints,
        "total_fields": total,
        "agreement_fields": agreements,
        "conflict_count": conflict_count,
        "agreement_rate": agreement_rate,
        "conflicts": conflicts,
    }
    return final_constraints, audit


def append_structured_conflict_log(
    user_text: str,
    semantic_parse_source: str,
    audit: Dict[str, Any],
) -> None:
    if not ENABLE_STRUCTURED_CONFLICT_LOG:
        return
    if not audit or int(audit.get("conflict_count", 0)) <= 0:
        return
    rec = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "policy": audit.get("policy"),
        "semantic_parse_source": semantic_parse_source,
        "user_text": user_text,
        "agreement_rate": audit.get("agreement_rate"),
        "conflict_count": audit.get("conflict_count"),
        "conflicts": audit.get("conflicts", []),
        "llm_constraints": audit.get("llm_constraints", {}),
        "rule_constraints": audit.get("rule_constraints", {}),
        "final_constraints": audit.get("final_constraints", {}),
    }
    _append_jsonl(STRUCTURED_CONFLICT_LOG_PATH, rec, "structured conflict log")


def append_structured_training_samples(
    user_text: str,
    semantic_parse_source: str,
    audit: Dict[str, Any],
) -> None:
    if not ENABLE_STRUCTURED_TRAINING_LOG:
        return
    if not audit:
        return
    conflicts = audit.get("conflicts", [])
    if not conflicts:
        return

    ts = datetime.utcnow().isoformat() + "Z"
    for item in conflicts:
        rec = {
            "timestamp": ts,
            "sample_type": "rule_disagreement_supervision",
            "policy": audit.get("policy"),
            "semantic_parse_source": semantic_parse_source,
            "user_text": user_text,
            "field": item.get("field"),
            "risk": item.get("risk"),
            "action": item.get("action"),
            "llm_value": item.get("llm_value"),
            "rule_value": item.get("rule_value"),
            "target_value": item.get("final_value"),
            "target_constraints": audit.get("final_constraints", {}),
        }
        _append_jsonl(STRUCTURED_TRAINING_LOG_PATH, rec, "structured training samples")

def format_listing_row(r: Dict[str, Any], i: int) -> str:
    title = str(r.get("title", "") or "").strip()
    url = str(r.get("url", "") or "").strip()
    address = str(r.get("address", "") or "").strip()
    price_pcm = r.get("price_pcm", None)
    beds = r.get("bedrooms", None)
    baths = r.get("bathrooms", None)

    def norm_num(x):
        if x is None:
            return None
        try:
            if isinstance(x, str):
                # allow "1998" or "£1998"
                x2 = re.sub(r"[^\d\.]", "", x)
                return float(x2) if x2 else None
            return float(x)
        except:
            return None

    price = norm_num(price_pcm)
    beds_n = None
    try:
        beds_n = int(float(beds)) if beds is not None and str(beds).strip() != "" else None
    except:
        beds_n = None

    baths_n = None
    try:
        baths_n = int(float(baths)) if baths is not None and str(baths).strip() != "" else None
    except:
        baths_n = None

    bits = []
    bits.append(f"{i}. {title}" if title else f"{i}. (no title)")
    line2 = []
    if price is not None:
        line2.append(f"£{int(round(price))}/pcm")
    else:
        # keep raw if exists
        if price_pcm is not None and str(price_pcm).strip():
            line2.append(f"{price_pcm} pcm")
    if beds_n is not None:
        line2.append(f"{beds_n} bed")
    elif beds is not None and str(beds).strip():
        line2.append(f"{beds} bed")
    if baths_n is not None:
        line2.append(f"{baths_n} bath")
    elif baths is not None and str(baths).strip():
        line2.append(f"{baths} bath")

    if address:
        line2.append(address)
    if line2:
        bits.append("   " + " | ".join(line2))
    if url:
        bits.append("   " + url)

    # scoring breakdown (if ranking fields exist)
    if r.get("final_score", None) is not None:
        def f(x):
            try:
                return f"{float(x):.4f}"
            except:
                return "0.0000"

        formula = str(r.get("score_formula", "") or "").strip()
        if formula:
            bits.append("   " + f"score: final={f(r.get('final_score'))} | {formula}")
        else:
            bits.append("   " + f"score: final={f(r.get('final_score'))}")

        if r.get("transit_score") is not None or r.get("school_score") is not None:
            bits.append(
                "   "
                + f"components: transit={f(r.get('transit_score'))}, "
                + f"school={f(r.get('school_score'))}, "
                + f"preference={f(r.get('preference_score'))}, "
                + f"penalty={f(r.get('penalty_score'))}"
            )
            try:
                wt = float(r.get("w_transit", 0.0))
                ws = float(r.get("w_school", 0.0))
                wp = float(r.get("w_preference", 0.0))
                wpen = float(r.get("w_penalty", 0.0))
                st = float(r.get("transit_score", 0.0))
                ss = float(r.get("school_score", 0.0))
                sp = float(r.get("preference_score", 0.0))
                spe = float(r.get("penalty_score", 0.0))
                c_t = wt * st
                c_s = ws * ss
                c_p = wp * sp
                c_pen = wpen * spe
                bits.append(
                    "   "
                    + "contrib: "
                    + f"transit={wt:.3f}*{st:.4f}={c_t:.4f}, "
                    + f"school={ws:.3f}*{ss:.4f}={c_s:.4f}, "
                    + f"preference={wp:.3f}*{sp:.4f}={c_p:.4f}, "
                    + f"penalty={wpen:.3f}*{spe:.4f}={c_pen:.4f}"
                )
                bits.append(
                    "   "
                    + "final_calc: "
                    + f"{c_t:.4f} + {c_s:.4f} + {c_p:.4f} - {c_pen:.4f} = "
                    + f"{(c_t + c_s + c_p - c_pen):.4f}"
                )
            except Exception:
                pass
        transit_hits = str(r.get("transit_hits", "") or "").strip()
        school_hits = str(r.get("school_hits", "") or "").strip()
        pref_hits = str(r.get("preference_hits", "") or "").strip()
        penalty_reasons = str(r.get("penalty_reasons", "") or "").strip()
        transit_detail = str(r.get("transit_detail", "") or "").strip()
        school_detail = str(r.get("school_detail", "") or "").strip()
        preference_detail = str(r.get("preference_detail", "") or "").strip()
        penalty_detail = str(r.get("penalty_detail", "") or "").strip()
        if transit_hits:
            bits.append("   " + f"transit_hits: {transit_hits}")
        if school_hits:
            bits.append("   " + f"school_hits: {school_hits}")
        if pref_hits:
            bits.append("   " + f"preference_hits: {pref_hits}")
        if penalty_reasons:
            bits.append("   " + f"penalty_reasons: {penalty_reasons}")
        if transit_detail:
            bits.append("   " + f"transit_calc: {transit_detail}")
        if school_detail:
            bits.append("   " + f"school_calc: {school_detail}")
        if preference_detail:
            bits.append("   " + f"preference_calc: {preference_detail}")
        if penalty_detail:
            bits.append("   " + f"penalty_calc: {penalty_detail}")

    evidence = r.get("evidence")
    if isinstance(evidence, dict) and evidence:
        bits.append("   evidence: " + json.dumps(evidence, ensure_ascii=False))
    return "\n".join(bits)


def build_evidence_for_row(r: Dict[str, Any], c: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
    url = str(r.get("url", "") or "").strip()
    source = str(r.get("source", "") or "").strip()
    if not source and url:
        try:
            host = (urlparse(url).netloc or "").lower()
            if "rightmove" in host:
                source = "rightmove"
            elif "zoopla" in host:
                source = "zoopla"
            elif host:
                source = host
        except Exception:
            source = ""
    if not source:
        source = "unknown"

    ev: Dict[str, Any] = {
        "source": source,
        "url": url,
    }

    fields: Dict[str, Any] = {}

    if c.get("max_rent_pcm") is not None:
        try:
            p = pd.to_numeric(r.get("price_pcm"), errors="coerce")
            fields["price_pcm"] = None if pd.isna(p) else float(p)
            fields["max_rent_pcm"] = float(c["max_rent_pcm"])
            fields["within_budget"] = None if pd.isna(p) else (float(p) <= float(c["max_rent_pcm"]))
        except Exception:
            fields["within_budget"] = None

    if c.get("bedrooms") is not None:
        try:
            b = pd.to_numeric(r.get("bedrooms"), errors="coerce")
            fields["bedrooms"] = None if pd.isna(b) else int(float(b))
            fields["bedrooms_required"] = int(c["bedrooms"])
            fields["bedrooms_op"] = str(c.get("bedrooms_op") or "eq")
        except Exception:
            fields["bedrooms"] = None

    if c.get("bathrooms") is not None:
        try:
            b = pd.to_numeric(r.get("bathrooms"), errors="coerce")
            fields["bathrooms"] = None if pd.isna(b) else float(b)
            fields["bathrooms_required"] = float(c["bathrooms"])
            fields["bathrooms_op"] = str(c.get("bathrooms_op") or "eq")
        except Exception:
            fields["bathrooms"] = None

    if c.get("available_from") is not None:
        dt = pd.to_datetime(r.get("available_from"), errors="coerce")
        fields["available_from"] = None if pd.isna(dt) else dt.date().isoformat()
        fields["available_required"] = str(c["available_from"])
        fields["available_op"] = "lte"

    if c.get("min_tenancy_months") is not None:
        try:
            raw_t = _safe_text(r.get("min_tenancy"))
            m = re.search(r"(\d+(?:\.\d+)?)", raw_t)
            fields["min_tenancy_months"] = float(m.group(1)) if m else None
            fields["min_tenancy_required_months"] = float(c["min_tenancy_months"])
        except Exception:
            fields["min_tenancy_months"] = None

    if c.get("min_size_sqm") is not None:
        try:
            sq_m = pd.to_numeric(r.get("size_sqm"), errors="coerce")
            sq_ft = pd.to_numeric(r.get("size_sqft"), errors="coerce")
            actual_sqm = None
            if not pd.isna(sq_m):
                actual_sqm = float(sq_m)
            elif not pd.isna(sq_ft):
                actual_sqm = float(sq_ft) * 0.092903
            fields["size_sqm"] = actual_sqm
            fields["min_size_required_sqm"] = float(c["min_size_sqm"])
        except Exception:
            fields["size_sqm"] = None

    # Always include key listing fields, even when no hard constraints were extracted.
    fields["price_pcm"] = None if pd.isna(pd.to_numeric(r.get("price_pcm"), errors="coerce")) else float(pd.to_numeric(r.get("price_pcm"), errors="coerce"))
    fields["bedrooms"] = None if pd.isna(pd.to_numeric(r.get("bedrooms"), errors="coerce")) else int(float(pd.to_numeric(r.get("bedrooms"), errors="coerce")))
    fields["bathrooms"] = None if pd.isna(pd.to_numeric(r.get("bathrooms"), errors="coerce")) else float(pd.to_numeric(r.get("bathrooms"), errors="coerce"))
    dt_any = pd.to_datetime(r.get("available_from"), errors="coerce")
    fields["available_from"] = None if pd.isna(dt_any) else dt_any.date().isoformat()

    ev["fields"] = fields
    pref_ctx: Dict[str, Any] = {}
    for key in ("transit_evidence", "school_evidence", "preference_evidence"):
        raw = r.get(key)
        if raw is None:
            continue
        vals: List[Dict[str, Any]] = []
        if isinstance(raw, list):
            vals = [x for x in raw if isinstance(x, dict)]
        elif isinstance(raw, str):
            s = raw.strip()
            if s:
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        vals = [x for x in parsed if isinstance(x, dict)]
                except Exception:
                    vals = []
        if vals:
            pref_ctx[key] = vals
    if pref_ctx:
        ev["preference_context"] = pref_ctx
    return ev


# ----------------------------
# Interactive CLI
# ----------------------------
def parse_command(s: str) -> Tuple[Optional[str], str]:
    s = s.strip()
    if not s.startswith("/"):
        return None, ""
    parts = s.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return cmd, arg


def run_chat():
    index, meta = load_stage_a_resources()
    embedder = SentenceTransformer(EMBED_MODEL)

    state = {
        "history": [],   # list of (user, assistant_text) for your own future use
        "k": DEFAULT_K,
        "recall": DEFAULT_RECALL,
        "last_query": None,
        "last_df": None,
        "constraints": None,
    }

    print("RentBot (minimal retrieval)")
    print("Commands: /exit /reset /k N /show /recall N /constraints /model")
    if STAGEA_BACKEND == "qdrant":
        print(f"StageA backend: qdrant")
        print(f"Qdrant path   : {QDRANT_LOCAL_PATH}")
        print(f"Collection    : {QDRANT_COLLECTION}")
        print(f"Qdrant prefilter: {QDRANT_ENABLE_PREFILTER}")
    else:
        print(f"StageA backend: faiss")
        print(f"Index: {LIST_INDEX_PATH}")
        print(f"Meta : {LIST_META_PATH}")
    print(f"Embed: {EMBED_MODEL}")
    print(f"Log  : {RANKING_LOG_PATH}")
    print(f"Structured policy: {STRUCTURED_POLICY}")
    print(f"Structured conflict log: {STRUCTURED_CONFLICT_LOG_PATH}")
    print(f"Structured training samples: {STRUCTURED_TRAINING_LOG_PATH}")
    print("----")

    while True:
        try:
            user_in = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_in:
            continue

        cmd, arg = parse_command(user_in)

        if cmd == "/exit":
            print("Bye.")
            break

        if cmd == "/reset":
            state["history"] = []
            state["last_query"] = None
            state["last_df"] = None
            state["constraints"] = None
            print("State reset.")
            continue

        if cmd == "/k":
            try:
                n = int(arg)
                if n <= 0 or n > 50:
                    raise ValueError()
                state["k"] = n
                # 关键：同时写入 constraints.k，保证后面用到的是新值
                if state["constraints"] is None:
                    state["constraints"] = {
                        "k": n,
                        "location_keywords": [],
                        "must_have_keywords": [],
                        "max_rent_pcm": None,
                        "bedrooms": None,
                        "bedrooms_op": None,
                        "bathrooms": None,
                        "bathrooms_op": None,
                        "available_from": None,
                        "available_from_op": None,
                        "furnish_type": None,
                        "let_type": None,
                        "property_type": None,
                        "min_tenancy_months": None,
                        "min_size_sqm": None,
                    }
                else:
                    state["constraints"]["k"] = n
                print(f"OK. k = {n}")
            except:
                print("Usage: /k 5   (1~50)")
            continue

        if cmd == "/recall":
            try:
                n = int(arg)
                if n <= 0 or n > 2000:
                    raise ValueError()
                state["recall"] = n
                print(f"OK. recall = {n}")
            except:
                print("Usage: /recall 200   (1~2000)")
            continue

        if cmd == "/show":
            if state["last_df"] is None or len(state["last_df"]) == 0:
                print("No previous results.")
                continue
            df = state["last_df"]
            print(f"\nBot> Showing last results (k={state['k']}, recall={state['recall']})")
            for i, r in df.iterrows():
                print(format_listing_row(r.to_dict(), i + 1))
            continue
        if cmd == "/constraints":
            print(json.dumps(state.get("constraints") or {}, ensure_ascii=False, indent=2))
            continue
        
        if cmd == "/model":
            print(f"QWEN_BASE_URL={QWEN_BASE_URL}")
            print(f"QWEN_MODEL={QWEN_MODEL}")
            print(f"RENT_STRUCTURED_POLICY={STRUCTURED_POLICY}")
            print(f"RENT_STAGEA_BACKEND={STAGEA_BACKEND}")
            print(f"RENT_QDRANT_ENABLE_PREFILTER={QDRANT_ENABLE_PREFILTER}")
            continue
        # normal query
        # query = user_in
        # k = int(state["k"])
        # recall = int(state["recall"])
        # df = faiss_search(index, embedder, meta, query=query, recall=recall, k=k)
        prev_constraints = dict(state["constraints"] or {})
        semantic_parse_source = "llm_combined"
        combined = {"constraints": {}, "semantic_terms": {}}
        llm_extracted: Dict[str, Any] = {}
        rule_extracted: Dict[str, Any] = {}
        structured_audit: Dict[str, Any] = {}
        try:
            combined = llm_extract_all_signals(user_in, state["constraints"])
            llm_extracted = combined.get("constraints") or {}
            semantic_terms = combined.get("semantic_terms") or {}
        except Exception:
            semantic_parse_source = "fallback_split_calls"
            llm_extracted = llm_extract(user_in, state["constraints"])
            semantic_terms = {}
        rule_extracted = repair_extracted_constraints(llm_extracted, user_in)
        extracted, structured_audit = apply_structured_policy(
            user_text=user_in,
            llm_constraints=llm_extracted,
            rule_constraints=rule_extracted,
            policy=STRUCTURED_POLICY,
        )
        append_structured_conflict_log(
            user_text=user_in,
            semantic_parse_source=semantic_parse_source,
            audit=structured_audit,
        )
        append_structured_training_samples(
            user_text=user_in,
            semantic_parse_source=semantic_parse_source,
            audit=structured_audit,
        )
        state["constraints"] = merge_constraints(state["constraints"], extracted)
        state["constraints"] = normalize_budget_to_pcm(state["constraints"])
        state["constraints"] = normalize_constraints(state["constraints"])

        changes_line = summarize_constraint_changes(prev_constraints, state["constraints"])
        active_line = compact_constraints_view(state["constraints"])
        c = state["constraints"] or {}
        signals = split_query_signals(
            user_in,
            c,
            precomputed_semantic_terms=semantic_terms,
            semantic_parse_source=semantic_parse_source,
        )

        print(f"[state] changes: {changes_line}")
        print(f"[state] llm_constraints: {json.dumps(llm_extracted, ensure_ascii=False)}")
        print(f"[state] rule_constraints: {json.dumps(rule_extracted, ensure_ascii=False)}")
        print(f"[state] selected_constraints: {json.dumps(extracted, ensure_ascii=False)}")
        print(
            f"[state] structured_conflicts: "
            f"policy={STRUCTURED_POLICY}, count={int(structured_audit.get('conflict_count', 0))}, "
            f"agreement_rate={float(structured_audit.get('agreement_rate', 1.0)):.3f}"
        )
        print(f"[state] llm_semantic_terms: {json.dumps(semantic_terms, ensure_ascii=False)}")
        print(f"[state] active_constraints: {json.dumps(active_line, ensure_ascii=False)}")
        print(f"[state] signals: {json.dumps(signals, ensure_ascii=False)}")

        k = int(c.get("k", DEFAULT_K) or DEFAULT_K)
        recall = int(state["recall"])
        query = build_stage_a_query(signals, user_in)

        # Stage A: recall pool
        stage_a_df = stage_a_search(index, embedder, meta, query=query, recall=recall, c=c)
        stage_a_records = []
        if stage_a_df is not None and len(stage_a_df) > 0:
            for i, row in stage_a_df.reset_index(drop=True).iterrows():
                rec = candidate_snapshot(row.to_dict())
                rec["rank"] = i + 1
                rec["score"] = rec.get("faiss_score")
                if STAGEA_BACKEND == "qdrant":
                    rec["score_formula"] = "score = qdrant_cosine_similarity(query_A, listing_embedding)"
                else:
                    rec["score_formula"] = "score = faiss_inner_product(query_A, listing_embedding)"
                stage_a_records.append(rec)

        # Stage B: hard filters (audit all candidates)
        filtered, hard_audits = apply_hard_filters_with_audit(stage_a_df, c)
        stage_b_pass_records = [x for x in hard_audits if x.get("hard_pass")]

        # Stage C: rerank only on topic/preference scores (faiss only tie-break)
        ranked, stage_c_weights = rank_stage_c(filtered, signals, embedder=embedder)
        stage_c_records = []
        if ranked is not None and len(ranked) > 0:
            for i, row in ranked.iterrows():
                rec = candidate_snapshot(row.to_dict())
                rec["rank"] = i + 1
                rec["score"] = float(row.get("final_score", 0.0))
                rec["score_formula"] = str(row.get("score_formula", ""))
                rec["components"] = {
                    "transit_score": float(row.get("transit_score", 0.0)),
                    "school_score": float(row.get("school_score", 0.0)),
                    "preference_score": float(row.get("preference_score", 0.0)),
                    "penalty_score": float(row.get("penalty_score", 0.0)),
                    "weights": stage_c_weights,
                }
                rec["hits"] = {
                    "transit_hits": str(row.get("transit_hits", "") or ""),
                    "school_hits": str(row.get("school_hits", "") or ""),
                    "preference_hits": str(row.get("preference_hits", "") or ""),
                    "penalty_reasons": str(row.get("penalty_reasons", "") or ""),
                }
                stage_c_records.append(rec)

        print(
            f"[debug] stageA_retrieved={len(stage_a_df)}, "
            f"stageB_after_hard={len(filtered)}, stageC_ranked={len(ranked)}, "
            f"k={k}, recall={recall}"
        )
        stage_d_payload: Optional[Dict[str, Any]] = None
        stage_d_output: str = ""
        stage_d_raw_output: str = ""
        stage_d_error: str = ""

        if len(filtered) < k:
            print("\nBot> 符合当前硬性条件（价格/卧室/卫生间/入住时间/配置/租期/面积）的房源不足。你可以放宽预算或修改约束。")
            df = ranked.reset_index(drop=True)
        else:
            df = ranked.head(k).reset_index(drop=True)

        if df is not None and len(df) > 0:
            df = df.copy()
            df["evidence"] = df.apply(lambda row: build_evidence_for_row(row.to_dict(), c, user_in), axis=1)


        
        if df is None or len(df) == 0:
            append_ranking_log(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "log_path": RANKING_LOG_PATH,
                    "user_query": user_in,
                    "stage_a_query": query,
                    "constraints": c,
                    "structured_audit": structured_audit,
                    "signals": signals,
                    "counts": {
                        "stage_a": len(stage_a_df),
                        "stage_b": len(filtered),
                        "stage_c": len(ranked),
                        "k": k,
                        "recall": recall,
                    },
                    "stage_a_candidates": stage_a_records,
                    "stage_b_hard_audit": hard_audits,
                    "stage_b_pass_candidates": stage_b_pass_records,
                    "stage_c_candidates": stage_c_records,
                    "stage_d": {
                        "enabled": True,
                        "system_prompt": GROUNDED_EXPLAIN_SYSTEM,
                        "payload": None,
                        "output": "",
                        "raw_output": "",
                        "error": "no_candidates",
                    },
                }
            )
            out = "I couldn't find any matching listings. Try different keywords (area, budget, bedrooms, bathrooms, available date)."
            print("\nBot> " + out)
            state["history"].append((user_in, out))
            state["last_query"] = query
            state["last_df"] = df
            continue

        # print results
        lines = [f"Top {min(k, len(df))} results:"]
        for i, r in df.iterrows():
            lines.append(format_listing_row(r.to_dict(), i + 1))
        try:
            grounded_out, stage_d_payload, stage_d_raw_output = llm_grounded_explain(
                user_query=user_in,
                c=c,
                signals=signals,
                df=df,
            )
            stage_d_output = grounded_out
            if grounded_out:
                lines.append("")
                lines.append("Grounded explanation:")
                lines.append(grounded_out)
            ev_txt = format_grounded_evidence(df=df, max_items=min(8, len(df)))
            if ev_txt:
                lines.append("")
                lines.append("Grounded evidence:")
                lines.append(ev_txt)
        except Exception as e:
            stage_d_error = str(e)
            lines.append("")
            lines.append(f"[warn] grounded explanation unavailable: {e}")
        append_ranking_log(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "log_path": RANKING_LOG_PATH,
                "user_query": user_in,
                "stage_a_query": query,
                "constraints": c,
                "structured_audit": structured_audit,
                "signals": signals,
                "counts": {
                    "stage_a": len(stage_a_df),
                    "stage_b": len(filtered),
                    "stage_c": len(ranked),
                    "k": k,
                    "recall": recall,
                },
                "stage_a_candidates": stage_a_records,
                "stage_b_hard_audit": hard_audits,
                "stage_b_pass_candidates": stage_b_pass_records,
                "stage_c_candidates": stage_c_records,
                "stage_d": {
                    "enabled": True,
                    "system_prompt": GROUNDED_EXPLAIN_SYSTEM,
                    "payload": stage_d_payload,
                    "output": stage_d_output,
                    "raw_output": stage_d_raw_output,
                    "error": stage_d_error,
                },
            }
        )
        out = "\n".join(lines)

        print("\nBot> " + out)

        state["history"].append((user_in, out))
        state["last_query"] = query
        state["last_df"] = df


if __name__ == "__main__":
    run_chat()
