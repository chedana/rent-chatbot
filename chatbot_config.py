import os
import re
from typing import Dict, List, Tuple

QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "./Qwen3-14B")
QWEN_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")

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
    "near subway", "near station", "near tube", "tube", "subway", "station", "close to station", "near metro",
    "near underground", "near tube station", "close to tube", "walk to station",
}

LET_TYPE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\\bshort\\s*[- ]?\\s*term\\b", re.I), "short term"),
    (re.compile(r"\\blong\\s*[- ]?\\s*term\\b", re.I), "long term"),
    (re.compile(r"\\bshort\\s*[- ]?\\s*let\\b", re.I), "short term"),
    (re.compile(r"\\blong\\s*[- ]?\\s*let\\b", re.I), "long term"),
    (re.compile(r"\\bst\\s*[- ]?\\s*let\\b", re.I), "short term"),
    (re.compile(r"\\blt\\s*[- ]?\\s*let\\b", re.I), "long term"),
    (re.compile(r"\\bshort\\s*[- ]?\\s*stay\\b", re.I), "short term"),
    (re.compile(r"\\blong\\s*[- ]?\\s*stay\\b", re.I), "long term"),
    (re.compile(r"\\btemporary\\s+let\\b", re.I), "short term"),
]

TENANCY_MONTH_PATTERNS: List[re.Pattern] = [
    re.compile(r"\\b(?:minimum|min)\\s+tenancy\\s*(?:of\\s*)?(\\d+(?:\\.\\d+)?)\\s*months?\\b", re.I),
    re.compile(r"\\b(?:minimum|min)\\s*tenancy\\s*(?:of\\s*)?(\\d+(?:\\.\\d+)?)\\s*months?\\b", re.I),
    re.compile(r"\\btenancy\\s*(?:of\\s*)?(\\d+(?:\\.\\d+)?)\\s*months?\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*months?\\s*(?:minimum|min)\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*months?(?:minimum|min)\\b", re.I),
    re.compile(r"\\bfor\\s+at\\s+least\\s+(\\d+(?:\\.\\d+)?)\\s*months?\\b", re.I),
    re.compile(r"\\bfor\\s*at\\s*least\\s*(\\d+(?:\\.\\d+)?)\\s*months?\\b", re.I),
    re.compile(r"\\b(?:at\\s+least|minimum|min)\\s+(\\d+(?:\\.\\d+)?)\\s*months?\\b", re.I),
    re.compile(r"\\b(?:at\\s*least|minimum|min)\\s*(\\d+(?:\\.\\d+)?)\\s*months?\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*mo\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*mos\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*mon(?:th)?s?\\b", re.I),
]

TENANCY_YEAR_FIXED_RULES: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\\bhalf\\s*(?:a\\s*)?year\\b", re.I), 6.0),
    (re.compile(r"\\bhalf\\s*(?:a\\s*)?yr\\b", re.I), 6.0),
    (re.compile(r"\\b(?:one|a)\\s*year\\b", re.I), 12.0),
    (re.compile(r"\\b(?:one|a)\\s*yr\\b", re.I), 12.0),
    (re.compile(r"\\btwo\\s*years?\\b", re.I), 24.0),
    (re.compile(r"\\btwo\\s*yrs?\\b", re.I), 24.0),
    (re.compile(r"\\bthree\\s*years?\\b", re.I), 36.0),
    (re.compile(r"\\bthree\\s*yrs?\\b", re.I), 36.0),
]
TENANCY_YEAR_NUMERIC_RULES: List[re.Pattern] = [
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*years?\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*yrs?\\b", re.I),
]

BEDROOM_EQ_PATTERNS: List[re.Pattern] = [
    re.compile(r"\\bexactly\\s*(\\d+(?:\\.\\d+)?)\\s*[- ]?bed(?:room)?s?\\b", re.I),
    re.compile(r"\\bonly\\s*(\\d+(?:\\.\\d+)?)\\s*[- ]?bed(?:room)?s?\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*[- ]?bed(?:room)?s?\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*bd\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*br\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*b\\b", re.I),
]
BATHROOM_EQ_PATTERNS: List[re.Pattern] = [
    re.compile(r"\\bexactly\\s*(\\d+(?:\\.\\d+)?)\\s*[- ]?bath(?:room)?s?\\b", re.I),
    re.compile(r"\\bonly\\s*(\\d+(?:\\.\\d+)?)\\s*[- ]?bath(?:room)?s?\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*[- ]?bath(?:room)?s?\\b", re.I),
    re.compile(r"\\b(\\d+(?:\\.\\d+)?)\\s*ba\\b", re.I),
]
BED_BATH_COMPACT_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"\\b(\\d+(?:\\.\\d+)?)\\s*(?:bed(?:room)?s?|bd|br|b)\\s*[/,-]?\\s*(\\d+(?:\\.\\d+)?)\\s*(?:bath(?:room)?s?|ba|b)\\b",
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
    (re.compile(r"\\bunfurnish(?:ed)?\\b", re.I), "unfurnished"),
    (re.compile(r"\\bpart\\s*[- ]?\\s*furnish(?:ed)?\\b", re.I), "part-furnished"),
    (re.compile(r"\\bfully\\s*[- ]?\\s*furnish(?:ed)?\\b", re.I), "furnished"),
    (re.compile(r"\\bfurnish(?:ed)?\\b", re.I), "furnished"),
]
PROPERTY_TYPE_QUERY_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\\bstudio\\b", re.I), "studio"),
    (re.compile(r"\\bapartment(?:s)?\\b", re.I), "apartment"),
    (re.compile(r"\\bapt(?:s)?\\b", re.I), "apartment"),
    (re.compile(r"\\bappartment(?:s)?\\b", re.I), "apartment"),
    (re.compile(r"\\bground\\s*flat\\b", re.I), "flat"),
    (re.compile(r"\\bflat(?:s)?\\b", re.I), "flat"),
    (re.compile(r"\\bsemi\\s*[- ]?\\s*detached\\b", re.I), "house"),
    (re.compile(r"\\bdetached\\b", re.I), "house"),
    (re.compile(r"\\btown\\s*house\\b", re.I), "house"),
    (re.compile(r"\\bterraced\\b", re.I), "house"),
    (re.compile(r"\\bmews\\b", re.I), "house"),
    (re.compile(r"\\bcottage\\b", re.I), "house"),
    (re.compile(r"\\bbungalow\\b", re.I), "house"),
    (re.compile(r"\\bhouse\\b", re.I), "house"),
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
    re.compile(
        r"\\b(?:budget|under|below|max(?:imum)?|up\\s*to|within|around|about|roughly)\\s*£?\\s*([0-9][0-9,]*(?:\\.\\d+)?)\\s*(?:pcm|per\\s*month|p/?m|pm)?\\b",
        re.I,
    ),
    re.compile(r"\\b£\\s*([0-9][0-9,]*(?:\\.\\d+)?)\\s*(?:pcm|per\\s*month|p/?m|pm)\\b", re.I),
    re.compile(r"\\b([0-9][0-9,]*(?:\\.\\d+)?)\\s*(?:pcm|per\\s*month|p/?m|pm)\\b", re.I),
]
RENT_PCW_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"\\b(?:budget|under|below|max(?:imum)?|up\\s*to|within)\\s*£?\\s*([0-9][0-9,]*(?:\\.\\d+)?)\\s*(?:pcw|per\\s*week|p/?w|pw)\\b",
        re.I,
    ),
    re.compile(r"\\b£\\s*([0-9][0-9,]*(?:\\.\\d+)?)\\s*(?:pcw|per\\s*week|p/?w|pw)\\b", re.I),
    re.compile(r"\\b([0-9][0-9,]*(?:\\.\\d+)?)\\s*(?:pcw|per\\s*week|p/?w|pw)\\b", re.I),
]

DATE_TOKEN_RE = (
    r"(?:\\d{4}[/-]\\d{1,2}[/-]\\d{1,2}"
    r"|\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}"
    r"|(?:\\d{1,2}\\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\\s,.-]+\\d{1,2}(?:[\\s,.-]+\\d{2,4})?"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\\s+\\d{1,2}(?:,\\s*\\d{2,4})?)"
)
AVAILABLE_FROM_PREFIX_PATTERNS: List[re.Pattern] = [
    re.compile(
        rf"\\b(?:by|before|no\\s+later\\s+than|latest(?:\\s+move[- ]?in|\\s+start)?(?:\\s+date)?|"
        rf"available\\s*from|starting\\s*from|start(?:ing)?\\s*from|starting|start\\s*date|from)\\s*[:=]?\\s*({DATE_TOKEN_RE})\\b",
        re.I,
    ),
]
AVAILABLE_FROM_BARE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\\b(\\d{4}[/-]\\d{1,2}[/-]\\d{1,2})\\b", re.I),
    re.compile(r"\\b(\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4})\\b", re.I),
    re.compile(
        r"\\b((?:\\d{1,2}\\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\\s,.-]+\\d{1,2}(?:[\\s,.-]+\\d{2,4})?)\\b",
        re.I,
    ),
    re.compile(r"\\b((?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\\s+\\d{1,2}(?:,\\s*\\d{2,4})?)\\b", re.I),
]
