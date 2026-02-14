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
  "available_from": string|null,
  "furnish_type": string|null,
  "let_type": string|null,
  "layout_options": [{"bedrooms": int|null, "bathrooms": number|null, "property_type": string|null, "layout_tag": string|null, "max_rent_pcm": number|null}],
  "min_tenancy_months": number|null,
  "min_size_sqm": number|null,
  "min_size_sqft": number|null,
  "location_keywords": string[],
  "k": int|null,
  "update_scope": string|null,
  "location_update_mode": string|null,
  "layout_update_mode": string|null
}
Rules:
- location_keywords are place names/areas/postcodes (e.g., "Canary Wharf", "E14", "Shoreditch").
- available_from should be an ISO date string "YYYY-MM-DD" when possible.
- available_from means user's latest move-in date.
- Do not output available_from_op.
- furnish_type should be one of: "furnished", "unfurnished", "part-furnished", or null.
- let_type examples: "long term", "short term", or null.
- Use layout_options as the only layout constraint representation.
- layout_options is for explicit layout alternatives (OR set). Each item is one acceptable layout option.
  - layout_tag currently supports: "studio" (optional).
  Examples:
  - "1 bed and 2 bed" -> [{"bedrooms":1,"bathrooms":null,"property_type":null,"layout_tag":null,"max_rent_pcm":null},{"bedrooms":2,"bathrooms":null,"property_type":null,"layout_tag":null,"max_rent_pcm":null}]
  - "studio and 1 bed" -> [{"bedrooms":null,"bathrooms":null,"property_type":"flat","layout_tag":"studio","max_rent_pcm":null},{"bedrooms":1,"bathrooms":null,"property_type":null,"layout_tag":null,"max_rent_pcm":null}]
  - "1b1b 2b1b" -> [{"bedrooms":1,"bathrooms":1,"property_type":null,"layout_tag":null,"max_rent_pcm":null},{"bedrooms":2,"bathrooms":1,"property_type":null,"layout_tag":null,"max_rent_pcm":null}]
  - "1b1b under 2600 and 2b2b under 3400" -> [{"bedrooms":1,"bathrooms":1,"property_type":null,"layout_tag":null,"max_rent_pcm":2600},{"bedrooms":2,"bathrooms":2,"property_type":null,"layout_tag":null,"max_rent_pcm":3400}]
  - "1b1b/2b1b" -> same as above.
  - If only one layout is requested, still output one item in layout_options.
- min_tenancy_months is numeric months (e.g., 6, 12) when user specifies tenancy term.
- size constraints:
  - "at least X sqm/sq m/m2" -> min_size_sqm = X
  - "at least X sqft/sq ft/ft2" -> min_size_sqft = X
- If unknown use null or [].
- update_scope: "patch" (default) or "replace_all".
- location_update_mode: "replace" (default), "append", or "keep".
- layout_update_mode: "replace" (default) or "append".
- Use "replace_all" only when user explicitly resets search (e.g., "start over", "ignore previous", "new search").
- Use append only when user explicitly adds options (e.g., "also", "in addition", "plus", "as well as").
- If no explicit update intent is present, set update_scope="patch", location_update_mode="replace", layout_update_mode="replace".
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
    "available_from": string|null,
    "furnish_type": string|null,
    "let_type": string|null,
    "layout_options": [{"bedrooms": int|null, "bathrooms": number|null, "property_type": string|null, "layout_tag": string|null, "max_rent_pcm": number|null}],
    "min_tenancy_months": number|null,
    "min_size_sqm": number|null,
    "min_size_sqft": number|null,
    "location_keywords": string[],
    "k": int|null,
    "update_scope": string|null,
    "location_update_mode": string|null,
    "layout_update_mode": string|null
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
- For any explicit layout request (single or multiple), put them into constraints.layout_options.
- Keep named entities as full phrases (e.g., "Seven Mills Primary School", "Heron Quays Station").
- Do NOT put hard constraints into semantic_terms (budget, bedroom count, property type, strict location filters).
- Do NOT split one entity into component words.
- Avoid generic filler terms like "school" when a concrete entity/phrase exists.
- If unknown use null or [].
- update_scope: "patch" (default) or "replace_all".
- location_update_mode: "replace" (default), "append", or "keep".
- layout_update_mode: "replace" (default) or "append".
- Use "replace_all" only when user explicitly resets search (e.g., "start over", "ignore previous", "new search").
- Use append only when user explicitly adds options (e.g., "also", "in addition", "plus", "as well as").
- If no explicit update intent is present, set update_scope="patch", location_update_mode="replace", layout_update_mode="replace".
"""

GROUNDED_EXPLAIN_SYSTEM = """You are a rental recommendation summarization assistant.
Input JSON contains:
- user_query: user rental intent and constraints
- top_k_candidates: Stage C ranked Top K listings

You MUST:
1) Preserve the exact rank order from input (no re-ranking).
2) For each listing, generate query-aligned recommendation summary.
3) Use features and description as the primary evidence.
4) Generate risk flags from deposit and ask_agent_items.
5) Output STRICT JSON only. No markdown. No extra commentary.

Constraints:
- Do not invent facts.
- If information is missing, use:
  - "Not explicitly stated in listing."
  - "Query constraint not verifiable from listing data."
- summary_reason:
  - 2-4 sentences, <= 90 words.
  - first sentence must align listing to user_query.
- highlights:
  - 1-3 items
  - category must be one of:
    - location_access
    - layout_space
    - amenities_building
    - condition_movein
  - claim <= 16 words
  - evidence must be a short quoted phrase from features/description (<= 12 words)
  - if no quote exists: "Not explicitly stated in listing"

Risk rules:
- risk_flags is required for every listing, min 1 max 3.
- deposit_risk_level:
  - HIGH: deposit >= 1.5 * monthly_rent
  - MEDIUM: 1.0 * monthly_rent <= deposit < 1.5 * monthly_rent
  - LOW: deposit < 1.0 * monthly_rent
  - UNKNOWN: rent or deposit missing/unparseable
- If ask_agent_items has unresolved entries, include at least one corresponding risk flag.
- risk flag format:
  - "<Risk Level>: <Issue>. <Why it matters / what is unknown>."

Output schema:
{
  "stage": "D",
  "top_k": <INTEGER>,
  "summary_generated_at": "<ISO_DATETIME>",
  "recommendations": [
    {
      "rank": 1,
      "listing_id": "string",
      "query_alignment": "Strong|Moderate|Weak match + brief reason",
      "summary_reason": "string",
      "highlights": [
        {
          "category": "location_access|layout_space|amenities_building|condition_movein",
          "claim": "string",
          "evidence": "string"
        }
      ],
      "deposit_risk_level": "LOW|MEDIUM|HIGH|UNKNOWN",
      "risk_flags": ["string"]
    }
  ]
}
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
    (re.compile(r"\\bstudio\\b", re.I), "flat"),
    (re.compile(r"\\bapartment(?:s)?\\b", re.I), "flat"),
    (re.compile(r"\\bapt(?:s)?\\b", re.I), "flat"),
    (re.compile(r"\\bappartment(?:s)?\\b", re.I), "flat"),
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
