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


from openai import OpenAI

QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL    = os.environ.get("QWEN_MODEL", "./Qwen2.5-7B-Instruct")  # 你启动时的 model 名称
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
  "available_from_op": string|null,
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
- available_from_op must be one of: "gte", "lte", or null.
- Set available_from/available_from_op as:
  - "available from DATE" -> {"available_from": "DATE", "available_from_op": "gte"}
  - "available by/before DATE" -> {"available_from": "DATE", "available_from_op": "lte"}
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
    "available_from_op": string|null,
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

NEAR_WORDS = {
    "near subway","near station","near tube","tube","subway","station","close to station","near metro",
    "near underground","near tube station","close to tube","walk to station"
}

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

    # normalize available_from hard-constraint operator
    date_op = c.get("available_from_op")
    if date_op is not None:
        date_op = str(date_op).strip().lower()
        if date_op in (">=", "after", "from", "available from", "no_earlier_than", "no earlier than", "gte"):
            date_op = "gte"
        elif date_op in ("<=", "before", "by", "no_later_than", "no later than", "lte"):
            date_op = "lte"
        else:
            date_op = None
    c["available_from_op"] = date_op

    if c.get("available_from") is not None:
        dt = pd.to_datetime(c.get("available_from"), errors="coerce")
        if pd.notna(dt):
            c["available_from"] = dt.date().isoformat()
        else:
            c["available_from"] = None

    # default to "available from" style when date exists but op is absent
    if c.get("available_from") is not None and c.get("available_from_op") is None:
        c["available_from_op"] = "gte"

    def _norm_cat_text(v: Any) -> Optional[str]:
        s = _safe_text(v).lower()
        if not s:
            return None
        s = s.replace("_", " ").replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s or None

    furn = _norm_cat_text(c.get("furnish_type"))
    if furn:
        if "unfurn" in furn:
            furn = "unfurnished"
        elif "part" in furn and "furnish" in furn:
            furn = "part-furnished"
        elif "furnish" in furn:
            furn = "furnished"
    c["furnish_type"] = furn
    c["let_type"] = _norm_cat_text(c.get("let_type"))
    c["property_type"] = _norm_cat_text(c.get("property_type"))

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

    locs = []
    must = set([str(x).strip() for x in (c.get("must_have_keywords") or []) if str(x).strip()])
    for x in (c.get("location_keywords") or []):
        s = str(x).strip()
        if not s:
            continue
        if s.lower() in NEAR_WORDS:
            must.add(s)
        else:
            locs.append(s)
    c["location_keywords"] = locs
    c["must_have_keywords"] = list(must)
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
        "available_from_op",
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
        "available_from", "available_from_op",
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
        "available_from", "available_from_op",
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
        dt_op = c.get("available_from_op", "gte")
        if dt_op == "gte":
            parts.append(f"available from {c['available_from']}")
        else:
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
DEFAULT_INDEX_DIR = NEW_INDEX_DIR if os.path.exists(NEW_INDEX_DIR) else LEGACY_INDEX_DIR

OUT_DIR = os.environ.get("RENT_INDEX_DIR", DEFAULT_INDEX_DIR)

LIST_INDEX_PATH = os.path.join(OUT_DIR, "listings_hnsw.faiss")
LIST_META_PATH  = os.path.join(OUT_DIR, "listings_meta.parquet")


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
FAISS_SCORE_WEIGHT = float(os.environ.get("RENT_FAISS_WEIGHT", "0.35"))
PREFERENCE_SCORE_WEIGHT = float(os.environ.get("RENT_PREF_WEIGHT", "1.0"))
UNKNOWN_PENALTY_WEIGHT = float(os.environ.get("RENT_UNKNOWN_PENALTY_WEIGHT", "1.0"))
SOFT_PENALTY_WEIGHT = float(os.environ.get("RENT_SOFT_PENALTY_WEIGHT", "1.0"))
RANKING_LOG_PATH = os.environ.get(
    "RENT_RANKING_LOG_PATH",
    os.path.join(ROOT_DIR, "artifacts", "debug", "ranking_log.jsonl"),
)
SEMANTIC_TOP_K = int(os.environ.get("RENT_SEMANTIC_TOPK", "4"))
SEMANTIC_FIELD_WEIGHTS = {
    "schools": 1.00,
    "stations": 1.00,
    "features": 0.80,
    "description": 0.60,
}

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

def embed_query(embedder: SentenceTransformer, q: str) -> np.ndarray:
    x = embedder.encode(
        [q],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,  # match your builder, then normalize with faiss
    ).astype("float32")
    faiss.normalize_L2(x)  # IMPORTANT: match builder
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

    for kw in must_keywords:
        kwl = kw.lower()
        if any(t in kwl for t in TRANSIT_KEYWORDS):
            transit_terms.append(kwl)
        elif any(t in kwl for t in SCHOOL_KEYWORDS):
            school_terms.append(kwl)

    general_semantic = [str(x).strip().lower() for x in model_terms.get("general_semantic_phrases", []) if str(x).strip()]
    seen = set()
    for t in general_semantic:
        seen.add(t)
    for tok in [str(x).strip().lower() for x in must_keywords]:
        t = tok.lower().strip()
        if not t or t in seen:
            continue
        seen.add(t)
        general_semantic.append(t)
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
            "available_from_op": c.get("available_from_op"),
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
            s = _norm_cat_text(v)
            if not s:
                return ""
            if "unfurn" in s:
                return "unfurnished"
            if "part" in s and "furnish" in s:
                return "part-furnished"
            if "furnish" in s:
                return "furnished"
            return s

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
            if furnish_val and furnish_val != furnish_req:
                reasons.append(f"furnish_type '{furnish_val}' != '{furnish_req}'")

        let_req = _norm_cat_text(c.get("let_type"))
        if let_req:
            let_val = _norm_cat_text(r.get("let_type"))
            checks["let_type"] = {"actual": let_val or None, "required": let_req, "op": "eq"}
            if let_val and let_val != let_req:
                reasons.append(f"let_type '{let_val}' != '{let_req}'")

        prop_req = _norm_cat_text(c.get("property_type"))
        if prop_req:
            prop_val = _norm_cat_text(r.get("property_type"))
            checks["property_type"] = {"actual": prop_val or None, "required": prop_req, "op": "eq"}
            if prop_val and prop_val != prop_req:
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
) -> Tuple[float, str]:
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
        return 0.0, f"intent='{intent}' no_candidates"
    num = sum(w * sim for _, w, sim, _, _ in top)
    den = sum(w for _, w, _, _, _ in top)
    score = (num / den) if den > 0 else 0.0
    top_show = []
    # Show all top_k matches (already sorted by weighted score desc) for transparent debugging.
    for rank, (weighted, w, sim, field, text) in enumerate(top, start=1):
        top_show.append(
            f"#{rank} {field}(weighted={weighted:.3f},w={w:.2f},sim={sim:.3f}):{text[:120]}"
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
    return float(score), detail

def _score_intent_group(
    intents: List[str],
    candidates: List[Dict[str, str]],
    top_k: int,
    embedder: SentenceTransformer,
    sim_cache: Dict[str, np.ndarray],
) -> Tuple[float, List[str], str]:
    cleaned = []
    seen = set()
    for i in intents or []:
        s = _safe_text(i).lower()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    if not cleaned:
        return 0.0, [], "no_intents"

    scores = []
    hit_terms = []
    details = []
    for it in cleaned:
        sc, dt = _score_single_intent(
            it,
            candidates,
            top_k=top_k,
            embedder=embedder,
            sim_cache=sim_cache,
        )
        scores.append(sc)
        details.append(dt)
        if sc >= 0.45:
            hit_terms.append(it)
    group_score = float(sum(scores) / max(1, len(scores)))
    return group_score, hit_terms, " | ".join(details)


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
        transit_score, transit_hits, transit_group_detail = _score_intent_group(
            transit_terms,
            candidates,
            top_k=SEMANTIC_TOP_K,
            embedder=embedder,
            sim_cache=sim_cache,
        )
        school_score, school_hits, school_group_detail = _score_intent_group(
            school_terms,
            candidates,
            top_k=SEMANTIC_TOP_K,
            embedder=embedder,
            sim_cache=sim_cache,
        )
        pref_score, pref_hits, pref_group_detail = _score_intent_group(
            pref_terms,
            candidates,
            top_k=SEMANTIC_TOP_K,
            embedder=embedder,
            sim_cache=sim_cache,
        )
        loc_hits = sum(1 for loc in location_terms if loc and loc in loc_text)

        penalties = []
        penalty_score = 0.0
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
        fields["available_op"] = str(c.get("available_from_op") or "gte")

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
    index, meta = load_index_and_meta()
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
    print(f"Index: {LIST_INDEX_PATH}")
    print(f"Meta : {LIST_META_PATH}")
    print(f"Embed: {EMBED_MODEL}")
    print(f"Log  : {RANKING_LOG_PATH}")
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
            continue
        # normal query
        # query = user_in
        # k = int(state["k"])
        # recall = int(state["recall"])
        # df = faiss_search(index, embedder, meta, query=query, recall=recall, k=k)
        prev_constraints = dict(state["constraints"] or {})
        semantic_parse_source = "llm_combined"
        combined = {"constraints": {}, "semantic_terms": {}}
        try:
            combined = llm_extract_all_signals(user_in, state["constraints"])
            extracted = combined.get("constraints") or {}
            semantic_terms = combined.get("semantic_terms") or {}
        except Exception:
            semantic_parse_source = "fallback_split_calls"
            extracted = llm_extract(user_in, state["constraints"])
            semantic_terms = {}
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
        print(f"[state] active_constraints: {json.dumps(active_line, ensure_ascii=False)}")
        print(f"[state] signals: {json.dumps(signals, ensure_ascii=False)}")

        k = int(c.get("k", DEFAULT_K) or DEFAULT_K)
        recall = int(state["recall"])
        query = build_stage_a_query(signals, user_in)

        # Stage A: recall pool
        stage_a_df = faiss_search(index, embedder, meta, query=query, recall=recall)
        stage_a_records = []
        if stage_a_df is not None and len(stage_a_df) > 0:
            for i, row in stage_a_df.reset_index(drop=True).iterrows():
                rec = candidate_snapshot(row.to_dict())
                rec["rank"] = i + 1
                rec["score"] = rec.get("faiss_score")
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

        append_ranking_log(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "log_path": RANKING_LOG_PATH,
                "user_query": user_in,
                "stage_a_query": query,
                "constraints": c,
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
            }
        )

        if len(filtered) < k:
            print("\nBot> 符合当前硬性条件（价格/卧室/卫生间/入住时间/配置/租期/面积）的房源不足。你可以放宽预算或修改约束。")
            df = ranked.reset_index(drop=True)
        else:
            df = ranked.head(k).reset_index(drop=True)

        if df is not None and len(df) > 0:
            df = df.copy()
            df["evidence"] = df.apply(lambda row: build_evidence_for_row(row.to_dict(), c, user_in), axis=1)


        
        if df is None or len(df) == 0:
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
        out = "\n".join(lines)

        print("\nBot> " + out)

        state["history"].append((user_in, out))
        state["last_query"] = query
        state["last_df"] = df


if __name__ == "__main__":
    run_chat()
