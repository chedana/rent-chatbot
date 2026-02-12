import os
import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from chatbot_config import (
    LET_TYPE_RULES,
    TENANCY_MONTH_PATTERNS,
    TENANCY_YEAR_FIXED_RULES,
    TENANCY_YEAR_NUMERIC_RULES,
    BEDROOM_EQ_PATTERNS,
    BATHROOM_EQ_PATTERNS,
    BED_BATH_COMPACT_PATTERNS,
    NUM_WORDS,
    FURNISH_QUERY_PATTERNS,
    PROPERTY_TYPE_QUERY_PATTERNS,
    PROPERTY_TYPE_HOUSE_LIKE,
    PROPERTY_TYPE_FLAT_LIKE,
    PROPERTY_TYPE_SPECIAL_OR_UNKNOWN,
    RENT_PCM_PATTERNS,
    RENT_PCW_PATTERNS,
    AVAILABLE_FROM_PREFIX_PATTERNS,
    AVAILABLE_FROM_BARE_PATTERNS,
)

DEFAULT_K = int(os.environ.get("RENT_K", "5"))

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
