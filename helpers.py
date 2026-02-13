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
        return "other"
    if s == "studio":
        return "flat"
    if s in {"apartment", "apartments"}:
        return "flat"
    if s in {"flat", "flats"}:
        return "flat"
    if s == "house":
        return "house"
    if s in PROPERTY_TYPE_HOUSE_LIKE:
        return "house"
    if s in PROPERTY_TYPE_FLAT_LIKE:
        return "flat"
    if s in PROPERTY_TYPE_SPECIAL_OR_UNKNOWN:
        return "other"
    return s

def _infer_property_type_from_query(text: Any) -> Optional[str]:
    src = _safe_text(text).lower()
    if not src:
        return None
    for pattern, mapped in PROPERTY_TYPE_QUERY_PATTERNS:
        if pattern.search(src):
            return mapped
    return None


def _normalize_layout_options(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in (raw or []):
        if not isinstance(item, dict):
            continue
        bed = item.get("bedrooms")
        bath = item.get("bathrooms")
        ptype = item.get("property_type")
        tag = _safe_text(item.get("layout_tag")).lower()
        budget = item.get("max_rent_pcm")
        bed_n: Optional[int] = None
        bath_n: Optional[float] = None
        ptype_n: Optional[str] = None
        tag_n: Optional[str] = None
        budget_n: Optional[float] = None
        if bed is not None:
            try:
                bed_n = int(float(bed))
            except Exception:
                bed_n = None
        if bath is not None:
            try:
                bath_n = float(bath)
            except Exception:
                bath_n = None
        pnorm = _norm_property_type_value(ptype)
        if pnorm in {"flat", "house", "other"}:
            ptype_n = pnorm
        if tag in {"studio"}:
            tag_n = tag
        if budget is not None:
            try:
                budget_n = float(budget)
                if budget_n <= 0:
                    budget_n = None
            except Exception:
                budget_n = None
        key = (bed_n, bath_n, ptype_n, tag_n, budget_n)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "bedrooms": bed_n,
                "bathrooms": bath_n,
                "property_type": ptype_n,
                "layout_tag": tag_n,
                "max_rent_pcm": budget_n,
            }
        )
    return out


def _extract_layout_options_candidates(text: Any) -> List[Dict[str, Any]]:
    src = _safe_text(text).lower()
    if not src:
        return []
    for w, d in NUM_WORDS.items():
        src = re.sub(rf"\b{w}\b", d, src)

    options: List[Dict[str, Any]] = []
    used_spans: List[Tuple[int, int]] = []

    budget_patterns = [
        re.compile(r"(?:under|below|max(?:imum)?|up\s*to|within|at\s*most|budget)\s*£?\s*([0-9][0-9,]*(?:\.\d+)?)", re.I),
        re.compile(r"£\s*([0-9][0-9,]*(?:\.\d+)?)\s*(?:pcm|per\s*month|p/?m|pm)\b", re.I),
        re.compile(r"\b([0-9][0-9,]*(?:\.\d+)?)\s*(?:pcm|per\s*month|p/?m|pm)\b", re.I),
    ]

    clause_boundaries: List[Tuple[int, int]] = []
    cut_points = [0]
    for sep in re.finditer(r"\b(?:and|or)\b|[,;]", src, flags=re.I):
        cut_points.append(sep.start())
        cut_points.append(sep.end())
    cut_points.append(len(src))
    cut_points = sorted(set(x for x in cut_points if 0 <= x <= len(src)))
    for i in range(0, len(cut_points) - 1, 2):
        a = cut_points[i]
        b = cut_points[i + 1]
        if a < b:
            clause_boundaries.append((a, b))
    if not clause_boundaries:
        clause_boundaries = [(0, len(src))]

    def _clause_index(pos: int) -> int:
        for i, (a, b) in enumerate(clause_boundaries):
            if a <= pos < b:
                return i
        return max(0, len(clause_boundaries) - 1)

    budget_hits: List[Tuple[int, int, float]] = []
    for pat in budget_patterns:
        for m in pat.finditer(src):
            try:
                v = float(str(m.group(1)).replace(",", ""))
                if v > 0:
                    budget_hits.append((m.start(), m.end(), v))
            except Exception:
                continue

    def _nearest_budget(start: int, end: int, max_gap: int = 48) -> Optional[float]:
        if not budget_hits:
            return None
        cid = _clause_index(start)
        local_hits = [(bs, be, v) for bs, be, v in budget_hits if _clause_index(bs) == cid]
        hits = local_hits if local_hits else budget_hits
        # Prefer explicit price mention immediately after the layout phrase.
        after = [(bs - end, v) for bs, be, v in hits if bs >= end and (bs - end) <= max_gap]
        if after:
            after.sort(key=lambda x: x[0])
            return float(after[0][1])

        # Then allow price mention immediately before the layout phrase.
        before = [(start - be, v) for bs, be, v in hits if be <= start and (start - be) <= max_gap]
        if before:
            before.sort(key=lambda x: x[0])
            return float(before[0][1])

        # Fallback to nearest absolute distance when still close enough.
        nearest: List[Tuple[int, float]] = []
        for bs, be, v in hits:
            if be <= start:
                d = start - be
            elif bs >= end:
                d = bs - end
            else:
                d = 0
            if d <= max_gap:
                nearest.append((d, v))
        if nearest:
            nearest.sort(key=lambda x: x[0])
            return float(nearest[0][1])
        return None

    def _local_budget(start: int, end: int) -> Optional[float]:
        return _nearest_budget(start, end)

    for m in re.finditer(
        r"\b(\d+(?:\.\d+)?)\s*(?:bed(?:room)?s?|bd|br|b)\s*[/,-]?\s*(\d+(?:\.\d+)?)\s*(?:bath(?:room)?s?|ba|b)\b",
        src,
        flags=re.I,
    ):
        try:
            bed = int(float(m.group(1)))
            bath = float(m.group(2))
            options.append(
                {
                    "bedrooms": bed,
                    "bathrooms": bath,
                    "property_type": None,
                    "max_rent_pcm": _local_budget(m.start(), m.end()),
                }
            )
            used_spans.append((m.start(), m.end()))
        except Exception:
            continue

    mask = [True] * len(src)
    for a, b in used_spans:
        for i in range(max(0, a), min(len(src), b)):
            mask[i] = False
    remain = "".join(ch if mask[i] else " " for i, ch in enumerate(src))

    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:bed(?:room)?s?|bd|br)\b", remain, flags=re.I):
        try:
            bed = int(float(m.group(1)))
            options.append(
                {
                    "bedrooms": bed,
                    "bathrooms": None,
                    "property_type": None,
                    "max_rent_pcm": _local_budget(m.start(), m.end()),
                }
            )
        except Exception:
            continue

    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:bath(?:room)?s?|ba)\b", remain, flags=re.I):
        try:
            bath = float(m.group(1))
            options.append(
                {
                    "bedrooms": None,
                    "bathrooms": bath,
                    "property_type": None,
                    "max_rent_pcm": _local_budget(m.start(), m.end()),
                }
            )
        except Exception:
            continue

    for _ in re.finditer(r"\bstudio\b", src, flags=re.I):
        options.append(
            {
                "bedrooms": None,
                "bathrooms": None,
                "property_type": "flat",
                "layout_tag": "studio",
                "max_rent_pcm": _local_budget(_.start(), _.end()),
            }
        )

    inferred_ptype = _infer_property_type_from_query(src)
    if inferred_ptype is not None:
        options.append(
            {
                "bedrooms": None,
                "bathrooms": None,
                "property_type": inferred_ptype,
                "layout_tag": None,
                "max_rent_pcm": None,
            }
        )

    normalized = _normalize_layout_options(options)
    return normalized


def _infer_layout_options_from_query(text: Any) -> List[Dict[str, Any]]:
    normalized = _extract_layout_options_candidates(text)
    return normalized if len(normalized) >= 1 else []


def _infer_layout_remove_ops_from_query(text: Any) -> Dict[str, Any]:
    src = _safe_text(text).lower()
    if not src:
        return {"remove_layout_options": []}

    has_remove_verb = bool(re.search(r"\b(?:remove|drop|delete|clear)\b", src))
    remove_layout_options: List[Dict[str, Any]] = []
    if has_remove_verb:
        remove_layout_options = _extract_layout_options_candidates(src)
        # Delete selectors should match layout identity, not budget amounts.
        for it in remove_layout_options:
            if isinstance(it, dict):
                it["max_rent_pcm"] = None
    return {
        "remove_layout_options": _normalize_layout_options(remove_layout_options),
    }


def _infer_replace_all_from_query(text: Any) -> bool:
    src = _safe_text(text).lower()
    if not src:
        return False
    patterns = [
        r"\bstart over\b",
        r"\bnew search\b",
        r"\bignore (the )?(previous|last)\b",
        r"\breset (the )?(constraints|filters|search)\b",
        r"\bfrom scratch\b",
    ]
    return any(re.search(p, src) for p in patterns)


def _infer_append_mode_from_query(text: Any) -> bool:
    src = _safe_text(text).lower()
    if not src:
        return False
    patterns = [
        r"\balso\b",
        r"\bin addition\b",
        r"\bas well\b",
        r"\bplus\b",
        r"\balong with\b",
    ]
    return any(re.search(p, src) for p in patterns)


def _infer_replace_mode_from_query(text: Any) -> bool:
    src = _safe_text(text).lower()
    if not src:
        return False
    patterns = [
        r"\binstead\b",
        r"\bswitch to\b",
        r"\bchange to\b",
        r"\breplace\b",
    ]
    return any(re.search(p, src) for p in patterns)


def _infer_clear_location_from_query(text: Any) -> bool:
    src = _safe_text(text).lower()
    if not src:
        return False
    patterns = [
        r"\bany location\b",
        r"\bno location preference\b",
        r"\bremove location\b",
        r"\bdon'?t care (about )?location\b",
    ]
    return any(re.search(p, src) for p in patterns)

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
    inferred_furnish = _infer_furnish_type_from_query(user_text)
    inferred_max_rent_pcm = _infer_max_rent_pcm_from_query(user_text)
    inferred_layout_options = _infer_layout_options_from_query(user_text)
    inferred_layout_remove_ops = _infer_layout_remove_ops_from_query(user_text)
    inferred_replace_all = _infer_replace_all_from_query(user_text)
    inferred_append = _infer_append_mode_from_query(user_text)
    inferred_replace = _infer_replace_mode_from_query(user_text)
    inferred_clear_location = _infer_clear_location_from_query(user_text)
    remove_opts = inferred_layout_remove_ops.get("remove_layout_options") or []

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

    if inferred_furnish is not None:
        out["furnish_type"] = inferred_furnish

    if inferred_max_rent_pcm is not None:
        out["max_rent_pcm"] = inferred_max_rent_pcm

    if len(inferred_layout_options) >= 1 and not remove_opts:
        out["layout_options"] = inferred_layout_options

    if remove_opts:
        out["_remove_layout_options"] = remove_opts

    llm_scope = _safe_text(out.get("update_scope")).lower()
    llm_replace_all = out.get("_replace_all_constraints")
    if not isinstance(llm_replace_all, bool):
        llm_replace_all = (llm_scope == "replace_all")
    if isinstance(llm_replace_all, bool):
        out["_replace_all_constraints"] = bool(llm_replace_all or inferred_replace_all)
    else:
        out["_replace_all_constraints"] = bool(inferred_replace_all)

    llm_loc_mode = _safe_text(out.get("_location_update_mode")).lower()
    if llm_loc_mode not in {"keep", "replace", "append"}:
        llm_loc_mode = _safe_text(out.get("location_update_mode")).lower()
    if inferred_append and not inferred_replace:
        out["_location_update_mode"] = "append"
    elif inferred_replace:
        out["_location_update_mode"] = "replace"
    elif llm_loc_mode in {"keep", "replace", "append"}:
        out["_location_update_mode"] = llm_loc_mode
    else:
        out["_location_update_mode"] = "replace"
    out["_clear_location_keywords"] = bool(inferred_clear_location)

    llm_layout_mode = _safe_text(out.get("_layout_update_mode")).lower()
    if llm_layout_mode not in {"replace", "append"}:
        llm_layout_mode = _safe_text(out.get("layout_update_mode")).lower()
    if inferred_append and not inferred_replace:
        out["_layout_update_mode"] = "append"
    elif inferred_replace:
        out["_layout_update_mode"] = "replace"
    elif llm_layout_mode in {"replace", "append"}:
        out["_layout_update_mode"] = llm_layout_mode
    else:
        out["_layout_update_mode"] = "replace"

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
    obj.setdefault("available_from", None)
    obj.setdefault("available_from_op", None)
    obj.setdefault("furnish_type", None)
    obj.setdefault("let_type", None)
    obj.setdefault("layout_options", [])
    obj.setdefault("min_tenancy_months", None)
    obj.setdefault("min_size_sqm", None)
    obj.setdefault("min_size_sqft", None)
    obj.setdefault("location_keywords", [])
    obj.setdefault("update_scope", "patch")
    obj.setdefault("location_update_mode", "replace")
    obj.setdefault("layout_update_mode", "replace")
    obj.setdefault("_replace_all_constraints", False)
    obj.setdefault("_location_update_mode", "replace")
    obj.setdefault("_layout_update_mode", "replace")
    obj.setdefault("_clear_location_keywords", False)
    obj.setdefault("_remove_layout_options", [])
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
    c["layout_options"] = _normalize_layout_options(c.get("layout_options") or [])

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

    return c

def merge_constraints(old: Optional[dict], new: dict) -> dict:
    if old is None:
        old = {}
    out = dict(old)
    if bool(new.get("_replace_all_constraints")):
        old_k = out.get("k")
        out = {}
        if old_k is not None and new.get("k") is None:
            out["k"] = old_k

    def _layout_selector_match(item: Dict[str, Any], selector: Dict[str, Any]) -> bool:
        if not isinstance(item, dict) or not isinstance(selector, dict):
            return False
        for k in ("bedrooms", "bathrooms", "property_type", "layout_tag"):
            sv = selector.get(k)
            if sv is None:
                continue
            if item.get(k) != sv:
                return False
        return True

    cur_layout = _normalize_layout_options(out.get("layout_options") or [])
    remove_selectors = _normalize_layout_options(new.get("_remove_layout_options") or [])
    if remove_selectors:
        kept = []
        for opt in cur_layout:
            if any(_layout_selector_match(opt, sel) for sel in remove_selectors):
                continue
            kept.append(opt)
        cur_layout = kept
    out["layout_options"] = cur_layout

    # scalar fields: new overrides if not null
    for key in [
        "max_rent_pcm",
        "available_from",
        "furnish_type",
        "let_type",
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

    old_locs = old.get("location_keywords") or []
    new_locs = new.get("location_keywords") or []
    if bool(new.get("_clear_location_keywords")):
        out["location_keywords"] = []
    else:
        location_mode = _safe_text(new.get("_location_update_mode")).lower()
        if location_mode == "append":
            out["location_keywords"] = merge_list(old_locs, new_locs)
        elif location_mode == "keep":
            out["location_keywords"] = merge_list(old_locs, [])
        else:
            # default hard-filter behavior: replace when new location is explicitly provided.
            out["location_keywords"] = merge_list(new_locs, []) if len(new_locs) > 0 else merge_list(old_locs, [])

    new_layout = _normalize_layout_options(new.get("layout_options") or [])
    if new_layout:
        layout_mode = _safe_text(new.get("_layout_update_mode")).lower()
        if layout_mode == "append":
            out["layout_options"] = _normalize_layout_options((out.get("layout_options") or []) + new_layout)
        else:
            out["layout_options"] = new_layout

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
        "available_from",
        "furnish_type", "let_type",
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

    for k in ["location_keywords"]:
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

    old_layout = _normalize_layout_options(old_c.get("layout_options") or [])
    new_layout = _normalize_layout_options(new_c.get("layout_options") or [])
    if _canon_for_structured_compare(old_layout) != _canon_for_structured_compare(new_layout):
        if not old_layout and new_layout:
            changes.append(f"added layout_options: {new_layout}")
        elif old_layout and not new_layout:
            changes.append("removed layout_options")
        else:
            changes.append(f"updated layout_options: {old_layout} -> {new_layout}")

    return "; ".join(changes) if changes else "no constraint changes"


def compact_constraints_view(c: Optional[dict]) -> dict:
    c = c or {}
    out = {}
    keep_keys = [
        "max_rent_pcm",
        "available_from",
        "furnish_type", "let_type", "layout_options",
        "min_tenancy_months", "min_size_sqm",
        "location_keywords",
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
        if len(v) > 0 and isinstance(v[0], dict):
            norm = _normalize_layout_options(v)
            canon_items = []
            for it in norm:
                canon_items.append(
                    {
                        "bedrooms": it.get("bedrooms"),
                        "bathrooms": it.get("bathrooms"),
                        "property_type": it.get("property_type"),
                        "layout_tag": it.get("layout_tag"),
                        "max_rent_pcm": it.get("max_rent_pcm"),
                    }
                )
            return sorted(
                canon_items,
                key=lambda x: (
                    -1 if x.get("bedrooms") is None else int(x.get("bedrooms")),
                    -1 if x.get("bathrooms") is None else float(x.get("bathrooms")),
                    str(x.get("property_type") or ""),
                    str(x.get("layout_tag") or ""),
                    -1 if x.get("max_rent_pcm") is None else float(x.get("max_rent_pcm")),
                ),
            )
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
