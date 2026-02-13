import re
import json
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from chatbot_config import (QWEN_BASE_URL, QWEN_MODEL, GROUNDED_EXPLAIN_SYSTEM)
from qwen import (
    format_grounded_evidence,
    llm_extract,
    llm_extract_all_signals,
    llm_grounded_explain,
)
from qdrant_search import load_stage_a_resources, stage_a_search
from internal_helpers import (
    HIGH_RISK_STRUCTURED_FIELDS,
    _choose_structured_value,
    _collect_value_candidates,
    _score_intent_group,
)
from helpers import (
    _canon_for_structured_compare,
    _norm_furnish_value,
    _normalize_for_structured_policy,
    _safe_text,
    _to_float,
    compact_constraints_view,
    merge_constraints,
    normalize_budget_to_pcm,
    normalize_constraints,
    parse_jsonish_items,
    repair_extracted_constraints,
    summarize_constraint_changes,
)
from log import (
    LOG_LEVEL,
    RANKING_LOG_DETAIL,
    append_jsonl,
    append_ranking_log_entry,
    log_message,
)
from settings import (
    BATCH,
    DEFAULT_K,
    DEFAULT_RECALL,
    EMBED_MODEL,
    ENABLE_STRUCTURED_CONFLICT_LOG,
    ENABLE_STRUCTURED_TRAINING_LOG,
    INTENT_EVIDENCE_TOP_N,
    INTENT_HIT_THRESHOLD,
    QDRANT_COLLECTION,
    QDRANT_ENABLE_PREFILTER,
    QDRANT_LOCAL_PATH,
    RANKING_LOG_PATH,
    SEMANTIC_FIELD_WEIGHTS,
    SEMANTIC_TOP_K,
    STRUCTURED_CONFLICT_LOG_PATH,
    STRUCTURED_POLICY,
    STRUCTURED_TRAINING_LOG_PATH,
    UNKNOWN_PENALTY_CAP,
    UNKNOWN_PENALTY_WEIGHTS,
    VERBOSE_STATE_LOG,
)

def split_query_signals(
    user_in: str,
    c: Dict[str, Any],
    precomputed_semantic_terms: Optional[Dict[str, Any]] = None,
    semantic_parse_source: str = "llm",
) -> Dict[str, Any]:
    c = c or {}
    location_intent = [str(x).strip() for x in (c.get("location_keywords") or []) if str(x).strip()]

    model_terms = precomputed_semantic_terms or {
        "transit_terms": [],
        "school_terms": [],
        "general_semantic_phrases": [],
    }
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
            "available_from": c.get("available_from"),
            "furnish_type": c.get("furnish_type"),
            "let_type": c.get("let_type"),
            "layout_options": c.get("layout_options") or [],
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
    if hard.get("max_rent_pcm") is not None:
        parts.append(f"under {int(float(hard.get('max_rent_pcm')))} pcm")
    if hard.get("furnish_type"):
        parts.append(str(hard.get("furnish_type")))
    if hard.get("let_type"):
        parts.append(str(hard.get("let_type")))
    for opt in (hard.get("layout_options") or []):
        if not isinstance(opt, dict):
            continue
        bed = opt.get("bedrooms")
        bath = opt.get("bathrooms")
        ptype = opt.get("property_type")
        ltag = str(opt.get("layout_tag") or "").strip().lower()
        obudget = opt.get("max_rent_pcm")
        seg: List[str] = []
        if ltag == "studio":
            seg.append("studio")
        if bed is not None:
            try:
                seg.append(f"{int(float(bed))} bedroom")
            except Exception:
                pass
        if bath is not None:
            try:
                seg.append(f"{float(bath):g} bathroom")
            except Exception:
                pass
        if ptype:
            seg.append(str(ptype))
        if obudget is not None:
            try:
                seg.append(f"under {int(float(obudget))} pcm")
            except Exception:
                pass
        if seg:
            parts.append(" ".join(seg))
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
        "retrieval_score": _to_float(r.get("retrieval_score")),
        "qdrant_score": _to_float(r.get("qdrant_score")),
        "_qdrant_id": _safe_text(r.get("_qdrant_id")) or None,
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

        layout_options = c.get("layout_options") or []
        use_layout_options = isinstance(layout_options, list) and len(layout_options) > 0

        if use_layout_options:
            prop_val = _safe_text(r.get("property_type")).lower()
            bed_val = _to_float(r.get("bedrooms"))
            bath_val = _to_float(r.get("bathrooms"))
            option_audits: List[Dict[str, Any]] = []
            any_pass = False

            for opt in layout_options:
                if not isinstance(opt, dict):
                    continue
                req_bed = opt.get("bedrooms")
                req_bath = opt.get("bathrooms")
                req_prop = _safe_text(opt.get("property_type")).lower()
                req_tag = str(opt.get("layout_tag") or "").strip().lower()
                req_rent = opt.get("max_rent_pcm")
                opt_fail: List[str] = []

                if req_bed is not None and bed_val is not None:
                    try:
                        if int(round(bed_val)) != int(float(req_bed)):
                            opt_fail.append(f"bedrooms {bed_val:g} != {int(float(req_bed))}")
                    except Exception:
                        pass
                if req_bath is not None and bath_val is not None:
                    try:
                        if float(bath_val) != float(req_bath):
                            opt_fail.append(f"bathrooms {bath_val:g} != {float(req_bath):g}")
                    except Exception:
                        pass
                if req_prop:
                    if prop_val and prop_val != req_prop:
                        opt_fail.append(f"property_type '{prop_val}' != '{req_prop}'")
                if req_tag == "studio":
                    raw_prop = _safe_text(r.get("property_type")).lower()
                    is_raw_studio = (raw_prop == "studio")
                    is_flat_zero_bed = (
                        (raw_prop in {"flat", "apartment", "studio"})
                        and bed_val is not None
                        and int(round(bed_val)) == 0
                    )
                    if not (is_raw_studio or is_flat_zero_bed):
                        opt_fail.append("layout_tag 'studio' not matched")
                rent_val = _to_float(r.get("price_pcm"))
                eff_rent_req = req_rent if req_rent is not None else c.get("max_rent_pcm")
                if eff_rent_req is not None and rent_val is not None:
                    try:
                        if float(rent_val) > float(eff_rent_req):
                            opt_fail.append(f"price {rent_val:g} > {float(eff_rent_req):g}")
                    except Exception:
                        pass

                passed = len(opt_fail) == 0
                any_pass = any_pass or passed
                option_audits.append(
                    {
                        "required": {
                            "bedrooms": req_bed,
                            "bathrooms": req_bath,
                            "property_type": req_prop or None,
                            "layout_tag": req_tag or None,
                            "max_rent_pcm": eff_rent_req,
                        },
                        "actual": {
                            "bedrooms": bed_val,
                            "bathrooms": bath_val,
                            "property_type": prop_val or None,
                            "price_pcm": rent_val,
                        },
                        "pass": passed,
                        "fail_reasons": opt_fail,
                    }
                )

            checks["layout_options"] = {
                "active": True,
                "option_count": len(option_audits),
                "pass": any_pass,
                "options": option_audits,
            }
            if not any_pass:
                reasons.append("layout_options no option matched")

        rent_req = c.get("max_rent_pcm")
        has_layout_budget = any(
            isinstance(x, dict) and x.get("max_rent_pcm") is not None
            for x in (layout_options or [])
        )
        if rent_req is not None and not use_layout_options and not has_layout_budget:
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
        unknown_item_set = set()

        def _add_unknown(field_key: str) -> None:
            nonlocal unknown_penalty_raw
            if field_key in unknown_item_set:
                return
            w = float(UNKNOWN_PENALTY_WEIGHTS.get(field_key, 0.0))
            if w <= 0.0:
                return
            unknown_item_set.add(field_key)
            unknown_items.append(field_key)
            unknown_penalty_raw += w

        # Penalize unknown values (e.g. "Ask agent") on active hard constraints.
        if hard.get("max_rent_pcm") is not None and _to_float(r.get("price_pcm")) is None:
            _add_unknown("price")
        layout_opts = hard.get("layout_options") or []
        requires_price = any(isinstance(x, dict) and x.get("max_rent_pcm") is not None for x in layout_opts)
        requires_bed = any(isinstance(x, dict) and x.get("bedrooms") is not None for x in layout_opts)
        requires_bath = any(isinstance(x, dict) and x.get("bathrooms") is not None for x in layout_opts)
        requires_prop = any(
            isinstance(x, dict) and _safe_text(x.get("property_type")).strip()
            for x in layout_opts
        )
        if requires_price and _to_float(r.get("price_pcm")) is None:
            _add_unknown("price")
        if requires_bed and _to_float(r.get("bedrooms")) is None:
            _add_unknown("bedrooms")
        if requires_bath and _to_float(r.get("bathrooms")) is None:
            _add_unknown("bathrooms")
        if requires_prop and not _safe_text(r.get("property_type")).strip():
            _add_unknown("property_type")
        if hard.get("available_from") is not None and pd.isna(pd.to_datetime(r.get("available_from"), errors="coerce")):
            _add_unknown("available_from")

        furnish_req = _norm_furnish_value(hard.get("furnish_type"))
        if furnish_req:
            furn_val = _norm_furnish_value(r.get("furnish_type"))
            if not furn_val or furn_val == "ask agent":
                _add_unknown("furnish_type")

        if _safe_text(hard.get("let_type")).strip() and not _safe_text(r.get("let_type")).strip():
            _add_unknown("let_type")

        if hard.get("min_tenancy_months") is not None:
            tenancy_txt = _safe_text(r.get("min_tenancy")).lower()
            if not re.search(r"(\d+(?:\.\d+)?)", tenancy_txt):
                _add_unknown("min_tenancy_months")

        if hard.get("min_size_sqm") is not None:
            if _to_float(r.get("size_sqm")) is None and _to_float(r.get("size_sqft")) is None:
                _add_unknown("min_size_sqm")

        if unknown_penalty_raw > 0.0:
            unknown_penalty = min(float(UNKNOWN_PENALTY_CAP), float(unknown_penalty_raw))
            penalty_score += unknown_penalty
            penalties.append(
                f"unknown_hard({','.join(unknown_items)};+{unknown_penalty:.2f})"
            )

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
        ["final_score", "location_hit_count", "qdrant_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return out, weights


STRUCTURED_FIELDS = [
    "max_rent_pcm",
    "available_from",
    "furnish_type",
    "let_type",
    "layout_options",
    "min_tenancy_months",
    "min_size_sqm",
    "location_keywords",
    "k",
]
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
    append_jsonl(STRUCTURED_CONFLICT_LOG_PATH, rec, "structured conflict log")


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
        append_jsonl(STRUCTURED_TRAINING_LOG_PATH, rec, "structured training samples")

def format_listing_row_debug(r: Dict[str, Any], i: int) -> str:
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


def format_listing_row_summary(r: Dict[str, Any], i: int) -> str:
    title = _safe_text(r.get("title")) or "(no title)"
    url = _safe_text(r.get("url"))
    address = _safe_text(r.get("address"))
    price = _to_float(r.get("price_pcm"))
    beds = _to_float(r.get("bedrooms"))
    baths = _to_float(r.get("bathrooms"))
    final_score = _to_float(r.get("final_score"))

    transit_hits = [x.strip() for x in _safe_text(r.get("transit_hits")).split(",") if x.strip()]
    school_hits = [x.strip() for x in _safe_text(r.get("school_hits")).split(",") if x.strip()]
    pref_hits = [x.strip() for x in _safe_text(r.get("preference_hits")).split(",") if x.strip()]
    hit_terms = (pref_hits + transit_hits + school_hits)[:2]
    penalty_reasons = [x.strip() for x in _safe_text(r.get("penalty_reasons")).split(",") if x.strip()]

    parts: List[str] = []
    parts.append(f"{i}. {title}")
    line2: List[str] = []
    if price is not None:
        line2.append(f"£{int(round(price))}/pcm")
    if beds is not None:
        line2.append(f"{int(round(beds))} bed")
    if baths is not None:
        line2.append(f"{int(round(baths))} bath")
    if address:
        line2.append(address)
    if line2:
        parts.append("   " + " | ".join(line2))
    if final_score is not None:
        parts.append(f"   Final score: {final_score:.4f}")
    if hit_terms:
        parts.append("   Because matched: " + ", ".join(hit_terms))
    if penalty_reasons:
        parts.append("   Because penalized: " + penalty_reasons[0])
    if url:
        parts.append("   " + url)
    return "\n".join(parts)


def format_listing_row(r: Dict[str, Any], i: int, view_mode: str = "summary") -> str:
    if str(view_mode).strip().lower() == "debug":
        return format_listing_row_debug(r, i)
    return format_listing_row_summary(r, i)


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
    qdrant_client = load_stage_a_resources()
    embedder = SentenceTransformer(EMBED_MODEL)

    state = {
        "history": [],   # list of (user, assistant_text) for your own future use
        "k": DEFAULT_K,
        "recall": DEFAULT_RECALL,
        "last_query": None,
        "last_df": None,
        "constraints": None,
        "view_mode": "summary",
    }

    def stage_note(stage: str, detail: str) -> None:
        print(f"\nBot> [{stage}] {detail}")

    print("RentBot (minimal retrieval)")
    print("Commands: /exit /reset /k N /show /recall N /constraints /model /view summary|debug")
    print(f"StageA backend: qdrant")
    print(f"Qdrant path   : {QDRANT_LOCAL_PATH}")
    print(f"Collection    : {QDRANT_COLLECTION}")
    print(f"Qdrant prefilter: {QDRANT_ENABLE_PREFILTER}")
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
                # Keep constraints.k in sync so downstream uses the updated k.
                if state["constraints"] is None:
                    state["constraints"] = {
                        "k": n,
                        "location_keywords": [],
                        "max_rent_pcm": None,
                        "available_from": None,
                        "available_from_op": None,
                        "furnish_type": None,
                        "let_type": None,
                        "layout_options": [],
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
                print(format_listing_row(r.to_dict(), i + 1, view_mode=state.get("view_mode", "summary")))
            continue
        if cmd == "/view":
            mode = str(arg or "").strip().lower()
            if mode not in {"summary", "debug"}:
                print("Usage: /view summary   or   /view debug")
                continue
            state["view_mode"] = mode
            print(f"OK. view = {mode}")
            continue
        if cmd == "/constraints":
            print(json.dumps(state.get("constraints") or {}, ensure_ascii=False, indent=2))
            continue
        
        if cmd == "/model":
            print(f"QWEN_BASE_URL={QWEN_BASE_URL}")
            print(f"QWEN_MODEL={QWEN_MODEL}")
            print(f"RENT_STRUCTURED_POLICY={STRUCTURED_POLICY}")
            print("RENT_STAGEA_BACKEND=qdrant (fixed)")
            print(f"RENT_QDRANT_ENABLE_PREFILTER={QDRANT_ENABLE_PREFILTER}")
            print(f"RENT_LOG_LEVEL={LOG_LEVEL}")
            print(f"RENT_RANKING_LOG_DETAIL={RANKING_LOG_DETAIL}")
            continue
        # normal query
        # query = user_in
        # k = int(state["k"])
        # recall = int(state["recall"])
        prev_constraints = dict(state["constraints"] or {})
        stage_note("Pre Stage A", "Parsing input and extracting/repairing constraints and preference signals")
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
        stage_note("Pre Stage A", f"Because the input changed constraints, state was updated to: {changes_line}")
        stage_note("Pre Stage A", f"Because retrieval/filtering depends on active constraints, current constraints: {json.dumps(active_line, ensure_ascii=False)}")
        c = state["constraints"] or {}
        signals = split_query_signals(
            user_in,
            c,
            precomputed_semantic_terms=semantic_terms,
            semantic_parse_source=semantic_parse_source,
        )

        log_message("INFO", f"state changes: {changes_line}")
        log_message("INFO", f"state active_constraints: {json.dumps(active_line, ensure_ascii=False)}")
        conflict_count = int(structured_audit.get("conflict_count", 0))
        if conflict_count > 0:
            log_message(
                "INFO",
                f"state structured_conflicts: policy={STRUCTURED_POLICY}, "
                f"count={conflict_count}, agreement_rate={float(structured_audit.get('agreement_rate', 1.0)):.3f}",
            )
        if VERBOSE_STATE_LOG:
            log_message("DEBUG", f"state verbose llm_constraints: {json.dumps(llm_extracted, ensure_ascii=False)}")
            log_message("DEBUG", f"state verbose rule_constraints: {json.dumps(rule_extracted, ensure_ascii=False)}")
            log_message("DEBUG", f"state verbose selected_constraints: {json.dumps(extracted, ensure_ascii=False)}")
            log_message("DEBUG", f"state verbose llm_semantic_terms: {json.dumps(semantic_terms, ensure_ascii=False)}")
            log_message("DEBUG", f"state verbose signals: {json.dumps(signals, ensure_ascii=False)}")

        k = int(c.get("k", DEFAULT_K) or DEFAULT_K)
        recall = int(state["recall"])
        query = build_stage_a_query(signals, user_in)

        # Stage A: recall pool
        stage_note("Stage A", f"Because we need a broad candidate pool first, running vector recall (recall={recall})")
        stage_a_df = stage_a_search(qdrant_client, embedder, query=query, recall=recall, c=c)
        stage_note("Stage A", f"Because recall finished, got {len(stage_a_df)} candidates")
        stage_a_records = []
        if stage_a_df is not None and len(stage_a_df) > 0:
            for i, row in stage_a_df.reset_index(drop=True).iterrows():
                rec = candidate_snapshot(row.to_dict())
                rec["rank"] = i + 1
                rec["score"] = rec.get("qdrant_score")
                rec["score_formula"] = "score = qdrant_cosine_similarity(query_A, listing_embedding)"
                stage_a_records.append(rec)

        # Stage B: hard filters (audit all candidates)
        stage_note("Stage B", "Because these are hard constraints, applying hard filters (budget/layout/move-in, etc.)")
        filtered, hard_audits = apply_hard_filters_with_audit(stage_a_df, c)
        stage_b_pass_records = [x for x in hard_audits if x.get("hard_pass")]
        fail_counter: Dict[str, int] = {}
        for rec in hard_audits:
            if rec.get("hard_pass"):
                continue
            reasons = rec.get("hard_fail_reasons") or []
            if not reasons:
                continue
            key = str(reasons[0]).split(" ", 1)[0]
            fail_counter[key] = fail_counter.get(key, 0) + 1
        fail_brief = ", ".join([f"{k}:{v}" for k, v in sorted(fail_counter.items(), key=lambda x: x[1], reverse=True)[:3]])
        if fail_brief:
            stage_note("Stage B", f"Because of hard filtering, result is pass={len(filtered)}/{len(stage_a_df)}; top eliminations: {fail_brief}")
        else:
            stage_note("Stage B", f"Because of hard filtering, result is pass={len(filtered)}/{len(stage_a_df)}")

        # Stage C: rerank only on topic/preference scores (qdrant score as tie-break)
        pref_terms_all = (
            list(signals.get("topic_preferences", {}).get("transit_terms", []) or [])
            + list(signals.get("topic_preferences", {}).get("school_terms", []) or [])
            + list(signals.get("general_semantic", []) or [])
        )
        pref_preview = ", ".join([str(x) for x in pref_terms_all[:3]]) if pref_terms_all else "no explicit preference"
        stage_note("Stage C", f"Because preference signals are [{pref_preview}], running soft rerank and unknown-pass penalties")
        ranked, stage_c_weights = rank_stage_c(filtered, signals, embedder=embedder)
        stage_note("Stage C", f"Because reranking finished, ranked={len(ranked)}; weights={json.dumps(stage_c_weights, ensure_ascii=False)}")
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

        log_message(
            "INFO",
            f"pipeline counts: stageA={len(stage_a_df)} stageB={len(filtered)} "
            f"stageC={len(ranked)} k={k} recall={recall}",
        )
        stage_d_payload: Optional[Dict[str, Any]] = None
        stage_d_output: str = ""
        stage_d_raw_output: str = ""
        stage_d_error: str = ""

        if len(filtered) < k:
            print("\nBot> Not enough listings pass current hard constraints (price/bedrooms/bathrooms/move-in/furnishing/tenancy/size). You can relax budget or update constraints.")
            df = ranked.reset_index(drop=True)
        else:
            df = ranked.head(k).reset_index(drop=True)

        if df is not None and len(df) > 0:
            stage_note("Stage D", "Because explainability is required, building evidence and generating grounded explanation")
            df = df.copy()
            df["evidence"] = df.apply(lambda row: build_evidence_for_row(row.to_dict(), c, user_in), axis=1)


        
        if df is None or len(df) == 0:
            append_ranking_log_entry(RANKING_LOG_PATH, 
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
            lines.append(format_listing_row(r.to_dict(), i + 1, view_mode=state.get("view_mode", "summary")))
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
                if state.get("view_mode", "summary") == "debug":
                    lines.append(grounded_out)
                else:
                    short_lines = [x for x in grounded_out.splitlines() if x.strip()][:5]
                    lines.append("\n".join(short_lines))
            if state.get("view_mode", "summary") == "debug":
                ev_txt = format_grounded_evidence(df=df, max_items=min(8, len(df)))
                if ev_txt:
                    lines.append("")
                    lines.append("Grounded evidence:")
                    lines.append(ev_txt)
        except Exception as e:
            stage_d_error = str(e)
            lines.append("")
            lines.append(f"[warn] grounded explanation unavailable: {e}")
        append_ranking_log_entry(RANKING_LOG_PATH, 
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
