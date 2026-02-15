import json
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

from chatbot_config import (
    EXTRACT_ALL_SYSTEM,
    EXTRACT_SYSTEM,
    GROUNDED_EXPLAIN_SYSTEM,
    QWEN_API_KEY,
    QWEN_BASE_URL,
    QWEN_MODEL,
)
from settings import DEFAULT_K
from helpers import (
    _extract_json_obj,
    _normalize_constraint_extract,
    _normalize_semantic_extract,
    _safe_text,
    _to_float,
)
from log import log_message

qwen_client = OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)


def _structured_debug_enabled() -> bool:
    return str(os.environ.get("RENT_STRUCTURED_DEBUG_PRINT", "0")).strip().lower() in {"1", "true", "yes", "on"}


def qwen_chat(messages, temperature=0.0) -> str:
    r = qwen_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return r.choices[0].message.content.strip()


def llm_extract(user_text: str, existing_constraints: Optional[dict]) -> dict:
    prefix = ""
    if existing_constraints:
        prefix = "Existing constraints (JSON):\n" + json.dumps(existing_constraints, ensure_ascii=False) + "\n\n"

    txt = qwen_chat(
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": prefix + "User says:\n" + user_text},
        ],
        temperature=0.0,
    )
    if _structured_debug_enabled():
        log_message("INFO", "llm_extract_raw " + txt)
    obj = _extract_json_obj(txt)
    return _normalize_constraint_extract(obj)


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
    if _structured_debug_enabled():
        log_message("INFO", "llm_extract_all_raw " + txt)
    obj = _extract_json_obj(txt)
    constraints = _normalize_constraint_extract(obj.get("constraints") or {})
    semantic_terms = _normalize_semantic_extract(obj.get("semantic_terms") or {})
    return {
        "constraints": constraints,
        "semantic_terms": semantic_terms,
    }


def _parse_feature_items(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = _safe_text(raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    return [s]


def _extract_ask_agent_items(r: Dict[str, Any]) -> List[str]:
    direct = r.get("ask_agent_items")
    items: List[str] = []
    if isinstance(direct, list):
        items.extend([str(x).strip() for x in direct if str(x).strip()])
    elif isinstance(direct, str):
        s = direct.strip()
        if s:
            if s.startswith("[") and s.endswith("]"):
                try:
                    obj = json.loads(s)
                    if isinstance(obj, list):
                        items.extend([str(x).strip() for x in obj if str(x).strip()])
                    elif isinstance(obj, dict):
                        for k, v in obj.items():
                            if str(v).strip():
                                items.append(str(k).strip())
                except Exception:
                    items.extend([x.strip() for x in s.split("|") if x.strip()])
            else:
                items.extend([x.strip() for x in s.split("|") if x.strip()])
    elif isinstance(direct, dict):
        for k, v in direct.items():
            if str(v).strip():
                items.append(str(k).strip())

    unresolved_keys = [
        "deposit",
        "available_from",
        "min_tenancy",
        "let_type",
        "furnish_type",
        "council_tax",
        "property_type",
        "size_sqft",
        "size_sqm",
        "bedrooms",
        "bathrooms",
    ]
    for key in unresolved_keys:
        val = _safe_text(r.get(key)).strip().lower()
        if val in {"ask agent", "unknown", "n/a", "na", "not provided", "not known"}:
            items.append(key)

    seen = set()
    deduped: List[str] = []
    for x in items:
        k = str(x).strip()
        if not k:
            continue
        kl = k.lower()
        if kl in seen:
            continue
        seen.add(kl)
        deduped.append(k)
    return deduped


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
            listing_id = _safe_text(r.get("listing_id"))
            if not listing_id:
                listing_id = _safe_text(r.get("url")) or f"row_{i+1}"
            rows.append(
                {
                    "rank": int(i + 1),
                    "listing_id": listing_id,
                    "title": _safe_text(r.get("title")),
                    "url": _safe_text(r.get("url")),
                    "monthly_rent": _to_float(r.get("price_pcm")),
                    "deposit": _safe_text(r.get("deposit")),
                    "features": _parse_feature_items(r.get("features")),
                    "description": _safe_text(r.get("description")),
                    "ask_agent_items": _extract_ask_agent_items(r),
                }
            )
    hard = signals.get("hard_constraints", {}) if isinstance(signals, dict) else {}
    query_obj = {
        "raw_query": user_query,
        "constraints": {
            "max_rent_pcm": hard.get("max_rent_pcm"),
            "layout_options": hard.get("layout_options"),
            "location_keywords": hard.get("location_keywords"),
            "available_from": hard.get("available_from"),
            "furnish_type": hard.get("furnish_type"),
            "let_type": hard.get("let_type"),
            "min_tenancy_months": hard.get("min_tenancy_months"),
            "min_size_sqm": hard.get("min_size_sqm"),
            "min_size_sqft": hard.get("min_size_sqft"),
        },
    }
    return {
        "user_query": query_obj,
        "top_k_candidates": rows,
        "top_k": int(c.get("k", DEFAULT_K) or DEFAULT_K),
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
                    "Generate Stage D recommendation summary from this JSON only:\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ],
        temperature=0.1,
    )
    out = txt.strip()
    try:
        parsed = _extract_json_obj(out)
        out = json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return out, payload, txt


def _fmt_listing_line_from_df(rank: int, row: Dict[str, Any]) -> List[str]:
    title = _safe_text(row.get("title")) or "(no title)"
    price = _to_float(row.get("price_pcm"))
    beds = _to_float(row.get("bedrooms"))
    baths = _to_float(row.get("bathrooms"))
    address = _safe_text(row.get("address"))
    url = _safe_text(row.get("url"))

    line2: List[str] = []
    if price is not None:
        line2.append(f"£{int(round(price))}/pcm")
    if beds is not None:
        line2.append(f"{int(round(beds))} bed")
    if baths is not None:
        line2.append(f"{int(round(baths))} bath")
    if address:
        line2.append(address)

    out = [f"{rank}. {title}"]
    if line2:
        out.append("   " + " | ".join(line2))
    if url:
        out.append("   " + url)
    return out


def _normalize_risk_order(risks: List[Any]) -> List[str]:
    items = [_safe_text(x) for x in (risks or []) if _safe_text(x)]
    if not items:
        return []

    def _is_deposit_risk(s: str) -> bool:
        t = s.lower()
        return "deposit" in t or "upfront" in t

    deposit_items = [x for x in items if _is_deposit_risk(x)]
    non_deposit_items = [x for x in items if not _is_deposit_risk(x)]
    ordered = non_deposit_items + deposit_items
    dedup: List[str] = []
    seen = set()
    for x in ordered:
        k = x.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        dedup.append(x)
    return dedup[:3]


def _infer_unknown_risks_from_row(row: Dict[str, Any], max_items: int = 2) -> List[str]:
    listing_key = _safe_text(row.get("listing_id")) or _safe_text(row.get("url")) or _safe_text(row.get("title"))
    rules = [
        ("bedrooms", "bedrooms", 4, "HIGH: Bedroom count is not confirmed. Core layout fit must be verified with agent."),
        ("bathrooms", "bathrooms", 4, "HIGH: Bathroom count is not confirmed. Core layout fit must be verified with agent."),
        ("availability_date", "available_from", 3, "HIGH: Move-in date is not confirmed. Start date risk may block planning."),
        ("lease_term_options", "min_tenancy", 3, "HIGH: Lease term details are unclear. Contract fit is not verifiable."),
        ("security_deposit_rule", "deposit", 3, "HIGH: Deposit rules are unclear. Refund/deduction risk remains unknown."),
        ("utilities_included", "council_tax", 2, "MEDIUM: Utility or council tax scope is unclear. Monthly total cost may change."),
        ("furnished_or_appliance_details", "furnish_type", 2, "MEDIUM: Furnishing details are unclear. Move-in readiness is uncertain."),
        ("size", "size_sqm", 1, "LOW: Unit size is not explicit. Space fit to query is uncertain."),
        ("property_type", "property_type", 1, "LOW: Property-type details need confirmation. Layout expectations may differ."),
    ]
    unknown_vals = {"", "ask agent", "unknown", "n/a", "na", "not provided", "not known"}
    candidates: List[Tuple[int, str]] = []
    for key, field, priority, msg in rules:
        if key == "size":
            raw_sqm = _safe_text(row.get("size_sqm")).strip().lower()
            raw_sqft = _safe_text(row.get("size_sqft")).strip().lower()
            if raw_sqm in unknown_vals and raw_sqft in unknown_vals:
                candidates.append((priority, msg))
            continue
        raw = _safe_text(row.get(field)).strip().lower()
        if raw in unknown_vals:
            candidates.append((priority, msg))

    if not candidates:
        return []

    # Important items first; for same priority, rotate by listing hash for slight variety.
    candidates.sort(key=lambda x: x[0], reverse=True)
    groups: Dict[int, List[str]] = {}
    for p, m in candidates:
        groups.setdefault(int(p), []).append(m)
    salt = listing_key or "listing"
    seed = int(hashlib.md5(salt.encode("utf-8")).hexdigest(), 16)
    msgs: List[str] = []
    for p in sorted(groups.keys(), reverse=True):
        bucket = groups[p]
        if len(bucket) > 1:
            off = seed % len(bucket)
            bucket = bucket[off:] + bucket[:off]
        msgs.extend(bucket)

    out: List[str] = []
    seen = set()
    for msg in msgs:
        k = msg.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(msg)
        if len(out) >= max_items:
            break
    return out


def _is_unknown_value(v: Any) -> bool:
    s = _safe_text(v).strip().lower()
    return s in {"", "ask agent", "unknown", "n/a", "na", "not provided", "not known"}


def _build_combined_quality_risk(row: Dict[str, Any], threshold: float = 0.60) -> Optional[str]:
    missing_core: List[str] = []
    if _is_unknown_value(row.get("bedrooms")):
        missing_core.append("bedroom count")
    if _is_unknown_value(row.get("bathrooms")):
        missing_core.append("bathroom count")
    if _is_unknown_value(row.get("available_from")):
        missing_core.append("move-in date")
    if _is_unknown_value(row.get("min_tenancy")):
        missing_core.append("lease term")

    sim = _extract_max_soft_similarity(row)
    soft_weak = sim is not None and sim < float(threshold)
    if not missing_core and not soft_weak:
        return None

    if missing_core and soft_weak:
        return (
            f"HIGH: Key fit fields need confirmation ({', '.join(missing_core)}), and soft-preference match is weak "
            f"(similarity {sim:.2f} < {threshold:.2f}). Confirm suitability with agent before final shortlist."
        )
    if missing_core:
        return (
            f"HIGH: Key fit fields need confirmation ({', '.join(missing_core)}). "
            "Core suitability is uncertain until agent confirmation."
        )
    return (
        f"MEDIUM: Soft-preference match is weak (similarity {sim:.2f} < {threshold:.2f}). "
        "Confirm priority preferences with agent."
    )


def _extract_max_soft_similarity(row: Dict[str, Any]) -> Optional[float]:
    raw = row.get("preference_evidence")
    items: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        items = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, str):
        s = raw.strip()
        if s:
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    items = [x for x in obj if isinstance(x, dict)]
            except Exception:
                items = []
    sims: List[float] = []
    for x in items:
        sv = _to_float(x.get("sim"))
        if sv is not None:
            sims.append(float(sv))
    if sims:
        return max(sims)
    ps = _to_float(row.get("preference_score"))
    if ps is not None:
        return float(ps)
    return None


def _infer_soft_preference_risk(row: Dict[str, Any], threshold: float = 0.60) -> Optional[str]:
    detail = _safe_text(row.get("preference_detail")).lower()
    source = _safe_text(row.get("preference_source")).lower()
    hits = _safe_text(row.get("preference_hits")).strip()
    if "no_intents" in detail or source == "no_intents":
        return None
    sim = _extract_max_soft_similarity(row)
    if sim is None:
        if hits:
            return "MEDIUM: Soft-preference evidence is incomplete. Match quality needs agent confirmation."
        return None
    if sim < float(threshold):
        return (
            f"MEDIUM: Soft-preference match is weak (similarity {sim:.2f} < {threshold:.2f}). "
            "Confirm priority preferences with agent."
        )
    return None


def render_stage_d_for_user(stage_d_text: str, df: Optional[pd.DataFrame] = None, max_items: int = 8) -> str:
    s = _safe_text(stage_d_text).strip()
    if not s:
        return ""
    try:
        obj = _extract_json_obj(s)
    except Exception:
        return s
    recs = obj.get("recommendations")
    if not isinstance(recs, list) or not recs:
        return s

    lines: List[str] = []
    top_k = obj.get("top_k")
    header = f"Recommended Listings (Top {top_k})" if isinstance(top_k, int) and top_k > 0 else "Recommended Listings"
    lines.append(header)
    lines.append("")

    shown = 0
    rank_to_row: Dict[int, Dict[str, Any]] = {}
    if df is not None and len(df) > 0:
        for i, row in df.reset_index(drop=True).iterrows():
            rank_to_row[int(i + 1)] = row.to_dict()
    for item in recs:
        if not isinstance(item, dict):
            continue
        if shown >= max_items:
            break
        rank_raw = item.get("rank")
        rank = int(rank_raw) if isinstance(rank_raw, (int, float)) else shown + 1
        reason = _safe_text(item.get("summary_reason"))
        dep_risk = _safe_text(item.get("deposit_risk_level"))
        risks_raw = item.get("risk_flags") if isinstance(item.get("risk_flags"), list) else []
        risks = _normalize_risk_order(risks_raw)
        highlights = item.get("highlights") if isinstance(item.get("highlights"), list) else []

        row = rank_to_row.get(rank)
        if row:
            lines.extend(_fmt_listing_line_from_df(rank, row))
            combined_quality_risk = _build_combined_quality_risk(row, threshold=0.60)
            if combined_quality_risk:
                risks = _normalize_risk_order([combined_quality_risk] + risks)
            if len(risks) < 2:
                unknown_extra = _infer_unknown_risks_from_row(row, max_items=max(0, 2 - len(risks)))
                risks = _normalize_risk_order(risks + unknown_extra)
        else:
            title = _safe_text(item.get("title")) or "(no title)"
            lines.append(f"{rank}. {title}")

        if reason:
            lines.append(reason)
        if highlights:
            lines.append("Highlights:")
            for h in highlights[:3]:
                if not isinstance(h, dict):
                    continue
                claim = _safe_text(h.get("claim"))
                ev = _safe_text(h.get("evidence"))
                if claim and ev:
                    lines.append(f"- {claim} (evidence: \"{ev}\")")
                elif claim:
                    lines.append(f"- {claim}")
        if risks:
            lines.append("Risks:")
            for rf in risks[:3]:
                txt = _safe_text(rf)
                if txt:
                    lines.append(f"- {txt}")
        elif dep_risk:
            lines.append("Risks:")
            lines.append(f"- {dep_risk}: Deposit-related information needs confirmation.")
        lines.append("")
        shown += 1
    return "\n".join(lines).strip()


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
            lines.append(f"  compare: '{intent}' vs '{text[:140]}'")
            lines.append(f"  score: sim={sim_txt} | field={field}")
    return "\n".join(lines)
