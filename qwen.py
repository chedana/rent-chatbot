import json
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

qwen_client = OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)


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
