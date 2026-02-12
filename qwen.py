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
            lines.append(f"  compare: '{intent}' vs '{text[:140]}'")
            lines.append(f"  score: sim={sim_txt} | field={field}")
    return "\n".join(lines)
