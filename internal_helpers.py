import re
from typing import Any, Dict, List, Tuple

import numpy as np

from helpers import _safe_text, parse_jsonish_items
from settings import (
    BATCH,
    INTENT_EVIDENCE_TOP_N,
    INTENT_HIT_THRESHOLD,
    SEMANTIC_FIELD_WEIGHTS,
)


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
    embedder,
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


def _sim_text(query: str, text: str, embedder, cache: Dict[str, np.ndarray]) -> float:
    q = _safe_text(query).lower()
    t = _safe_text(text).lower()
    if not q or not t:
        return 0.0
    if q in t:
        return 1.0
    qv, tv = _embed_texts_cached(embedder, [q, t], cache)
    cos = float(np.dot(qv, tv))
    return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))


def _collect_value_candidates(r: Dict[str, Any]) -> List[Dict[str, str]]:
    cands: List[Dict[str, str]] = []
    for v in parse_jsonish_items(r.get("schools")):
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
    embedder,
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
    top = scored[: max(1, top_k)]
    if not top:
        return 0.0, f"intent='{intent}' no_candidates", []
    num = sum(w * sim for _, w, sim, _, _ in top)
    den = sum(w for _, w, _, _, _ in top)
    score = (num / den) if den > 0 else 0.0
    top_show = []
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
    embedder,
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
            for item in top_struct[: max(1, INTENT_EVIDENCE_TOP_N)]:
                selected_evidence.append(
                    {
                        "intent": str(it),
                        "intent_score": float(sc),
                        **item,
                    }
                )
    group_score = float(sum(scores) / max(1, len(scores)))
    return group_score, hit_terms, " | ".join(details), selected_evidence


HIGH_RISK_STRUCTURED_FIELDS = {
    "max_rent_pcm",
    "bedrooms",
    "bedrooms_op",
    "bathrooms",
    "bathrooms_op",
    "available_from",
    "let_type",
    "property_type",
    "layout_options",
    "min_tenancy_months",
    "min_size_sqm",
}


def _choose_structured_value(policy: str, field: str, llm_v: Any, rule_v: Any) -> Tuple[Any, str]:
    from helpers import _canon_for_structured_compare

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

    if high_risk:
        if rule_v is not None:
            return rule_v, "override_with_rule_high_risk"
        return llm_v, "fallback_llm_high_risk"

    if llm_v is not None:
        return llm_v, "prefer_llm_low_risk"
    if rule_v is not None:
        return rule_v, "fill_from_rule_low_risk"
    return None, "both_none"
