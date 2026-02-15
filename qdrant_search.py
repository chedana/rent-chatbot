import json
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from qdrant_client import QdrantClient, models
except Exception:
    QdrantClient = None
    models = None

from helpers import _safe_text, expand_location_keyword_candidates
from log import log_message
from settings import (
    QDRANT_COLLECTION,
    QDRANT_ENABLE_PREFILTER,
    QDRANT_LOCAL_PATH,
    STAGEA_TRACE,
)


def load_qdrant_client() -> QdrantClient:
    if QdrantClient is None or models is None:
        raise ImportError("qdrant-client is not installed. Please run: pip install qdrant-client")
    client = QdrantClient(path=QDRANT_LOCAL_PATH)
    if not client.collection_exists(QDRANT_COLLECTION):
        raise FileNotFoundError(
            f"Missing Qdrant collection: {QDRANT_COLLECTION} (path={QDRANT_LOCAL_PATH})"
        )
    info = client.get_collection(QDRANT_COLLECTION)
    log_message("INFO", f"boot qdrant collection={QDRANT_COLLECTION}, points={info.points_count}")
    return client


def load_stage_a_resources():
    return load_qdrant_client()


def embed_query(embedder, q: str) -> np.ndarray:
    x = embedder.encode(
        [q],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x = x / norms
    return x


def qdrant_search(
    client: QdrantClient,
    embedder,
    query: str,
    recall: int,
    c: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    if models is None:
        raise ImportError("qdrant-client models are unavailable. Please run: pip install qdrant-client")

    trace_info: Dict[str, Any] = {
        "location_keywords": [],
        "location_keyword_expansions": {},
        "location_tokens": [],
        "prefilter_count": None,
    }

    def _count_prefilter_candidates(qfilter: Optional["models.Filter"]) -> Optional[int]:
        if qfilter is None:
            return None
        try:
            # Fast path: qdrant count API.
            resp = client.count(
                collection_name=QDRANT_COLLECTION,
                count_filter=qfilter,
                exact=True,
            )
            n = getattr(resp, "count", None)
            return int(n) if n is not None else None
        except Exception:
            # Fallback for older qdrant-client versions: scroll and count.
            try:
                total = 0
                offset = None
                while True:
                    points, offset = client.scroll(
                        collection_name=QDRANT_COLLECTION,
                        scroll_filter=qfilter,
                        limit=512,
                        offset=offset,
                        with_payload=False,
                        with_vectors=False,
                    )
                    total += len(points or [])
                    if offset is None:
                        break
                return int(total)
            except Exception:
                # Still unavailable on this backend/version.
                return None

    def _build_qdrant_filter(c: Optional[Dict[str, Any]]) -> Optional["models.Filter"]:
        c = c or {}
        must: List[Any] = []

        loc_values: List[str] = []
        station_values: List[str] = []
        region_values: List[str] = []
        postcode_values: List[str] = []
        loc_keywords = [str(x).strip() for x in (c.get("location_keywords") or []) if str(x).strip()]
        trace_info["location_keywords"] = loc_keywords
        for term in loc_keywords:
            expanded = expand_location_keyword_candidates(term, limit=20, min_score=0.80)
            expanded_terms = [term] + [x for x in expanded if _safe_text(x)]
            # Preserve order while removing duplicates.
            seen_local = set()
            uniq_terms: List[str] = []
            for x in expanded_terms:
                k = _safe_text(x).lower().strip()
                if not k or k in seen_local:
                    continue
                seen_local.add(k)
                uniq_terms.append(k)
            trace_info["location_keyword_expansions"][_safe_text(term)] = uniq_terms

            for raw in uniq_terms:
                raw = re.sub(r"\s+", " ", raw).strip()
                slug = re.sub(r"[^a-z0-9]+", "_", raw)
                slug = re.sub(r"_+", "_", slug).strip("_")
                for m in re.findall(r"\b[a-z]{1,2}\d[a-z0-9]?\s?\d[a-z]{2}\b", raw):
                    postcode_values.append(m.replace(" ", ""))
                if raw:
                    loc_values.append(raw)
                    if " " in raw:
                        loc_values.append(raw.replace(" ", ""))
                    station_values.append(raw)
                if slug:
                    loc_values.append(slug)
                    loc_values.append(f"{slug}_london")
                    if not slug.endswith("_station"):
                        loc_values.append(f"{slug}_station")
                    station_values.append(slug)
                    if not slug.endswith("_station"):
                        station_values.append(f"{slug}_station")
                    region_values.append(slug)
                    region_values.append(f"{slug}_london")

        loc_values = list(dict.fromkeys([x for x in loc_values if x]))
        station_values = list(dict.fromkeys([x for x in station_values if x]))
        region_values = list(dict.fromkeys([x for x in region_values if x]))
        postcode_values = list(dict.fromkeys([x for x in postcode_values if x]))
        trace_info["location_tokens"] = loc_values
        trace_info["location_station_tokens"] = station_values
        trace_info["location_region_tokens"] = region_values
        trace_info["location_postcode_tokens"] = postcode_values
        should_conditions: List[Any] = []
        if postcode_values:
            should_conditions.append(
                models.FieldCondition(
                    key="location_postcode_tokens",
                    match=models.MatchAny(any=postcode_values),
                )
            )
        if station_values:
            should_conditions.append(
                models.FieldCondition(
                    key="location_station_tokens",
                    match=models.MatchAny(any=station_values),
                )
            )
        if region_values:
            should_conditions.append(
                models.FieldCondition(
                    key="location_region_tokens",
                    match=models.MatchAny(any=region_values),
                )
            )
        # Backward-compatible fallback for older indexes.
        if loc_values:
            should_conditions.append(
                models.FieldCondition(
                    key="location_tokens",
                    match=models.MatchAny(any=loc_values),
                )
            )
        if should_conditions:
            must.append(models.Filter(should=should_conditions))

        if not must:
            return None
        return models.Filter(must=must)

    qx = embed_query(embedder, query)[0].tolist()
    qfilter = _build_qdrant_filter(c) if QDRANT_ENABLE_PREFILTER else None
    prefilter_count = _count_prefilter_candidates(qfilter)
    trace_info["prefilter_count"] = prefilter_count
    if STAGEA_TRACE:
        log_message("DEBUG", f"stageA backend=qdrant recall={recall} prefilter={QDRANT_ENABLE_PREFILTER}")
        log_message("DEBUG", f"stageA query={query}")
        if prefilter_count is not None:
            log_message("DEBUG", f"stageA prefilter_count={prefilter_count}")
        if trace_info.get("location_keywords"):
            log_message(
                "DEBUG",
                "stageA location keywords="
                + json.dumps(trace_info.get("location_keywords", []), ensure_ascii=False)
                + " expanded="
                + json.dumps(trace_info.get("location_keyword_expansions", {}), ensure_ascii=False)
                + " any_tokens="
                + json.dumps(trace_info.get("location_tokens", []), ensure_ascii=False),
            )
            log_message(
                "DEBUG",
                "stageA grouped location tokens="
                + json.dumps(
                    {
                        "postcode": trace_info.get("location_postcode_tokens", []),
                        "station": trace_info.get("location_station_tokens", []),
                        "region": trace_info.get("location_region_tokens", []),
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            log_message("DEBUG", "stageA location keywords=[]")

    if hasattr(client, "search"):
        hits = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=qx,
            query_filter=qfilter,
            limit=recall,
            with_payload=True,
            with_vectors=False,
        )
    else:
        qp = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=qx,
            query_filter=qfilter,
            limit=recall,
            with_payload=True,
            with_vectors=False,
        )
        hits = list(getattr(qp, "points", []) or [])

    rows = []
    for h in hits:
        payload = dict(h.payload or {})
        score = float(h.score)
        payload["retrieval_score"] = score
        payload["qdrant_score"] = score
        payload["_qdrant_id"] = h.id
        rows.append(payload)
    if not rows:
        df = pd.DataFrame()
        df.attrs["prefilter_count"] = prefilter_count
        return df
    df = pd.DataFrame(rows).reset_index(drop=True)
    df.attrs["prefilter_count"] = prefilter_count
    return df


def stage_a_search(
    client,
    embedder,
    query: str,
    recall: int,
    c: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    return qdrant_search(client, embedder, query=query, recall=recall, c=c)
