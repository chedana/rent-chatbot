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

from helpers import _safe_text
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
        "location_tokens": [],
    }

    def _build_qdrant_filter(c: Optional[Dict[str, Any]]) -> Optional["models.Filter"]:
        c = c or {}
        must: List[Any] = []

        loc_values: List[str] = []
        loc_keywords = [str(x).strip() for x in (c.get("location_keywords") or []) if str(x).strip()]
        trace_info["location_keywords"] = loc_keywords
        for term in loc_keywords:
            raw = _safe_text(term).lower()
            if not raw:
                continue
            raw = re.sub(r"\s+", " ", raw).strip()
            slug = re.sub(r"[^a-z0-9]+", "_", raw)
            slug = re.sub(r"_+", "_", slug).strip("_")
            if raw:
                loc_values.append(raw)
                if " " in raw:
                    loc_values.append(raw.replace(" ", ""))
            if slug:
                loc_values.append(slug)
                loc_values.append(f"{slug}_london")
                if not slug.endswith("_station"):
                    loc_values.append(f"{slug}_station")

        loc_values = list(dict.fromkeys([x for x in loc_values if x]))
        trace_info["location_tokens"] = loc_values
        if loc_values:
            must.append(
                models.FieldCondition(
                    key="location_tokens",
                    match=models.MatchAny(any=loc_values),
                )
            )

        if not must:
            return None
        return models.Filter(must=must)

    qx = embed_query(embedder, query)[0].tolist()
    qfilter = _build_qdrant_filter(c) if QDRANT_ENABLE_PREFILTER else None
    if STAGEA_TRACE:
        log_message("DEBUG", f"stageA backend=qdrant recall={recall} prefilter={QDRANT_ENABLE_PREFILTER}")
        log_message("DEBUG", f"stageA query={query}")
        if trace_info.get("location_keywords"):
            log_message(
                "DEBUG",
                "stageA location keywords="
                + json.dumps(trace_info.get("location_keywords", []), ensure_ascii=False)
                + " tokens="
                + json.dumps(trace_info.get("location_tokens", []), ensure_ascii=False),
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
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def stage_a_search(
    client,
    embedder,
    query: str,
    recall: int,
    c: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    return qdrant_search(client, embedder, query=query, recall=recall, c=c)
