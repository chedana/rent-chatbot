import json
import os
from typing import Any, Dict, List

import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QDRANT_PATH = os.environ.get(
    "RENT_QDRANT_PATH",
    os.path.join(ROOT_DIR, "artifacts", "qdrant_local"),
)
COLLECTION = os.environ.get("RENT_QDRANT_COLLECTION", "rent_listings")
SOURCE_PATH = os.environ.get(
    "RENT_QDRANT_SOURCE_PATH",
    "/workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/global_properties_with_discovery.jsonl",
)
EMBED_MODEL = os.environ.get("RENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = int(os.environ.get("RENT_QDRANT_BATCH", "512"))
RESET_COLLECTION = os.environ.get("RENT_QDRANT_RESET", "1") == "1"
DEDUP_BY_URL = os.environ.get("RENT_QDRANT_DEDUP_BY_URL", "1") == "1"


def _clean_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    return s


def _as_payload_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [_as_payload_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _as_payload_value(x) for k, x in v.items()}
    return str(v)


def _to_float(v: Any) -> Any:
    if v is None:
        return None
    try:
        s = str(v).strip()
        if not s:
            return None
        s = s.replace(",", "")
        s = "".join(ch for ch in s if ch.isdigit() or ch in {".", "-"})
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _to_int(v: Any) -> Any:
    fv = _to_float(v)
    if fv is None:
        return None
    try:
        return int(round(float(fv)))
    except Exception:
        return None


def _norm_cat(v: Any) -> str:
    s = _clean_text(v).lower()
    if not s:
        return ""
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s


def _norm_furnish(v: Any) -> str:
    s = _norm_cat(v)
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


def _norm_property_type(v: Any) -> str:
    s = _norm_cat(v)
    if not s:
        return ""
    if s in {"ask agent", "ask the agent", "unknown", "not provided", "not known", "n/a", "na"}:
        return "ask agent"
    if s == "studio":
        return "studio"
    if s in {"apartment", "apartments", "apt", "apts"}:
        return "apartment"
    if s in {"flat", "flats", "ground flat", "maisonette", "duplex", "penthouse"}:
        return "flat"
    if s in {"house", "detached", "semi detached", "semi-detached", "town house", "terraced", "cottage", "bungalow"}:
        return "house"
    return s


def _norm_let_type(v: Any) -> str:
    s = _norm_cat(v)
    if not s:
        return ""
    if "short" in s:
        return "short term"
    if "long" in s:
        return "long term"
    return s


def build_doc_text(row: pd.Series) -> str:
    fields: List[str] = []
    for k in (
        "title",
        "description",
        "features",
        "address",
        "property_type",
        "furnish_type",
        "let_type",
        "stations",
        "schools",
    ):
        t = _clean_text(row.get(k))
        if t:
            fields.append(t)
    return " | ".join(fields)


def load_source_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing source file: {path}")

    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path).reset_index(drop=True)
    elif path.lower().endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    rows.append(json.loads(s))
                except Exception as e:
                    raise ValueError(f"Invalid JSONL at line {ln}: {e}") from e
        df = pd.DataFrame(rows).reset_index(drop=True)
    else:
        raise ValueError("Source file must be .jsonl or .parquet")

    if len(df) == 0:
        raise RuntimeError("Source file is empty, nothing to index.")

    if DEDUP_BY_URL and "url" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
        print(f"[dedup] by url: {before} -> {len(df)}")
    return df


def main() -> None:
    os.makedirs(QDRANT_PATH, exist_ok=True)
    client = QdrantClient(path=QDRANT_PATH)
    df = load_source_df(SOURCE_PATH)
    texts = [build_doc_text(r) for _, r in df.iterrows()]

    embedder = SentenceTransformer(EMBED_MODEL)
    vecs = embedder.encode(
        texts,
        batch_size=min(256, len(texts)),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dim = int(vecs.shape[1])

    if client.collection_exists(COLLECTION):
        if RESET_COLLECTION:
            client.delete_collection(COLLECTION)
            print(f"[reset] deleted old collection: {COLLECTION}")
        else:
            print(f"[info] Collection exists, append/upsert mode: {COLLECTION}")

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    for i in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[i : i + BATCH_SIZE]
        points = []
        for j, (_, row) in enumerate(chunk.iterrows(), start=i):
            payload = {str(k): _as_payload_value(v) for k, v in row.to_dict().items()}
            payload["_doc_text"] = texts[j]
            payload["price_pcm_num"] = _to_float(row.get("price_pcm"))
            payload["bedrooms_num"] = _to_int(row.get("bedrooms"))
            payload["bathrooms_num"] = _to_float(row.get("bathrooms"))
            payload["let_type_norm"] = _norm_let_type(row.get("let_type"))
            payload["property_type_norm"] = _norm_property_type(row.get("property_type"))
            payload["furnish_type_norm"] = _norm_furnish(row.get("furnish_type"))
            points.append(
                models.PointStruct(
                    id=j,
                    vector=vecs[j].tolist(),
                    payload=payload,
                )
            )
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"[upsert] {min(i + BATCH_SIZE, len(df))}/{len(df)}")

    info = client.get_collection(COLLECTION)
    print(f"[done] source={SOURCE_PATH}")
    print(f"[done] collection={COLLECTION}, points={info.points_count}, dim={dim}")
    print(f"[done] qdrant_path={QDRANT_PATH}")


if __name__ == "__main__":
    main()
