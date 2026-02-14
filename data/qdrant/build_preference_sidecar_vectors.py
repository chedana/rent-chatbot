import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCE_PATH = os.environ.get(
    "RENT_PREF_VECTOR_SOURCE_PATH",
    "/workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/global_properties_with_discovery.jsonl",
)
OUTPUT_PATH = os.environ.get(
    "RENT_PREF_VECTOR_PATH",
    os.path.join(ROOT_DIR, "artifacts", "features", "pref_vectors.parquet"),
)
EMBED_MODEL = os.environ.get("RENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH_SIZE = int(os.environ.get("RENT_EMBED_BATCH", "256"))


def _clean_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if s.lower() in {"", "none", "nan"}:
        return ""
    return s


def _load_source_df(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing source file: {path}")
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path).reset_index(drop=True)
    if path.lower().endswith(".jsonl"):
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
        return pd.DataFrame(rows).reset_index(drop=True)
    raise ValueError("Source must be .jsonl or .parquet")


def _parse_features(v: Any) -> List[str]:
    if isinstance(v, list):
        return [_clean_text(x) for x in v if _clean_text(x)]
    s = _clean_text(v)
    if not s:
        return []
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [_clean_text(x) for x in arr if _clean_text(x)]
        except Exception:
            pass
    # fallback for non-json payloads
    out: List[str] = []
    for part in s.split("|"):
        t = _clean_text(part)
        if t:
            out.append(t)
    return out


def _split_description_para(v: Any) -> List[str]:
    s = _clean_text(v)
    if not s:
        return []
    out: List[str] = []
    for part in s.split("<PARA>"):
        t = _clean_text(part)
        if t:
            out.append(t)
    return out


def main() -> None:
    df = _load_source_df(SOURCE_PATH)
    if len(df) == 0:
        raise RuntimeError("Source is empty.")

    embedder = SentenceTransformer(EMBED_MODEL)
    records: List[Dict[str, Any]] = []
    segment_meta: List[Tuple[int, str]] = []
    segment_texts: List[str] = []

    for i, (_, row) in enumerate(df.iterrows()):
        url = _clean_text(row.get("url"))
        listing_id = _clean_text(row.get("listing_id"))
        features_segments = _parse_features(row.get("features"))
        description_segments = _split_description_para(row.get("description"))
        rec = {
            "url": url,
            "listing_id": listing_id,
            "features_segments": features_segments,
            "description_segments": description_segments,
            "features_vecs": [],
            "description_vecs": [],
        }
        records.append(rec)
        for txt in features_segments:
            segment_meta.append((i, "features"))
            segment_texts.append(txt)
        for txt in description_segments:
            segment_meta.append((i, "description"))
            segment_texts.append(txt)

    if segment_texts:
        vecs = embedder.encode(
            segment_texts,
            batch_size=min(BATCH_SIZE, max(1, len(segment_texts))),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        for (row_idx, field), vec in zip(segment_meta, vecs):
            records[row_idx][f"{field}_vecs"].append(vec.tolist())

    dim = 0
    for rec in records:
        if rec["features_vecs"]:
            dim = len(rec["features_vecs"][0])
            break
        if rec["description_vecs"]:
            dim = len(rec["description_vecs"][0])
            break

    out_rows: List[Dict[str, Any]] = []
    for rec in records:
        out_rows.append(
            {
                "url": rec["url"],
                "listing_id": rec["listing_id"],
                "features_segments": json.dumps(rec["features_segments"], ensure_ascii=False),
                "description_segments": json.dumps(rec["description_segments"], ensure_ascii=False),
                "features_vecs": json.dumps(rec["features_vecs"], ensure_ascii=False),
                "description_vecs": json.dumps(rec["description_vecs"], ensure_ascii=False),
                "vec_model": EMBED_MODEL,
                "vec_dim": int(dim),
            }
        )

    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"[done] source={SOURCE_PATH}")
    print(f"[done] rows={len(out_df)}")
    print(f"[done] output={OUTPUT_PATH}")
    print(f"[done] model={EMBED_MODEL}, dim={dim}")


if __name__ == "__main__":
    main()
