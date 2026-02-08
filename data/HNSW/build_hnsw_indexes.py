import os
import json
import hashlib
import argparse
import glob
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# ----------------------------
# Config
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_HNSW_OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "hnsw", "index")
LEGACY_WEB_CLEAN_JSONL = os.path.join(PROJECT_ROOT, "data", "web_data", "properties_clean.jsonl")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 256

HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 128

PARA_DELIM = "<PARA>"
ASK = "Ask agent"


# ----------------------------
# Helpers
# ----------------------------
def stable_int_id_from_url(url: str) -> int:
    h = hashlib.blake2b(url.encode("utf-8"), digest_size=8).digest()
    u64 = int.from_bytes(h, byteorder="big", signed=False)
    return u64 & ((1 << 63) - 1)

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == ASK.lower() else s

def as_list(x: Any) -> List[str]:
    """
    兼容：
    - list
    - "a | b | c"  (你如果 clean 时把 list join 了)
    - "Ask agent" / "" -> []
    """
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for i in x:
            s = str(i).strip()
            if s and s.lower() != ASK.lower():
                out.append(s)
        return out

    s = str(x).strip()
    if not s or s.lower() == ASK.lower():
        return []

    # JSON string case (common after "all-string" postprocess):
    # examples:
    # - '[{"name":"Canary Wharf Station","miles":0.2}, ...]'
    # - '["feature a","feature b"]'
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
        try:
            parsed = json.loads(s)
            out: List[str] = []
            if isinstance(parsed, list):
                for item in parsed:
                    if item is None:
                        continue
                    if isinstance(item, dict):
                        name = str(item.get("name", "")).strip()
                        miles = item.get("miles", None)
                        if name:
                            if miles is None or str(miles).strip() == "":
                                out.append(name)
                            else:
                                out.append(f"{name} ({miles} miles)")
                        continue
                    t = str(item).strip()
                    if t and t.lower() != ASK.lower():
                        out.append(t)
                return out
            if isinstance(parsed, dict):
                name = str(parsed.get("name", "")).strip()
                miles = parsed.get("miles", None)
                if name:
                    if miles is None or str(miles).strip() == "":
                        return [name]
                    return [f"{name} ({miles} miles)"]
        except Exception:
            # fallback to legacy parsing below
            pass

    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
        return [p for p in parts if p and p.lower() != ASK.lower()]
    return [s]

def build_listing_text(r: Dict[str, Any]) -> str:
    title = safe_str(r.get("title"))
    addr  = safe_str(r.get("address"))
    let_type = safe_str(r.get("let_type"))
    furn = safe_str(r.get("furnish_type"))
    ptype = safe_str(r.get("property_type"))
    bed = safe_str(r.get("bedrooms"))
    bath = safe_str(r.get("bathrooms"))
    price = safe_str(r.get("price_pcm"))

    feats = as_list(r.get("features"))
    feats_txt = "; ".join(feats[:30])

    parts = [
        title,
        f"Address: {addr}" if addr else "",
        f"Bedrooms: {bed} | Bathrooms: {bath}" if (bed or bath) else "",
        f"Price: {price} pcm" if price else "",
        f"Let: {let_type} | Furnished: {furn}" if (let_type or furn) else "",
        f"Type: {ptype}" if ptype else "",
        f"Features: {feats_txt}" if feats_txt else "",
        f"Features: {feats_txt}" if feats_txt else "",  # mild weighting
    ]
    return "\n".join([p for p in parts if p]).strip()

def make_hnsw_ip_index(dim: int) -> faiss.Index:
    idx = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = EF_CONSTRUCTION
    idx.hnsw.efSearch = EF_SEARCH
    return idx

def embed_texts(embedder: SentenceTransformer, texts: List[str]) -> np.ndarray:
    x = embedder.encode(
        texts,
        batch_size=BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    faiss.normalize_L2(x)
    return x

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def enforce_parquet_all_str_except_ids(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    """
    你现在 clean 后全是 str：我们这里也显式让 meta 全是 str（除 id 列）。
    避免 pyarrow 推断类型导致报错。
    """
    for col in df.columns:
        if col in id_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")
        else:
            df[col] = df[col].astype(str)
    return df


def find_latest_clean_jsonl(web_output_root: str) -> str:
    pattern = os.path.join(web_output_root, "*", "properties_clean.jsonl")
    candidates = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    if os.path.isfile(LEGACY_WEB_CLEAN_JSONL):
        return LEGACY_WEB_CLEAN_JSONL
    raise RuntimeError(
        "No properties_clean.jsonl found. "
        f"Checked: {pattern} and legacy path {LEGACY_WEB_CLEAN_JSONL}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-jsonl",
        default=None,
        help="Input cleaned JSONL path. If omitted, auto-pick latest artifacts/web_data/*/properties_clean.jsonl.",
    )
    parser.add_argument(
        "--web-output-root",
        default=os.path.join(PROJECT_ROOT, "artifacts", "web_data"),
        help="Root web output folder used when auto-resolving input.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_HNSW_OUT_DIR,
        help="Output directory for HNSW indexes and metadata.",
    )
    return parser.parse_args()


# ----------------------------
# Main
# ----------------------------
def build_indexes(in_jsonl: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    faiss.omp_set_num_threads(max(1, (os.cpu_count() or 1) - 1))

    list_index_path = os.path.join(out_dir, "listings_hnsw.faiss")
    list_meta_path = os.path.join(out_dir, "listings_meta.parquet")
    evi_index_path = os.path.join(out_dir, "evidence_hnsw.faiss")
    evi_meta_path = os.path.join(out_dir, "evidence_meta.parquet")

    rows = load_jsonl(in_jsonl)
    if not rows:
        raise RuntimeError(f"No records found in {in_jsonl}")

    # ---------- Listing corpus ----------
    listing_meta: List[Dict[str, Any]] = []
    listing_texts: List[str] = []

    for r in rows:
        url = safe_str(r.get("url"))
        if not url:
            continue

        text = build_listing_text(r)
        if not text:
            continue

        listing_texts.append(text)

        # 注意：这里不再依赖 listing_int_id 做 faiss id，只作为字段保留
        listing_meta.append({
            "url": url,
            "listing_int_id": stable_int_id_from_url(url),  # 可选保留
            "title": r.get("title", ""),
            "address": r.get("address", ""),
            "price_pcm": r.get("price_pcm", ""),
            "price_pw": r.get("price_pw", ""),
            "bedrooms": r.get("bedrooms", ""),
            "bathrooms": r.get("bathrooms", ""),
            "let_type": r.get("let_type", ""),
            "furnish_type": r.get("furnish_type", ""),
            "property_type": r.get("property_type", ""),
            "available_from": r.get("available_from", ""),
            "added_date": r.get("added_date", ""),
            "text_for_embedding": text,
        })

    if not listing_texts:
        raise RuntimeError("No valid listing texts were built (check url/title/address/fields).")

    # ---------- Evidence corpus ----------
    evi_meta: List[Dict[str, Any]] = []
    evi_texts: List[str] = []

    for r in rows:
        url = safe_str(r.get("url"))
        if not url:
            continue
        lid = stable_int_id_from_url(url)

        desc = safe_str(r.get("description"))
        if desc:
            paras = [p.strip() for p in desc.split(PARA_DELIM)]
            for p in paras:
                if not p:
                    continue
                evi_texts.append(p)
                evi_meta.append({
                    "url": url,
                    "listing_int_id": lid,
                    "chunk_type": "description",
                    "chunk_text": p,
                })

        for field, ctype in [("features", "features"), ("stations", "stations"), ("schools", "schools")]:
            items = as_list(r.get(field))
            for it in items:
                it = it.strip()
                if not it:
                    continue
                evi_texts.append(it)
                evi_meta.append({
                    "url": url,
                    "listing_int_id": lid,
                    "chunk_type": ctype,
                    "chunk_text": it,
                })

    # ---------- Embed ----------
    embedder = SentenceTransformer(EMBED_MODEL)
    dim = embedder.get_sentence_embedding_dimension()

    print(f"[Listing] {len(listing_texts)} vectors, dim={dim}")
    list_x = embed_texts(embedder, listing_texts)

    print(f"[Evidence] {len(evi_texts)} vectors, dim={dim}")
    evi_x = embed_texts(embedder, evi_texts) if evi_texts else None

    # ---------- Build FAISS (关键：用 add()，不要 add_with_ids) ----------
    list_index = make_hnsw_ip_index(dim)
    list_index.add(list_x)  # faiss_id = 0..N-1

    faiss.write_index(list_index, list_index_path)

    # 给 listing_meta 补 faiss_id（严格对应 add() 的顺序）
    for i, m in enumerate(listing_meta):
        m["faiss_id"] = i

    df_list = pd.DataFrame(listing_meta)
    df_list = enforce_parquet_all_str_except_ids(df_list, id_cols=["faiss_id", "listing_int_id"])
    df_list.to_parquet(list_meta_path, index=False)

    print(f"Saved: {list_index_path}")
    print(f"Saved: {list_meta_path}")

    if evi_x is not None and len(evi_texts) > 0:
        evi_index = make_hnsw_ip_index(dim)
        evi_index.add(evi_x)  # evidence faiss_id = 0..M-1

        faiss.write_index(evi_index, evi_index_path)

        for i, m in enumerate(evi_meta):
            m["faiss_id"] = i

        df_evi = pd.DataFrame(evi_meta)
        df_evi = enforce_parquet_all_str_except_ids(df_evi, id_cols=["faiss_id", "listing_int_id"])
        df_evi.to_parquet(evi_meta_path, index=False)

        print(f"Saved: {evi_index_path}")
        print(f"Saved: {evi_meta_path}")
    else:
        print("No evidence texts found; evidence index not created.")

    print("Done.")

if __name__ == "__main__":
    args = parse_args()
    in_jsonl = args.in_jsonl or find_latest_clean_jsonl(args.web_output_root)
    print(f"Input JSONL: {in_jsonl}")
    print(f"Output dir : {args.out_dir}")
    build_indexes(in_jsonl=in_jsonl, out_dir=args.out_dir)
