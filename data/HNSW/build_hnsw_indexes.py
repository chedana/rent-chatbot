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
DEFAULT_SECTION_WEIGHTS = {
    "core": 3.0,
    "stations": 2.0,
    "schools": 1.5,
    "features": 1.2,
    "description": 0.9,
}
CORE_SUBSECTION_WEIGHTS = {
    "location": 0.40,
    "rooms": 0.20,
    "type": 0.15,
    "price": 0.15,
    "furnish": 0.10,
}


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

def clean_description(desc: Any, max_paras: int, max_chars: int = 2400) -> str:
    raw = safe_str(desc)
    if not raw:
        return ""
    paras = [p.strip() for p in raw.split(PARA_DELIM) if p.strip()]
    out = " ".join(paras[:max_paras]).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0].strip()
    return out

def join_items(items: List[str], max_items: int) -> str:
    if not items:
        return ""
    return "; ".join([x for x in items[:max_items] if x.strip()])

def truncate_text(s: str, max_chars: int) -> str:
    s = safe_str(s)
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    clipped = s[:max_chars]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.strip()

def repeat_fill(chunks: List[str], max_chars: int, sep: str = " | ") -> str:
    target = max(0, int(max_chars))
    if target <= 0:
        return ""
    clean = [safe_str(x) for x in (chunks or []) if safe_str(x)]
    if not clean:
        return ""

    out = ""
    i = 0
    n = len(clean)
    while len(out) < target and n > 0:
        piece = clean[i % n]
        candidate = piece if not out else (out + sep + piece)
        if len(candidate) >= target:
            out = candidate[:target]
            break
        out = candidate
        i += 1
        if i > 20000:
            break
    return out.rstrip(" |;\n\t")

def section_char_limits(cfg: Dict[str, Any]) -> Dict[str, int]:
    total = float(cfg["section_char_budget"])
    weights = {
        "core": float(cfg["w_core"]),
        "stations": float(cfg["w_stations"]),
        "schools": float(cfg["w_schools"]),
        "features": float(cfg["w_features"]),
        "description": float(cfg["w_description"]),
    }
    sum_w = sum(weights.values()) or 1.0
    limits = {}
    for k, w in weights.items():
        ratio = max(0.0, w) / sum_w
        limits[k] = max(120, int(round(total * ratio)))
    return limits

def core_subsection_char_limits(core_budget: int) -> Dict[str, int]:
    total = max(120, int(core_budget))
    sum_w = sum(CORE_SUBSECTION_WEIGHTS.values()) or 1.0
    limits: Dict[str, int] = {}
    for k, w in CORE_SUBSECTION_WEIGHTS.items():
        limits[k] = max(32, int(round(total * (float(w) / sum_w))))
    return limits

def _flatten_discovery_queries(v: Any) -> List[str]:
    if v is None:
        return []
    obj = v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
        except Exception:
            return [s]
    out: List[str] = []
    if isinstance(obj, dict):
        for _, items in obj.items():
            if isinstance(items, list):
                for it in items:
                    t = safe_str(it)
                    if t:
                        out.append(t)
            else:
                t = safe_str(items)
                if t:
                    out.append(t)
    elif isinstance(obj, list):
        for it in obj:
            t = safe_str(it)
            if t:
                out.append(t)
    seen = set()
    dedup = []
    for x in out:
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(x)
    return dedup

def _parse_discovery_map(v: Any) -> Dict[str, List[str]]:
    obj = v
    if v is None:
        return {}
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
        except Exception:
            return {}
    if not isinstance(obj, dict):
        return {}

    out: Dict[str, List[str]] = {}
    for k, items in obj.items():
        key = safe_str(k).lower()
        vals: List[str] = []
        if isinstance(items, list):
            for it in items:
                t = safe_str(it)
                if t:
                    vals.append(t)
        else:
            t = safe_str(items)
            if t:
                vals.append(t)
        if vals:
            seen = set()
            dedup = []
            for x in vals:
                xl = x.lower()
                if xl in seen:
                    continue
                seen.add(xl)
                dedup.append(x)
            out[key] = dedup
    return out

def build_core_text(r: Dict[str, Any], cfg: Dict[str, Any], include_labels: bool = True) -> str:
    title = safe_str(r.get("title"))
    addr = safe_str(r.get("address"))
    let_type = safe_str(r.get("let_type"))
    furn = safe_str(r.get("furnish_type"))
    ptype = safe_str(r.get("property_type"))
    bed = safe_str(r.get("bedrooms"))
    bath = safe_str(r.get("bathrooms"))
    price = safe_str(r.get("price_pcm"))
    discovery_map = _parse_discovery_map(r.get("discovery_queries_by_method"))
    stations = as_list(r.get("stations"))

    limits = section_char_limits(cfg)
    core_limits = core_subsection_char_limits(limits["core"])

    loc_bits: List[str] = []
    if discovery_map.get("region"):
        loc_bits.append("Discovery region: " + "; ".join(discovery_map["region"][:8]))
    if discovery_map.get("station"):
        loc_bits.append("Discovery station: " + "; ".join(discovery_map["station"][:8]))
    extra_discovery = []
    for k, vals in discovery_map.items():
        if k in ("region", "station"):
            continue
        if vals:
            extra_discovery.append(f"{k}: " + "; ".join(vals[:8]))
    if extra_discovery:
        loc_bits.extend(extra_discovery)
    if addr:
        loc_bits.append("Address: " + addr)
    if stations:
        loc_bits.append("Nearby stations: " + "; ".join(stations[:12]))
    location_txt = repeat_fill(loc_bits, core_limits["location"], sep=" | ")

    rooms_txt = ""
    if bed or bath:
        rooms_txt = repeat_fill(
            [f"Bedrooms: {bed if bed else 'unknown'}", f"Bathrooms: {bath if bath else 'unknown'}"],
            core_limits["rooms"],
            sep=" | ",
        )

    type_bits: List[str] = []
    if ptype:
        type_bits.append(f"Type: {ptype}")
    if title:
        type_bits.append(f"Listing: {title}")
    type_txt = repeat_fill(type_bits, core_limits["type"], sep=" | ")

    price_chunks: List[str] = []
    if price:
        price_chunks.append(f"Price: {price} pcm")
    price_pw = safe_str(r.get("price_pw"))
    if price_pw:
        price_chunks.append(f"Price: {price_pw} pw")
    deposit = safe_str(r.get("deposit"))
    if deposit:
        price_chunks.append(f"Deposit: {deposit}")
    price_txt = repeat_fill(price_chunks, core_limits["price"], sep=" | ")

    furnish_bits: List[str] = []
    if furn:
        furnish_bits.append(f"Furnished: {furn}")
    if let_type:
        furnish_bits.append(f"Let: {let_type}")
    min_tenancy = safe_str(r.get("min_tenancy"))
    if min_tenancy:
        furnish_bits.append(f"Min tenancy: {min_tenancy}")
    furnish_txt = repeat_fill(furnish_bits, core_limits["furnish"], sep=" | ")

    if include_labels:
        parts = [
            f"Location: {location_txt}" if location_txt else "",
            f"Rooms: {rooms_txt}" if rooms_txt else "",
            f"Type: {type_txt}" if type_txt else "",
            f"Price: {price_txt}" if price_txt else "",
            f"Furnish: {furnish_txt}" if furnish_txt else "",
        ]
    else:
        parts = [x for x in [location_txt, rooms_txt, type_txt, price_txt, furnish_txt] if x]
    return " | ".join([p for p in parts if p]).strip()

def build_listing_text(r: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    feats = as_list(r.get("features"))
    stations = as_list(r.get("stations"))
    schools = as_list(r.get("schools"))
    desc = clean_description(
        r.get("description"),
        max_paras=cfg["max_desc_paras"],
        max_chars=cfg["max_desc_chars"],
    )
    feats = feats[: cfg["max_features"]]
    stations = stations[: cfg["max_stations"]]
    schools = schools[: cfg["max_schools"]]
    limits = section_char_limits(cfg)

    core_text = build_core_text(r, cfg, include_labels=False)
    stations_txt = repeat_fill(stations, limits["stations"], sep="; ")
    schools_txt = repeat_fill(schools, limits["schools"], sep="; ")
    feats_txt = repeat_fill(feats, limits["features"], sep="; ")
    desc_chunks = [p.strip() for p in desc.split(PARA_DELIM) if p.strip()] if desc else []
    if not desc_chunks and desc:
        desc_chunks = [desc]
    desc = repeat_fill(desc_chunks, limits["description"], sep=" ")

    # One section per field group: avoid accidental multi-weighting from repeated blocks.
    parts = [
        f"Core: {core_text}" if core_text else "",
        f"Stations: {stations_txt}" if stations_txt else "",
        f"Schools: {schools_txt}" if schools_txt else "",
        f"Features: {feats_txt}" if feats_txt else "",
        f"Description: {desc}" if desc else "",
    ]

    return "\n".join([p for p in parts if p]).strip()

def build_core_text_for_meta(r: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    return build_core_text(r, cfg, include_labels=False)

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
    parser.add_argument("--max-desc-paras", type=int, default=8, help="Max description paragraphs included per listing.")
    parser.add_argument("--max-desc-chars", type=int, default=2400, help="Max description chars included per listing.")
    parser.add_argument("--max-features", type=int, default=40, help="Max feature items included per listing.")
    parser.add_argument("--max-stations", type=int, default=20, help="Max station items included per listing.")
    parser.add_argument("--max-schools", type=int, default=20, help="Max school items included per listing.")
    parser.add_argument("--section-char-budget", type=int, default=3600, help="Total char budget distributed by section weights.")
    parser.add_argument("--w-core", type=float, default=DEFAULT_SECTION_WEIGHTS["core"], help="Weight for core section.")
    parser.add_argument("--w-stations", type=float, default=DEFAULT_SECTION_WEIGHTS["stations"], help="Weight for stations section.")
    parser.add_argument("--w-schools", type=float, default=DEFAULT_SECTION_WEIGHTS["schools"], help="Weight for schools section.")
    parser.add_argument("--w-features", type=float, default=DEFAULT_SECTION_WEIGHTS["features"], help="Weight for features section.")
    parser.add_argument("--w-description", type=float, default=DEFAULT_SECTION_WEIGHTS["description"], help="Weight for description section.")
    return parser.parse_args()


# ----------------------------
# Main
# ----------------------------
def build_indexes(in_jsonl: str, out_dir: str, cfg: Dict[str, Any]):
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

        text = build_listing_text(r, cfg)
        if not text:
            continue

        features_txt = join_items(as_list(r.get("features")), cfg["max_features"])
        stations_txt = join_items(as_list(r.get("stations")), cfg["max_stations"])
        schools_txt = join_items(as_list(r.get("schools")), cfg["max_schools"])
        desc_txt = clean_description(
            r.get("description"),
            max_paras=cfg["max_desc_paras"],
            max_chars=cfg["max_desc_chars"],
        )
        limits = section_char_limits(cfg)
        features_txt = truncate_text(features_txt, limits["features"])
        stations_txt = truncate_text(stations_txt, limits["stations"])
        schools_txt = truncate_text(schools_txt, limits["schools"])
        desc_txt = truncate_text(desc_txt, limits["description"])
        core_txt = build_core_text_for_meta(r, cfg)

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
            "core": core_txt,
            "description": desc_txt,
            "features": features_txt,
            "stations": stations_txt,
            "schools": schools_txt,
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
    cfg = {
        "max_desc_paras": max(1, int(args.max_desc_paras)),
        "max_desc_chars": max(200, int(args.max_desc_chars)),
        "max_features": max(1, int(args.max_features)),
        "max_stations": max(1, int(args.max_stations)),
        "max_schools": max(1, int(args.max_schools)),
        "section_char_budget": max(1200, int(args.section_char_budget)),
        "w_core": max(0.1, float(args.w_core)),
        "w_stations": max(0.1, float(args.w_stations)),
        "w_schools": max(0.1, float(args.w_schools)),
        "w_features": max(0.1, float(args.w_features)),
        "w_description": max(0.1, float(args.w_description)),
    }
    print(f"Input JSONL: {in_jsonl}")
    print(f"Output dir : {args.out_dir}")
    print(f"Build cfg  : {json.dumps(cfg, ensure_ascii=False)}")
    build_indexes(in_jsonl=in_jsonl, out_dir=args.out_dir, cfg=cfg)
