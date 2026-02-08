import os
import sys
import re
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


from openai import OpenAI

QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
QWEN_MODEL    = os.environ.get("QWEN_MODEL", "./Qwen2.5-7B-Instruct")  # 你启动时的 model 名称
QWEN_API_KEY  = os.environ.get("OPENAI_API_KEY", "dummy")

qwen_client = OpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY)

EXTRACT_SYSTEM = """You output STRICT JSON only (no markdown, no explanation).
Schema:
{
  "max_rent_pcm": number|null,
  "bedrooms": int|null,
  "bedrooms_op": string|null,
  "bathrooms": number|null,
  "bathrooms_op": string|null,
  "available_from": string|null,
  "available_from_op": string|null,
  "location_keywords": string[],
  "must_have_keywords": string[],
  "k": int|null
}
Rules:
- location_keywords are place names/areas/postcodes (e.g., "Canary Wharf", "E14", "Shoreditch").
- must_have_keywords are requirements/features (e.g., "balcony", "pet friendly", "near tube").
- bedrooms_op must be one of: "eq", "gte", or null.
- Set bedrooms/bedrooms_op only for hard constraints:
  - "at least/minimum/>= X bedrooms" -> {"bedrooms": X, "bedrooms_op": "gte"}
  - "exactly/only X bedrooms" -> {"bedrooms": X, "bedrooms_op": "eq"}
  - soft wording (prefer/ideally/nice to have) -> bedrooms = null, bedrooms_op = null
- bathrooms_op must be one of: "eq", "gte", or null.
- Set bathrooms/bathrooms_op only for hard constraints:
  - "at least/minimum/>= X bathrooms" -> {"bathrooms": X, "bathrooms_op": "gte"}
  - "exactly/only X bathrooms" -> {"bathrooms": X, "bathrooms_op": "eq"}
  - soft wording (prefer/ideally/nice to have) -> bathrooms = null, bathrooms_op = null
- available_from should be an ISO date string "YYYY-MM-DD" when possible.
- available_from_op must be one of: "gte", "lte", or null.
- Set available_from/available_from_op as:
  - "available from DATE" -> {"available_from": "DATE", "available_from_op": "gte"}
  - "available by/before DATE" -> {"available_from": "DATE", "available_from_op": "lte"}
- If unknown use null or [].
"""

NEAR_WORDS = {
    "near subway","near station","near tube","tube","subway","station","close to station","near metro",
    "near underground","near tube station","close to tube","walk to station"
}

def qwen_chat(messages, temperature=0.0) -> str:
    r = qwen_client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        temperature=temperature
    )
    return r.choices[0].message.content.strip()

def _extract_json_obj(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError("No JSON found. Got:\n" + txt)
    return json.loads(m.group(0))

def llm_extract(user_text: str, existing_constraints: Optional[dict]) -> dict:
    prefix = ""
    if existing_constraints:
        prefix = "Existing constraints (JSON):\n" + json.dumps(existing_constraints, ensure_ascii=False) + "\n\n"

    txt = qwen_chat(
        [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": prefix + "User says:\n" + user_text}
        ],
        temperature=0.0
    )
    obj = _extract_json_obj(txt)
    obj.setdefault("k", None)
    obj.setdefault("bedrooms_op", None)
    obj.setdefault("bathrooms", None)
    obj.setdefault("bathrooms_op", None)
    obj.setdefault("available_from", None)
    obj.setdefault("available_from_op", None)
    obj.setdefault("location_keywords", [])
    obj.setdefault("must_have_keywords", [])
    return obj
def normalize_budget_to_pcm(c: dict) -> dict:
    """
    Normalize budget constraints to pcm.
    Supports:
      - max_rent_pcm
      - max_rent_pcw
    Priority:
      - If both provided, pcm wins.
    """
    if c is None:
        return c

    # if user gave pcw, convert to pcm
    if c.get("max_rent_pcm") is None and c.get("max_rent_pcw") is not None:
        try:
            pcw = float(c["max_rent_pcw"])
            c["max_rent_pcm"] = pcw * 52.0 / 12.0
        except:
            pass

    return c
def normalize_constraints(c: dict) -> dict:
    # normalize bedrooms hard-constraint operator
    bed_op = c.get("bedrooms_op")
    if bed_op is not None:
        bed_op = str(bed_op).strip().lower()
        if bed_op in ("==", "=", "exact", "exactly", "eq"):
            bed_op = "eq"
        elif bed_op in (">=", "min", "minimum", "at_least", "at least", "gte"):
            bed_op = "gte"
        else:
            bed_op = None
    c["bedrooms_op"] = bed_op

    if c.get("bedrooms") is not None:
        try:
            c["bedrooms"] = int(float(c["bedrooms"]))
        except:
            c["bedrooms"] = None

    # default to strict equality when bedrooms is set but operator is absent
    if c.get("bedrooms") is not None and c.get("bedrooms_op") is None:
        c["bedrooms_op"] = "eq"

    # normalize bathrooms hard-constraint operator
    op = c.get("bathrooms_op")
    if op is not None:
        op = str(op).strip().lower()
        if op in ("==", "=", "exact", "exactly", "eq"):
            op = "eq"
        elif op in (">=", "min", "minimum", "at_least", "at least", "gte"):
            op = "gte"
        else:
            op = None
    c["bathrooms_op"] = op

    if c.get("bathrooms") is not None:
        try:
            c["bathrooms"] = float(c["bathrooms"])
        except:
            c["bathrooms"] = None

    # default to strict equality when bathrooms is set but operator is absent
    if c.get("bathrooms") is not None and c.get("bathrooms_op") is None:
        c["bathrooms_op"] = "eq"

    # normalize available_from hard-constraint operator
    date_op = c.get("available_from_op")
    if date_op is not None:
        date_op = str(date_op).strip().lower()
        if date_op in (">=", "after", "from", "available from", "no_earlier_than", "no earlier than", "gte"):
            date_op = "gte"
        elif date_op in ("<=", "before", "by", "no_later_than", "no later than", "lte"):
            date_op = "lte"
        else:
            date_op = None
    c["available_from_op"] = date_op

    if c.get("available_from") is not None:
        dt = pd.to_datetime(c.get("available_from"), errors="coerce")
        if pd.notna(dt):
            c["available_from"] = dt.date().isoformat()
        else:
            c["available_from"] = None

    # default to "available from" style when date exists but op is absent
    if c.get("available_from") is not None and c.get("available_from_op") is None:
        c["available_from_op"] = "gte"

    locs = []
    must = set([str(x).strip() for x in (c.get("must_have_keywords") or []) if str(x).strip()])
    for x in (c.get("location_keywords") or []):
        s = str(x).strip()
        if not s:
            continue
        if s.lower() in NEAR_WORDS:
            must.add(s)
        else:
            locs.append(s)
    c["location_keywords"] = locs
    c["must_have_keywords"] = list(must)
    return c

def merge_constraints(old: Optional[dict], new: dict) -> dict:
    if old is None:
        old = {}
    out = dict(old)

    # scalar fields: new overrides if not null
    for key in [
        "max_rent_pcm",
        "bedrooms",
        "bedrooms_op",
        "bathrooms",
        "bathrooms_op",
        "available_from",
        "available_from_op",
        "k",
    ]:
        if new.get(key) is not None:
            out[key] = new.get(key)

    def merge_list(a, b):
        a = a or []
        b = b or []
        seen = set()
        res = []
        for x in a + b:
            s = str(x).strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            res.append(s)
        return res

    out["location_keywords"] = merge_list(old.get("location_keywords"), new.get("location_keywords"))
    out["must_have_keywords"] = merge_list(old.get("must_have_keywords"), new.get("must_have_keywords"))

    # default k
    if out.get("k") is None:
        out["k"] = DEFAULT_K

    return normalize_constraints(out)

def constraints_to_query_hint(c: dict) -> str:
    parts = []
    if c.get("bedrooms") is not None:
        bed_op = c.get("bedrooms_op", "eq")
        if bed_op == "gte":
            parts.append(f"at least {int(c['bedrooms'])} bedroom")
        else:
            parts.append(f"{int(c['bedrooms'])} bedroom")
    if c.get("bathrooms") is not None:
        op = c.get("bathrooms_op", "eq")
        if op == "gte":
            parts.append(f"at least {float(c['bathrooms']):g} bathroom")
        else:
            parts.append(f"{float(c['bathrooms']):g} bathroom")
    if c.get("max_rent_pcm") is not None:
        parts.append(f"budget {float(c['max_rent_pcm'])} pcm")
    if c.get("available_from") is not None:
        dt_op = c.get("available_from_op", "gte")
        if dt_op == "gte":
            parts.append(f"available from {c['available_from']}")
        else:
            parts.append(f"available by {c['available_from']}")
    for x in (c.get("location_keywords") or [])[:5]:
        parts.append(x)
    for x in (c.get("must_have_keywords") or [])[:5]:
        parts.append(x)
    return " | ".join(parts)


# project root = this file's directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

OUT_DIR = os.environ.get(
    "RENT_INDEX_DIR",
    os.path.join(ROOT_DIR, "data/HNSW/index")
)

LIST_INDEX_PATH = os.path.join(OUT_DIR, "listings_hnsw.faiss")
LIST_META_PATH  = os.path.join(OUT_DIR, "listings_meta.parquet")


EMBED_MODEL = os.environ.get("RENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH = int(os.environ.get("RENT_EMBED_BATCH", "256"))

DEFAULT_K = int(os.environ.get("RENT_K", "5"))
DEFAULT_RECALL = int(os.environ.get("RENT_RECALL", "200"))  # pull more then slice to k
UNKNOWN_PENALTY_WEIGHTS = {
    "price": 0.35,
    "bedrooms": 0.30,
    "bathrooms": 0.20,
    "available_from": 0.15,
}
UNKNOWN_PENALTY_CAP = 0.60
SOFT_MUST_HIT_BONUS = 0.04
SOFT_MUST_MISS_PENALTY = 0.06
SOFT_MUST_MAX_BONUS = 0.20
SOFT_MUST_MAX_PENALTY = 0.30
FAISS_SCORE_WEIGHT = float(os.environ.get("RENT_FAISS_WEIGHT", "0.35"))
PREFERENCE_SCORE_WEIGHT = float(os.environ.get("RENT_PREF_WEIGHT", "1.0"))
UNKNOWN_PENALTY_WEIGHT = float(os.environ.get("RENT_UNKNOWN_PENALTY_WEIGHT", "1.0"))
SOFT_PENALTY_WEIGHT = float(os.environ.get("RENT_SOFT_PENALTY_WEIGHT", "1.0"))

ROUTE_WEIGHT_TEMPLATES = {
    "HARD_DOMINANT": {
        "w_faiss": 0.15,
        "w_pref": 0.80,
        "w_unknown": 1.20,
        "w_soft": 0.80,
    },
    "HYBRID": {
        "w_faiss": 0.25,
        "w_pref": 1.00,
        "w_unknown": 1.00,
        "w_soft": 1.00,
    },
    "SEMANTIC_DOMINANT": {
        "w_faiss": 0.40,
        "w_pref": 1.10,
        "w_unknown": 0.80,
        "w_soft": 1.10,
    },
}

def decide_route(c: dict) -> str:
    if c is None:
        return "HYBRID"

    hard_fields = [
        "max_rent_pcm",
        "bedrooms",
        "bathrooms",
        "available_from",
        "furnish_type",
        "let_type",
        "property_type",
    ]
    hard_count = sum(1 for k in hard_fields if c.get(k) is not None)

    semantic_count = 0
    semantic_count += len([x for x in (c.get("must_have_keywords") or []) if str(x).strip()])
    semantic_count += len([x for x in (c.get("location_keywords") or []) if str(x).strip()])

    if hard_count >= 3 and semantic_count <= 1:
        return "HARD_DOMINANT"
    if hard_count <= 1 and semantic_count >= 2:
        return "SEMANTIC_DOMINANT"
    return "HYBRID"


# ----------------------------
# Load resources
# ----------------------------
def load_index_and_meta():
    if not os.path.exists(LIST_INDEX_PATH):
        raise FileNotFoundError(f"Missing FAISS index: {LIST_INDEX_PATH}")
    if not os.path.exists(LIST_META_PATH):
        raise FileNotFoundError(f"Missing meta parquet: {LIST_META_PATH}")

    index = faiss.read_index(LIST_INDEX_PATH)
    meta = pd.read_parquet(LIST_META_PATH).copy()

    # 关键：保证 0..N-1 行号对齐（用于 iloc）
    meta = meta.reset_index(drop=True)

    print(f"[boot] faiss ntotal={index.ntotal}, meta rows={len(meta)}")
    return index, meta

def embed_query(embedder: SentenceTransformer, q: str) -> np.ndarray:
    x = embedder.encode(
        [q],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,  # match your builder, then normalize with faiss
    ).astype("float32")
    faiss.normalize_L2(x)  # IMPORTANT: match builder
    return x


# ----------------------------
# Retrieval
# ----------------------------
def faiss_search(index, embedder, meta: pd.DataFrame, query: str, recall: int) -> pd.DataFrame:
    qx = embed_query(embedder, query)
    scores, ids = index.search(qx, recall)

    ids = ids[0].tolist()
    scores = scores[0].tolist()

    rows = []
    n = len(meta)

    for lid, sc in zip(ids, scores):
        if lid is None:
            continue
        lid = int(lid)
        if lid < 0 or lid >= n:
            continue

        r = meta.iloc[lid].to_dict()
        r["faiss_score"] = float(sc)
        r["_faiss_id"] = lid
        rows.append(r)

    if not rows:
        return meta.head(0).copy()

    # IMPORTANT:
    # Do not truncate to k before hard filters. We need a larger recall pool first.
    return pd.DataFrame(rows).reset_index(drop=True)

def format_listing_row(r: Dict[str, Any], i: int) -> str:
    title = str(r.get("title", "") or "").strip()
    url = str(r.get("url", "") or "").strip()
    address = str(r.get("address", "") or "").strip()
    price_pcm = r.get("price_pcm", None)
    beds = r.get("bedrooms", None)
    baths = r.get("bathrooms", None)

    def norm_num(x):
        if x is None:
            return None
        try:
            if isinstance(x, str):
                # allow "1998" or "£1998"
                x2 = re.sub(r"[^\d\.]", "", x)
                return float(x2) if x2 else None
            return float(x)
        except:
            return None

    price = norm_num(price_pcm)
    beds_n = None
    try:
        beds_n = int(float(beds)) if beds is not None and str(beds).strip() != "" else None
    except:
        beds_n = None

    baths_n = None
    try:
        baths_n = int(float(baths)) if baths is not None and str(baths).strip() != "" else None
    except:
        baths_n = None

    bits = []
    bits.append(f"{i}. {title}" if title else f"{i}. (no title)")
    line2 = []
    if price is not None:
        line2.append(f"£{int(round(price))}/pcm")
    else:
        # keep raw if exists
        if price_pcm is not None and str(price_pcm).strip():
            line2.append(f"{price_pcm} pcm")
    if beds_n is not None:
        line2.append(f"{beds_n} bed")
    elif beds is not None and str(beds).strip():
        line2.append(f"{beds} bed")
    if baths_n is not None:
        line2.append(f"{baths_n} bath")
    elif baths is not None and str(baths).strip():
        line2.append(f"{baths} bath")

    if address:
        line2.append(address)
    if line2:
        bits.append("   " + " | ".join(line2))
    if url:
        bits.append("   " + url)

    # scoring breakdown (if ranking fields exist)
    if r.get("final_score", None) is not None:
        def f(x):
            try:
                return f"{float(x):.4f}"
            except:
                return "0.0000"

        w_faiss = r.get("w_faiss", FAISS_SCORE_WEIGHT)
        w_pref = r.get("w_pref", PREFERENCE_SCORE_WEIGHT)
        w_unknown = r.get("w_unknown", UNKNOWN_PENALTY_WEIGHT)
        w_soft = r.get("w_soft", SOFT_PENALTY_WEIGHT)
        route = str(r.get("route", "HYBRID"))
        bits.append(
            "   "
            + f"score: final={f(r.get('final_score'))} = "
            + f"{f(w_faiss)}*faiss_norm({f(r.get('faiss_score_norm'))}) + "
            + f"{f(w_pref)}*pref({f(r.get('preference_score'))}) - "
            + f"{f(w_unknown)}*unknown({f(r.get('unknown_penalty'))}) - "
            + f"{f(w_soft)}*soft_miss({f(r.get('violation_soft_penalty'))})"
        )
        bits.append("   " + f"route: {route}")

        unknown_reason = str(r.get("unknown_penalty_reasons", "") or "").strip()
        soft_reason = str(r.get("soft_penalty_reasons", "") or "").strip()
        if unknown_reason:
            bits.append("   " + f"unknown_penalty_reasons: {unknown_reason}")
        if soft_reason:
            bits.append("   " + f"soft_penalty_reasons: {soft_reason}")
    return "\n".join(bits)


# ----------------------------
# Interactive CLI
# ----------------------------
def parse_command(s: str) -> Tuple[Optional[str], str]:
    s = s.strip()
    if not s.startswith("/"):
        return None, ""
    parts = s.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return cmd, arg


def run_chat():
    index, meta = load_index_and_meta()
    embedder = SentenceTransformer(EMBED_MODEL)

    state = {
        "history": [],   # list of (user, assistant_text) for your own future use
        "k": DEFAULT_K,
        "recall": DEFAULT_RECALL,
        "last_query": None,
        "last_df": None,
        "constraints": None,
    }

    print("RentBot (minimal retrieval)")
    print("Commands: /exit /reset /k N /show /recall N /constraints /model")
    print(f"Index: {LIST_INDEX_PATH}")
    print(f"Meta : {LIST_META_PATH}")
    print(f"Embed: {EMBED_MODEL}")
    print("----")

    while True:
        try:
            user_in = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_in:
            continue

        cmd, arg = parse_command(user_in)

        if cmd == "/exit":
            print("Bye.")
            break

        if cmd == "/reset":
            state["history"] = []
            state["last_query"] = None
            state["last_df"] = None
            state["constraints"] = None
            print("State reset.")
            continue

        if cmd == "/k":
            try:
                n = int(arg)
                if n <= 0 or n > 50:
                    raise ValueError()
                state["k"] = n
                # 关键：同时写入 constraints.k，保证后面用到的是新值
                if state["constraints"] is None:
                    state["constraints"] = {
                        "k": n,
                        "location_keywords": [],
                        "must_have_keywords": [],
                        "max_rent_pcm": None,
                        "bedrooms": None,
                        "bedrooms_op": None,
                        "bathrooms": None,
                        "bathrooms_op": None,
                        "available_from": None,
                        "available_from_op": None,
                    }
                else:
                    state["constraints"]["k"] = n
                print(f"OK. k = {n}")
            except:
                print("Usage: /k 5   (1~50)")
            continue

        if cmd == "/recall":
            try:
                n = int(arg)
                if n <= 0 or n > 2000:
                    raise ValueError()
                state["recall"] = n
                print(f"OK. recall = {n}")
            except:
                print("Usage: /recall 200   (1~2000)")
            continue

        if cmd == "/show":
            if state["last_df"] is None or len(state["last_df"]) == 0:
                print("No previous results.")
                continue
            df = state["last_df"]
            print(f"\nBot> Showing last results (k={state['k']}, recall={state['recall']})")
            for i, r in df.iterrows():
                print(format_listing_row(r.to_dict(), i + 1))
            continue
        if cmd == "/constraints":
            print(json.dumps(state.get("constraints") or {}, ensure_ascii=False, indent=2))
            continue
        
        if cmd == "/model":
            print(f"QWEN_BASE_URL={QWEN_BASE_URL}")
            print(f"QWEN_MODEL={QWEN_MODEL}")
            continue
        # normal query
        # query = user_in
        # k = int(state["k"])
        # recall = int(state["recall"])
        # df = faiss_search(index, embedder, meta, query=query, recall=recall, k=k)
        extracted = llm_extract(user_in, state["constraints"])
        state["constraints"] = merge_constraints(state["constraints"], extracted)
        state["constraints"] = normalize_budget_to_pcm(state["constraints"])
        
        k = int(state["constraints"].get("k", DEFAULT_K) or DEFAULT_K)
        recall = int(state["recall"])
        
        # --- 2) build retrieval query (user text + constraint hint) ---
        hint = constraints_to_query_hint(state["constraints"])
        query = user_in if not hint else (user_in + " || " + hint)
        
        # --- 3) retrieve ---
        df = faiss_search(index, embedder, meta, query=query, recall=recall)


        # --- Stage 2: hard filters (no fallback) ---
        c = state["constraints"] or {}
        route = decide_route(c)
        route_weights = ROUTE_WEIGHT_TEMPLATES.get(route, ROUTE_WEIGHT_TEMPLATES["HYBRID"])
        w_faiss = float(route_weights["w_faiss"])
        w_pref = float(route_weights["w_pref"])
        w_unknown = float(route_weights["w_unknown"])
        w_soft = float(route_weights["w_soft"])

        filtered = df.copy()
        pre_filter_n = len(filtered)

        # bedrooms: known violations fail; unknown values pass and get penalized in ranking
        if c.get("bedrooms") is not None and "bedrooms" in filtered.columns:
            filtered["bedrooms"] = pd.to_numeric(filtered["bedrooms"], errors="coerce")
            bed_op = str(c.get("bedrooms_op") or "eq").lower()
            if bed_op == "gte":
                keep_mask = filtered["bedrooms"].isna() | (filtered["bedrooms"] >= int(c["bedrooms"]))
            else:
                keep_mask = filtered["bedrooms"].isna() | (filtered["bedrooms"] == int(c["bedrooms"]))
            filtered = filtered[keep_mask]

        # bathrooms: known violations fail; unknown values pass and get penalized in ranking
        if c.get("bathrooms") is not None and "bathrooms" in filtered.columns:
            filtered["bathrooms"] = pd.to_numeric(filtered["bathrooms"], errors="coerce")
            op = str(c.get("bathrooms_op") or "eq").lower()
            if op == "gte":
                keep_mask = filtered["bathrooms"].isna() | (filtered["bathrooms"] >= float(c["bathrooms"]))
            else:
                keep_mask = filtered["bathrooms"].isna() | (filtered["bathrooms"] == float(c["bathrooms"]))
            filtered = filtered[keep_mask]
        
        # budget: known violations fail; unknown values pass and get penalized in ranking
        rent_col = "price_pcm" if "price_pcm" in filtered.columns else ("rent_pcm" if "rent_pcm" in filtered.columns else None)
        if c.get("max_rent_pcm") is not None and rent_col:
            filtered[rent_col] = pd.to_numeric(filtered[rent_col], errors="coerce")
            keep_mask = filtered[rent_col].isna() | (filtered[rent_col] <= float(c["max_rent_pcm"]))
            filtered = filtered[keep_mask]

        # available_from:
        # - "available from DATE" and "available by/before DATE" both map to:
        #   listing available_from <= tenant target date
        if c.get("available_from") is not None and "available_from" in filtered.columns:
            listing_dates = pd.to_datetime(filtered["available_from"], errors="coerce")
            target_date = pd.to_datetime(c["available_from"], errors="coerce")
            if pd.notna(target_date):
                keep_mask = listing_dates.isna() | (listing_dates <= target_date)
                filtered = filtered[keep_mask]

        print(
            f"[debug] route={route}, retrieved={pre_filter_n}, after_hard_filters={len(filtered)}, "
            f"k={k}, recall={recall}, weights={{faiss:{w_faiss}, pref:{w_pref}, unknown:{w_unknown}, soft:{w_soft}}}"
        )

        # --- Stage 3: ranking with unknown penalty + soft preference penalty ---
        if len(filtered) > 0:
            # base score from FAISS similarity (min-max normalized to 0..1)
            if "faiss_score" in filtered.columns:
                fs = pd.to_numeric(filtered["faiss_score"], errors="coerce").fillna(0.0)
            else:
                fs = pd.Series(np.zeros(len(filtered), dtype=float), index=filtered.index)
            fs_min, fs_max = float(fs.min()), float(fs.max())
            if fs_max > fs_min:
                filtered["faiss_score_norm"] = (fs - fs_min) / (fs_max - fs_min)
            else:
                filtered["faiss_score_norm"] = 0.0

            # unknown penalty for hard-constraint fields (only when that constraint is active)
            filtered["unknown_penalty"] = 0.0
            if c.get("bedrooms") is not None:
                if "bedrooms" in filtered.columns:
                    bed_num = pd.to_numeric(filtered["bedrooms"], errors="coerce")
                    filtered.loc[bed_num.isna(), "unknown_penalty"] += UNKNOWN_PENALTY_WEIGHTS["bedrooms"]
                else:
                    filtered["unknown_penalty"] += UNKNOWN_PENALTY_WEIGHTS["bedrooms"]
            if c.get("bathrooms") is not None:
                if "bathrooms" in filtered.columns:
                    bath_num = pd.to_numeric(filtered["bathrooms"], errors="coerce")
                    filtered.loc[bath_num.isna(), "unknown_penalty"] += UNKNOWN_PENALTY_WEIGHTS["bathrooms"]
                else:
                    filtered["unknown_penalty"] += UNKNOWN_PENALTY_WEIGHTS["bathrooms"]
            if c.get("max_rent_pcm") is not None:
                if rent_col and rent_col in filtered.columns:
                    rent_num = pd.to_numeric(filtered[rent_col], errors="coerce")
                    filtered.loc[rent_num.isna(), "unknown_penalty"] += UNKNOWN_PENALTY_WEIGHTS["price"]
                else:
                    filtered["unknown_penalty"] += UNKNOWN_PENALTY_WEIGHTS["price"]
            if c.get("available_from") is not None:
                if "available_from" in filtered.columns:
                    date_num = pd.to_datetime(filtered["available_from"], errors="coerce")
                    filtered.loc[date_num.isna(), "unknown_penalty"] += UNKNOWN_PENALTY_WEIGHTS["available_from"]
                else:
                    filtered["unknown_penalty"] += UNKNOWN_PENALTY_WEIGHTS["available_from"]

            filtered["unknown_penalty"] = filtered["unknown_penalty"].clip(upper=UNKNOWN_PENALTY_CAP)

            # per-row reason strings for unknown penalties
            def _unknown_reasons(row: pd.Series) -> str:
                reasons = []
                if c.get("bedrooms") is not None:
                    if "bedrooms" not in filtered.columns or pd.isna(row.get("bedrooms")):
                        reasons.append(f"bedrooms(+{UNKNOWN_PENALTY_WEIGHTS['bedrooms']})")
                if c.get("bathrooms") is not None:
                    if "bathrooms" not in filtered.columns or pd.isna(row.get("bathrooms")):
                        reasons.append(f"bathrooms(+{UNKNOWN_PENALTY_WEIGHTS['bathrooms']})")
                if c.get("max_rent_pcm") is not None:
                    if (not rent_col) or (rent_col not in filtered.columns) or pd.isna(row.get(rent_col)):
                        reasons.append(f"price(+{UNKNOWN_PENALTY_WEIGHTS['price']})")
                if c.get("available_from") is not None:
                    dt = pd.to_datetime(row.get("available_from"), errors="coerce")
                    if ("available_from" not in filtered.columns) or pd.isna(dt):
                        reasons.append(f"available_from(+{UNKNOWN_PENALTY_WEIGHTS['available_from']})")
                return ", ".join(reasons)

            filtered["unknown_penalty_reasons"] = filtered.apply(_unknown_reasons, axis=1)

            # soft must-have preferences: miss => penalty, hit => bonus
            ###
            must_keywords = [str(x).strip().lower() for x in (c.get("must_have_keywords") or []) if str(x).strip()]
            filtered["preference_score"] = 0.0
            filtered["violation_soft_penalty"] = 0.0
            filtered["soft_penalty_reasons"] = ""

            if must_keywords:
                text_cols = [col for col in ["title", "address", "description", "features", "stations", "schools"] if col in filtered.columns]
                if text_cols:
                    combo = filtered[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                else:
                    combo = pd.Series([""] * len(filtered), index=filtered.index)

                hit_counts = pd.Series(np.zeros(len(filtered), dtype=float), index=filtered.index)
                for kw in must_keywords:
                    hit_counts += combo.str.contains(re.escape(kw), regex=True).astype(float)

                miss_counts = len(must_keywords) - hit_counts
                filtered["preference_score"] = (hit_counts * SOFT_MUST_HIT_BONUS).clip(upper=SOFT_MUST_MAX_BONUS)
                filtered["violation_soft_penalty"] = (miss_counts * SOFT_MUST_MISS_PENALTY).clip(upper=SOFT_MUST_MAX_PENALTY)

                def _soft_reasons(text: str) -> str:
                    misses = []
                    for kw in must_keywords:
                        if not re.search(re.escape(kw), str(text).lower()):
                            misses.append(f"{kw}(+{SOFT_MUST_MISS_PENALTY})")
                    return ", ".join(misses)

                filtered["soft_penalty_reasons"] = combo.apply(_soft_reasons)

            filtered["final_score"] = (
                (w_faiss * filtered["faiss_score_norm"])
                + (w_pref * filtered["preference_score"])
                - (w_unknown * filtered["unknown_penalty"])
                - (w_soft * filtered["violation_soft_penalty"])
            )
            filtered["route"] = route
            filtered["w_faiss"] = w_faiss
            filtered["w_pref"] = w_pref
            filtered["w_unknown"] = w_unknown
            filtered["w_soft"] = w_soft
            filtered = filtered.sort_values(["final_score", "faiss_score_norm"], ascending=False)

        # no fallback: if insufficient, tell user
        k = int(c.get("k", DEFAULT_K) or DEFAULT_K)
        if len(filtered) < k:
            print("\nBot> 符合当前价格/卧室/卫生间/入住时间条件的房源不足。你可以放宽预算或修改约束。")
            df = filtered.reset_index(drop=True)
        else:
            df = filtered.head(k).reset_index(drop=True)


        
        if df is None or len(df) == 0:
            out = "I couldn't find any matching listings. Try different keywords (area, budget, bedrooms, bathrooms, available date)."
            print("\nBot> " + out)
            state["history"].append((user_in, out))
            state["last_query"] = query
            state["last_df"] = df
            continue

        # print results
        lines = [f"Top {min(k, len(df))} results:"]
        for i, r in df.iterrows():
            lines.append(format_listing_row(r.to_dict(), i + 1))
        out = "\n".join(lines)

        print("\nBot> " + out)

        state["history"].append((user_in, out))
        state["last_query"] = query
        state["last_df"] = df


if __name__ == "__main__":
    run_chat()
