import os, re, json
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI

BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("QWEN_MODEL", "/workspace/Qwen2.5-14B-Instruct")
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")

INDEX_PATH = "rentbot_data/index/listings.faiss"
META_PATH  = "rentbot_data/index/meta.parquet"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
index = faiss.read_index(INDEX_PATH)
meta = pd.read_parquet(META_PATH)
embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

EXTRACT_SYSTEM = """You output STRICT JSON only (no markdown, no explanation).
Schema:
{
  "max_rent_pcm": number|null,
  "bedrooms": int|null,
  "location_keywords": string[],
  "must_have_keywords": string[],
  "k": int
}
If unknown use null or [].
"""

ANSWER_SYSTEM = """You are a London rental assistant.
You will be given:
- user request
- extracted constraints
- candidate listings with computed facts (e.g., within_budget true/false)
Rules:
- Do NOT invent facts.
- Use the provided computed facts for budget compliance.
- Recommend best options with brief reasons.
- Always include rent_pcm, bedrooms, location, and url.
- Keep it concise.
"""

NEAR_WORDS = {
    "near subway","near station","near tube","tube","subway","station","close to station","near metro"
}

def _chat(messages, temperature=0.3):
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature
    )
    return r.choices[0].message.content.strip()

def _llm_extract(q: str) -> dict:
    txt = _chat(
        [{"role":"system","content":EXTRACT_SYSTEM},
         {"role":"user","content":q}],
        temperature=0
    )
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError("No JSON found. Got:\n" + txt)
    obj = json.loads(m.group(0))
    obj.setdefault("k", 5)
    obj.setdefault("location_keywords", [])
    obj.setdefault("must_have_keywords", [])
    return obj

def _normalize_constraints(c: dict) -> dict:
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
    if c.get("k") is None:
        c["k"] = 5
    return c

def _faiss_recall(query: str, recall: int = 200) -> pd.DataFrame:
    q_emb = embed.encode(query, normalize_embeddings=True).astype("float32")[None, :]
    scores, ids = index.search(q_emb, recall)
    ids = ids[0].tolist()
    scores = scores[0].tolist()
    df = meta.iloc[ids].copy()
    df["score"] = scores
    return df

def _apply_filters(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    out = df.copy()

    if c.get("bedrooms") is not None:
        out = out[out["bedrooms"] == int(c["bedrooms"])]

    if c.get("max_rent_pcm") is not None:
        out = out[out["rent_pcm"] <= float(c["max_rent_pcm"])]

    locs = c.get("location_keywords") or []
    if locs:
        pat = "|".join([re.escape(x) for x in locs])
        out = out[out["location"].astype(str).str.contains(pat, case=False, na=False)]

    must = c.get("must_have_keywords") or []
    if must:
        pat = "|".join([re.escape(x) for x in must])
        out = out[out["rag_text"].astype(str).str.contains(pat, case=False, na=False)]

    return out

def _build_listing_payload(df: pd.DataFrame, c: dict):
    max_rent = c.get("max_rent_pcm", None)
    items = []
    for _, r in df.iterrows():
        within_budget = None
        budget_gap = None
        if max_rent is not None:
            within_budget = float(r["rent_pcm"]) <= float(max_rent)
            budget_gap = float(r["rent_pcm"]) - float(max_rent)

        items.append({
            "rent_pcm": float(r["rent_pcm"]),
            "bedrooms": int(r["bedrooms"]),
            "location": str(r["location"]),
            "title": str(r["title"]),
            "url": str(r["url"]),
            "score": float(r["score"]),
            "within_budget": within_budget,
            "budget_gap_pcm": budget_gap
        })
    return items

def _answer(user_q: str, c: dict, items: list[dict]):
    return _chat(
        [
            {"role":"system","content":ANSWER_SYSTEM},
            {"role":"user","content":"User request:\n" + user_q},
            {"role":"user","content":"Extracted constraints (JSON):\n" + json.dumps(c, ensure_ascii=False)},
            {"role":"user","content":"Candidate listings (JSON):\n" + json.dumps(items, ensure_ascii=False)},
            {"role":"user","content":"Recommend the best options with brief reasons. Use within_budget/budget_gap_pcm fields for budget statements."}
        ],
        temperature=0.3
    )

def run_search(user_q: str) -> str:
    c = _normalize_constraints(_llm_extract(user_q))
    k = int(c.get("k", 5) or 5)

    cand = _faiss_recall(user_q, recall=200)
    filtered = _apply_filters(cand, c)

    if len(filtered) < k and (c.get("must_have_keywords") or []):
        c2 = dict(c)
        c2["must_have_keywords"] = []
        filtered = _apply_filters(cand, c2)

    filtered = filtered.sort_values("score", ascending=False).head(k)
    items = _build_listing_payload(filtered, c)

    return _answer(user_q, c, items)
