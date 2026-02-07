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

def llm_extract(q: str) -> dict:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":EXTRACT_SYSTEM},
                  {"role":"user","content":q}],
        temperature=0
    )
    txt = r.choices[0].message.content.strip()
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError("No JSON found. Got:\n" + txt)
    obj = json.loads(m.group(0))
    obj.setdefault("k", 5)
    obj.setdefault("location_keywords", [])
    obj.setdefault("must_have_keywords", [])
    return obj

NEAR_WORDS = {
    "near subway","near station","near tube","tube","subway","station","close to station","near metro"
}

def normalize_constraints(c: dict) -> dict:
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

def faiss_recall(query: str, recall: int = 120) -> pd.DataFrame:
    q_emb = embed.encode(query, normalize_embeddings=True).astype("float32")[None, :]
    scores, ids = index.search(q_emb, recall)
    ids = ids[0].tolist()
    scores = scores[0].tolist()
    df = meta.iloc[ids].copy()
    df["score"] = scores
    return df

def apply_filters(df: pd.DataFrame, c: dict) -> pd.DataFrame:
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

def run_query(q: str):
    c = llm_extract(q)
    c = normalize_constraints(c)
    k = int(c.get("k", 5) or 5)

    cand = faiss_recall(q, recall=200)
    filtered = apply_filters(cand, c)

    # 如果太少：去掉 must-have 再试一次（保留预算/卧室/地点）
    if len(filtered) < k and (c.get("must_have_keywords") or []):
        c2 = dict(c)
        c2["must_have_keywords"] = []
        filtered = apply_filters(cand, c2)

    filtered = filtered.sort_values("score", ascending=False).head(k)

    print("\nExtracted constraints:", json.dumps(c, ensure_ascii=False))
    if len(filtered) == 0:
        print("No results after filtering.")
        return
    print(filtered[["score","rent_pcm","bedrooms","location","title","url"]].to_string(index=False))

if __name__ == "__main__":
    print("Type query (exit to quit)")
    while True:
        q = input("\nUser> ").strip()
        if not q or q.lower() in ["exit","quit"]:
            break
        run_query(q)
