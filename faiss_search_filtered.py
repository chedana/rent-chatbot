import re
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

INDEX_PATH = "rentbot_data/index/listings.faiss"
META_PATH  = "rentbot_data/index/meta.parquet"

index = faiss.read_index(INDEX_PATH)
meta = pd.read_parquet(META_PATH)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def parse_hard_constraints(q: str):
    ql = q.lower()

    # bedrooms: "1 bed", "2 bedroom"
    beds = None
    m = re.search(r"\b(\d)\s*(bed|bedroom)\b", ql)
    if m:
        beds = int(m.group(1))

    # budget: "under 2500", "max 2000", "<= 1800", "£2500"
    max_rent = None
    m = re.search(r"\b(under|max|<=)\s*£?\s*(\d{3,5})\b", ql)
    if m:
        max_rent = float(m.group(2))
    else:
        m = re.search(r"£\s*(\d{3,5})\b", ql)
        if m:
            max_rent = float(m.group(1))

    return beds, max_rent

def search(query: str, k: int = 5, recall: int = 50):
    beds, max_rent = parse_hard_constraints(query)

    q_emb = model.encode(query, normalize_embeddings=True).astype("float32")[None, :]
    scores, ids = index.search(q_emb, recall)

    ids = ids[0].tolist()
    scores = scores[0].tolist()

    df = meta.iloc[ids].copy()
    df["score"] = scores

    # hard filters
    if beds is not None:
        df = df[df["bedrooms"] == beds]
    if max_rent is not None:
        df = df[df["rent_pcm"] <= max_rent]

    df = df.sort_values("score", ascending=False).head(k)

    print(f"\nParsed constraints: bedrooms={beds}, max_rent_pcm={max_rent}")
    print(df[["score","rent_pcm","bedrooms","location","title","url"]].to_string(index=False))

if __name__ == "__main__":
    print("Type query (Ctrl+C to exit)")
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            continue
        search(q, k=5, recall=50)
