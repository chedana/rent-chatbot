import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "rentbot_data/index/listings.faiss"
META_PATH  = "rentbot_data/index/meta.parquet"

index = faiss.read_index(INDEX_PATH)
meta = pd.read_parquet(META_PATH)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def search(query: str, k: int = 5, max_rent: float | None = None, beds: int | None = None):
    q = query.strip()
    if not q:
        return

    q_emb = model.encode(q, normalize_embeddings=True).astype("float32")[None, :]

    # vector search
    scores, ids = index.search(q_emb, k=50)
    ids = ids[0].tolist()
    scores = scores[0].tolist()

    df = meta.iloc[ids].copy()
    df["score"] = scores

    # optional structured filters after retrieval
    if max_rent is not None:
        df = df[df["rent_pcm"] <= max_rent]
    if beds is not None:
        df = df[df["bedrooms"] == beds]

    df = df.sort_values("score", ascending=False).head(k)

    print(df[["score","rent_pcm","bedrooms","location","title","url"]].to_string(index=False))

if __name__ == "__main__":
    print("Type a query. Examples:")
    print("- 1 bed flat near Canary Wharf, budget 2500")
    print("- Holloway Road N7 2 bed")
    print("- near tube station furnished")
    print("Ctrl+C to exit.\n")

    while True:
        q = input("Query> ")
        search(q, k=5)
