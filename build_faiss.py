import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

DATA = Path("rentbot_data/processed/zoopla_london_listings.parquet")
OUT  = Path("rentbot_data/index")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = df["rag_text"].fillna("").astype(str).tolist()

embs = []
for t in tqdm(texts, desc="Embedding"):
    embs.append(model.encode(t, normalize_embeddings=True))

X = np.vstack(embs).astype("float32")

index = faiss.IndexFlatIP(X.shape[1])
index.add(X)

faiss.write_index(index, str(OUT / "listings.faiss"))
df[["rent_pcm","bedrooms","bathrooms","location","title","url","rag_text"]].to_parquet(OUT / "meta.parquet", index=False)

print("OK")
print("rows:", len(df))
print("dim:", X.shape[1])
print("saved:", OUT / "listings.faiss", "and", OUT / "meta.parquet")
