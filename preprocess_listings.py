import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW = Path("rentbot_data/raw/inside_airbnb/london_listings.csv.gz")
OUT = Path("rentbot_data/processed")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading CSV...")
df = pd.read_csv(RAW)

print("Total rows:", len(df))

keep_cols = [
            "id",
                "name",
                    "description",
                        "neighbourhood_cleansed",
                            "latitude",
                                "longitude",
                                    "room_type",
                                        "accommodates",
                                            "bedrooms",
                                                "bathrooms_text",
                                                    "price"
                                                    ]

df = df[keep_cols]

df["price"] = (
            df["price"]
                .astype(str)
                    .str.replace(r"[£$,]", "", regex=True)
                        .astype(float)
                        )

df = df[(df["price"] > 300) & (df["price"] < 10000)]

df["bedrooms"] = df["bedrooms"].fillna(0).astype(int)
df["bathrooms_text"] = df["bathrooms_text"].fillna("unknown")
df["description"] = df["description"].fillna("")

def build_rag_text(row):
        return (
                        f"Location: {row.neighbourhood_cleansed}, London.\n"
                                f"Property type: {row.room_type}.\n"
                                        f"Bedrooms: {row.bedrooms}. Bathrooms: {row.bathrooms_text}.\n"
                                                f"Accommodates: {row.accommodates} people.\n"
                                                        f"Monthly rent: £{int(row.price)}.\n"
                                                                f"Description: {row.description}"
                                                                    )

        tqdm.pandas()
        df["rag_text"] = df.progress_apply(build_rag_text, axis=1)

        df_out = df[
                    [
                                "id",
                                        "neighbourhood_cleansed",
                                                "price",
                                                        "bedrooms",
                                                                "latitude",
                                                                        "longitude",
                                                                                "rag_text",
                                                                                    ]
                    ]

        out_path = OUT / "listings.parquet"
        df_out.to_parquet(out_path, index=False)

        print(f"Saved {len(df_out)} listings to {out_path}")
