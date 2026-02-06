import pandas as pd
from pathlib import Path

RAW = Path("rentbot_data/raw/ons/iphrp.csv")
OUT = Path("rentbot_data/processed")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

df = df.rename(columns=lambda x: x.strip())
df = df[df["Geography"].str.contains("London", na=False)]

df = df[[
    "Date",
    "Geography",
    "Index"
]]

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

df.to_parquet(OUT / "ons_london_rent_index.parquet", index=False)

print("Saved ONS London rent index:", len(df))
