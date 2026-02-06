import re
import pandas as pd
from pathlib import Path

RAW = Path("rentbot_data/raw/listings/listings.csv")
OUT = Path("rentbot_data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def to_float_money(s):
    if s is None:
        return None
    s = str(s)
    s = s.replace(",", "")
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None

rows = []
with open(RAW, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")

        # Heuristic: Zoopla rows often look like:
        # url | £xxxx pcm | £xxx pw | bedrooms | bathrooms | ... | title | location | nearby stations | listed date | agent | available date?
        url = parts[0].strip() if len(parts) > 0 else ""

        rent_pcm = None
        rent_pw = None
        bedrooms = None
        bathrooms = None
        title = ""
        location = ""
        nearby = ""
        listed = ""
        agent = ""
        available = ""

        # Find money fields
        for p in parts:
            pl = p.lower()
            if "pcm" in pl and rent_pcm is None:
                rent_pcm = to_float_money(p)
            if "pw" in pl and rent_pw is None:
                rent_pw = to_float_money(p)

        # Bedrooms / bathrooms: often small ints early in the row
        # We take the first two integer-like fields after the money fields, if present
        ints = []
        for p in parts:
            p2 = p.strip()
            if re.fullmatch(r"\d+", p2):
                ints.append(int(p2))
        if len(ints) >= 2:
            bedrooms, bathrooms = ints[0], ints[1]
        elif len(ints) == 1:
            bedrooms = ints[0]

        # Title: usually contains "to rent"
        for p in parts:
            if " to rent" in p.lower():
                title = p.strip()
                break

        # Location: often contains ", London"
        for p in parts:
            if ", London" in p or "London " in p:
                location = p.strip()
                break

        # Nearby stations: often contains "miles" or station-like pattern
        for p in parts:
            if "mile" in p.lower() or "station" in p.lower():
                nearby = p.strip()
                break

        # Listed date / agent / available: near the end usually
        if len(parts) >= 3:
            tail = [x.strip() for x in parts[-5:]]
            # pick first tail item that looks like a date (e.g., "4th jan 2023")
            for t in tail:
                if re.search(r"\b\d{1,2}(st|nd|rd|th)?\s+[a-z]{3}\s+\d{4}\b", t.lower()):
                    listed = t
                    break
            # agent often has a comma and postcode-like suffix, or just a name
            if tail:
                agent = tail[-2] if len(tail) >= 2 else ""
                available = tail[-1] if len(tail) >= 1 else ""

        rows.append({
            "url": url,
            "rent_pcm": rent_pcm,
            "rent_pw": rent_pw,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "title": title,
            "location": location,
            "nearby": nearby,
            "listed_date": listed,
            "agent": agent,
            "available_date": available,
            "raw": line
        })

df = pd.DataFrame(rows)

# Basic cleaning
df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(0).astype(int)
df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
df = df[df["url"].astype(str).str.startswith("http")]

# Keep plausible rents
df = df[df["rent_pcm"].notna()]
df = df[(df["rent_pcm"] >= 200) & (df["rent_pcm"] <= 20000)]

# Build RAG text
def build_text(r):
    parts = []
    if r["location"]:
        parts.append(f"Location: {r['location']}")
    parts.append(f"Bedrooms: {int(r['bedrooms'])}")
    if pd.notna(r["bathrooms"]):
        parts.append(f"Bathrooms: {r['bathrooms']}")
    parts.append(f"Rent: £{int(r['rent_pcm'])} per month (PCM)")
    if r["title"]:
        parts.append(f"Listing: {r['title']}")
    if r["nearby"]:
        parts.append(f"Nearby: {r['nearby']}")
    if r["listed_date"]:
        parts.append(f"Listed: {r['listed_date']}")
    if r["agent"]:
        parts.append(f"Agent: {r['agent']}")
    parts.append(f"URL: {r['url']}")
    return "\n".join(parts)

df["rag_text"] = df.apply(build_text, axis=1)

out_path = OUT / "zoopla_london_listings.parquet"
df.to_parquet(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(df))
print(df[["rent_pcm","bedrooms","location","title","url"]].head(5).to_string(index=False))
