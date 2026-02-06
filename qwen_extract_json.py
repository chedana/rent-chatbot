import os, re, json
from openai import OpenAI

BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("QWEN_MODEL", "/workspace/Qwen2.5-14B-Instruct")
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM = """You output STRICT JSON only (no markdown, no explanation, no extra text).
If a field is unknown, use null. Use integers for bedrooms.
Schema:
{
  "max_rent_pcm": number|null,
  "bedrooms": int|null,
  "location_keywords": string[],
  "must_have_keywords": string[],
  "k": int
}
"""

def extract(user_query: str):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":user_query}
        ],
        temperature=0
    )
    txt = r.choices[0].message.content.strip()

    # Safety: keep only JSON object if model adds anything
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        raise ValueError("No JSON found. Got:\n" + txt)
    obj = json.loads(m.group(0))

    # Defaults
    if "k" not in obj or obj["k"] is None:
        obj["k"] = 5
    if "location_keywords" not in obj or obj["location_keywords"] is None:
        obj["location_keywords"] = []
    if "must_have_keywords" not in obj or obj["must_have_keywords"] is None:
        obj["must_have_keywords"] = []

    return obj, txt

if __name__ == "__main__":
    tests = [
        "Canary Wharf 附近 1 bed，预算 2500 以内，最好靠近地铁，有阳台更好",
        "我想在 N7 或 Islington，1-2居，2000pcm以内，furnished",
        "预算1800，想要Holloway一居室，离地铁近"
    ]

    for q in tests:
        obj, raw = extract(q)
        print("\nQUERY:", q)
        print("JSON :", json.dumps(obj, ensure_ascii=False))
