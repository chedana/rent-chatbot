import os, json
from openai import OpenAI

BASE_URL = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("QWEN_MODEL", "/workspace/Qwen2.5-14B-Instruct")
API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM = """You are a London rental assistant.
Given a user request and a small list of candidate listings, recommend the best options.
Rules:
- Do NOT invent facts.
- Base reasons strictly on provided fields.
- Be concise and practical.
- Always include the URL for each listing.
"""

def generate(user_query: str, listings: list[dict]):
    messages = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":"User request:\n" + user_query},
        {"role":"user","content":"Candidate listings (JSON):\n" + json.dumps(listings, ensure_ascii=False)},
        {"role":"user","content":"Recommend these listings with brief reasons."}
    ]
    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3
    )
    return r.choices[0].message.content.strip()

if __name__ == "__main__":
    # Minimal test with hard-coded listings (from your last output)
    user_q = "Canary Wharf 附近 1 bed，预算 2400 以内，最好靠近地铁，有阳台更好"

    listings = [
        {
            "rent_pcm": 1850,
            "bedrooms": 1,
            "location": "Sailors House, Canary Wharf, London E14",
            "title": "1 bed flat to rent",
            "url": "http://zoopla.co.uk//to-rent/details/63569966/"
        },
        {
            "rent_pcm": 2383,
            "bedrooms": 1,
            "location": "26 Hertsmere Road, Canary Wharf, London E14",
            "title": "1 bed flat to rent",
            "url": "http://zoopla.co.uk//to-rent/details/63565942/"
        }
    ]

    print(generate(user_q, listings))
