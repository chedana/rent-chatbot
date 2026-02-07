# import json
# import time
# from dataclasses import asdict
# from typing import List

# # ✅ CHANGE THIS import to match your file name that contains:
# # - fetch_rendered_html(url) -> html
# # - build_record_from_html(html, url, source=...) -> PropertyRecord
# # - (optional) upsert_sqlite(db_path, rec)
# from extract_one_page import fetch_rendered_html_and_nearby, build_record_from_html


# URLS_FILE = "listing_urls.txt"
# OUT_JSONL = "properties.jsonl"
# DB_PATH = None
# SOURCE_NAME = "rightmove"
# SLEEP_SEC = 1.0              # polite delay


# def read_urls(path: str) -> List[str]:
#     with open(path, "r", encoding="utf-8") as f:
#         return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


# def main():
#     urls = read_urls(URLS_FILE)
#     print(f"Loaded {len(urls)} URLs")

#     ok, fail = 0, 0

#     with open(OUT_JSONL, "w", encoding="utf-8") as out:
#         for i, url in enumerate(urls, 1):
#             try:
#                 print(f"[{i}/{len(urls)}] Extracting: {url}")

#                 html = fetch_rendered_html_and_nearby(url)
#                 rec = build_record_from_html(html, url=url, source=SOURCE_NAME)

#                 # write jsonl
#                 out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
#                 out.flush()

#                 # # optional: write to sqlite
#                 # if DB_PATH:
#                 #     upsert_sqlite(DB_PATH, rec)

#                 ok += 1
#                 time.sleep(SLEEP_SEC)

#             except Exception as e:
#                 fail += 1
#                 print(f"❌ Failed: {url}\n   {e}")

#     print(f"Done. OK={ok}, FAIL={fail}")
#     print(f"Wrote JSONL to {OUT_JSONL}")
#     if DB_PATH:
#         print(f"Saved/updated SQLite to {DB_PATH}")


# if __name__ == "__main__":
#     main()
import json
import time
from dataclasses import asdict

# ✅ import your latest extractor
from extract_one_page import fetch_rendered_html_and_nearby, build_record_from_html

URLS_FILE = "listing_urls.txt"
OUT_JSONL = "properties.jsonl"
SOURCE_NAME = "rightmove"
SLEEP_SEC = 1.0


def normalize_url_item(x):
    """
    Make sure we always end up with a URL string.
    Handles:
      - normal str
      - tuple like (idx, url) or (url, ...)
    """
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, tuple) and len(x) > 0:
        # common cases: (idx, url) or (url, something)
        for item in x[::-1]:
            if isinstance(item, str) and "rightmove.co.uk" in item:
                return item.strip()
        # fallback: first str element
        for item in x:
            if isinstance(item, str):
                return item.strip()
    return ""


def read_urls(path: str):
    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            urls.append(s)
    return urls


def main():
    urls = read_urls(URLS_FILE)
    print(f"Loaded {len(urls)} urls from {URLS_FILE}")

    ok, fail = 0, 0

    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for i, raw in enumerate(urls, 1):
            url = normalize_url_item(raw)
            if not url:
                continue

            print(f"[{i}/{len(urls)}] Extracting: {url}")

            try:
                html, stations, schools = fetch_rendered_html_and_nearby(url)
                rec = build_record_from_html(
                    html,
                    url=url,
                    source=SOURCE_NAME,
                    stations=stations,
                    schools=schools,
                )

                out.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                out.flush()

                ok += 1
                time.sleep(SLEEP_SEC)

            except Exception as e:
                fail += 1
                print(f"❌ Failed: {url}\n   {e}")

    print(f"Done. OK={ok}, FAIL={fail}")
    print(f"Wrote JSONL to {OUT_JSONL}")


if __name__ == "__main__":
    main()