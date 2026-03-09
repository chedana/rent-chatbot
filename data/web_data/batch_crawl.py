import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from dataclasses import asdict
from typing import Iterable, List, Optional, Tuple

DEFAULT_URLS_FILE = "listing_urls.txt"
DEFAULT_OUT_JSONL = "properties.jsonl"
DEFAULT_SOURCE_NAME = "rightmove"
DEFAULT_SLEEP_SEC = 1.0
DEFAULT_WORKERS = 1


def normalize_url_item(x) -> str:
    """
    Make sure we always end up with a URL string.
    Handles:
    - normal str
    - tuple like (idx, url) or (url, ...)
    """
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, tuple) and len(x) > 0:
        for item in x[::-1]:
            if isinstance(item, str) and "rightmove.co.uk" in item:
                return item.strip()
        for item in x:
            if isinstance(item, str):
                return item.strip()
    return ""


def read_urls(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            urls.append(s)
    return urls


def crawl_urls(
    urls: Iterable[str],
    out_jsonl: str = DEFAULT_OUT_JSONL,
    source_name: str = DEFAULT_SOURCE_NAME,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    workers: int = DEFAULT_WORKERS,
) -> Tuple[int, int]:
    from extract_one_page import build_record_from_html, fetch_rendered_html_and_nearby

    urls_list = list(urls)
    workers = max(1, workers)

    def process_one(i: int, raw: str) -> Tuple[int, Optional[str], Optional[str]]:
        url = normalize_url_item(raw)
        if not url:
            return i, None, "Empty URL"

        print(f"[{i}/{len(urls_list)}] Extracting: {url}")
        try:
            html, stations, schools = fetch_rendered_html_and_nearby(url)
            rec = build_record_from_html(
                html,
                url=url,
                source=source_name,
                stations=stations,
                schools=schools,
            )
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            return i, json.dumps(asdict(rec), ensure_ascii=False), None
        except Exception as e:
            return i, None, f"Failed: {url}\n  {e}"

    if workers == 1:
        ok, fail = 0, 0
        with open(out_jsonl, "w", encoding="utf-8") as out:
            for i, raw in enumerate(urls_list, 1):
                idx, line, err = process_one(i, raw)
                if line:
                    out.write(line + "\n")
                    out.flush()
                    ok += 1
                else:
                    fail += 1
                    if err:
                        print(err)
        return ok, fail

    ok, fail = 0, 0
    completed = 0
    rows_by_idx: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_one, i, raw) for i, raw in enumerate(urls_list, 1)]
        for fut in as_completed(futures):
            idx, line, err = fut.result()
            completed += 1
            if line:
                rows_by_idx[idx] = line
                ok += 1
            else:
                fail += 1
                if err:
                    print(err)
            print(f"Progress: {completed}/{len(urls_list)} done")

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for idx in sorted(rows_by_idx):
            out.write(rows_by_idx[idx] + "\n")

    return ok, fail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls-file", default=DEFAULT_URLS_FILE, help="Input file containing listing URLs.")
    parser.add_argument("--out-jsonl", default=DEFAULT_OUT_JSONL, help="Output properties JSONL file.")
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME, help="Source label stored in output.")
    parser.add_argument("--sleep-sec", type=float, default=DEFAULT_SLEEP_SEC, help="Delay between listings.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers for detail crawling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    urls = read_urls(args.urls_file)
    print(f"Loaded {len(urls)} urls from {args.urls_file}")

    ok, fail = crawl_urls(
        urls=urls,
        out_jsonl=args.out_jsonl,
        source_name=args.source_name,
        sleep_sec=args.sleep_sec,
        workers=args.workers,
    )
    print(f"Done. OK={ok}, FAIL={fail}")
    print(f"Wrote JSONL to {args.out_jsonl}")


if __name__ == "__main__":
    main()
