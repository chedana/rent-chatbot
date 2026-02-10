import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "web_data")
if WEB_DATA_DIR not in sys.path:
    sys.path.insert(0, WEB_DATA_DIR)

from batch_crawl import crawl_urls  # noqa: E402
from run_pipeline import slugify_area  # noqa: E402


DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "artifacts", "web_data")


@dataclass
class QueryInput:
    method: str
    query: str
    urls_file: str
    slug: Optional[str] = None


def extract_property_id(url: str) -> str:
    m = re.search(r"/properties/(\d+)", url or "")
    return m.group(1) if m else ""


def normalize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    pid = extract_property_id(u)
    if pid:
        return f"https://www.rightmove.co.uk/properties/{pid}#/?channel=RES_LET"
    return u


def listing_id(url: str) -> str:
    pid = extract_property_id(url)
    if pid:
        return f"rightmove:{pid}"
    return f"rightmove:{normalize_url(url)}"


def load_queries(path: str) -> List[QueryInput]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("queries-file must be a non-empty JSON array")

    out: List[QueryInput] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"query[{i}] must be an object")
        method = str(item.get("method", "")).strip().lower()
        query = str(item.get("query", "")).strip()
        urls_file = str(item.get("urls_file", "")).strip()
        slug = str(item.get("slug", "")).strip() or None
        if method not in {"station", "region"}:
            raise ValueError(f"query[{i}].method must be station or region")
        if not query:
            raise ValueError(f"query[{i}].query is required")
        if not urls_file:
            raise ValueError(f"query[{i}].urls_file is required")
        out.append(QueryInput(method=method, query=query, urls_file=urls_file, slug=slug))
    return out


def read_urls(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = normalize_url(line.strip())
            if u:
                urls.append(u)
    return urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build global dedup dataset from existing query URL files and optional detail crawl.",
    )
    parser.add_argument("--queries-file", required=True, help="JSON file: [{method,query,urls_file,slug?}, ...].")
    parser.add_argument("--run-name", required=True, help="Output run name under output-root.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Output root.")
    parser.add_argument(
        "--index-workers",
        type=int,
        default=4,
        help="Workers for preprocessing query URL files (read/normalize/query-level dedupe).",
    )
    parser.add_argument("--crawl-details", action="store_true", help="Crawl dedup URLs and output full properties.")
    parser.add_argument("--source-name", default="rightmove", help="Source name for detail crawl.")
    parser.add_argument("--sleep-sec", type=float, default=1.0, help="Delay between detail crawls.")
    parser.add_argument("--crawl-workers", type=int, default=4, help="Detail crawl workers.")
    return parser.parse_args()


def process_one_query(q: QueryInput) -> Dict:
    urls = read_urls(q.urls_file)
    seen: Set[str] = set()
    unique_ids_in_query: List[str] = []
    query_lid_to_url: Dict[str, str] = {}

    for u in urls:
        lid = listing_id(u)
        if lid in seen:
            continue
        seen.add(lid)
        unique_ids_in_query.append(lid)
        query_lid_to_url[lid] = u

    slug = q.slug or slugify_area(f"{q.method}_{q.query}")
    return {
        "query": q,
        "slug": slug,
        "raw_url_count": len(urls),
        "unique_ids_in_query": unique_ids_in_query,
        "query_lid_to_url": query_lid_to_url,
    }


def main() -> None:
    args = parse_args()
    queries = load_queries(args.queries_file)

    run_dir = os.path.join(args.output_root, slugify_area(args.run_name))
    os.makedirs(run_dir, exist_ok=True)
    query_indexes_dir = os.path.join(run_dir, "query_indexes")
    os.makedirs(query_indexes_dir, exist_ok=True)

    global_map: Dict[str, Dict] = {}
    summary_queries: List[Dict] = []

    workers = max(1, int(args.index_workers))
    results: List[Dict] = []
    if workers == 1:
        for q in queries:
            results.append(process_one_query(q))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_one_query, q) for q in queries]
            for fut in as_completed(futures):
                results.append(fut.result())

    # Keep output ordering stable according to input query list.
    order = {(q.method, q.query, q.urls_file): i for i, q in enumerate(queries)}
    results.sort(key=lambda x: order[(x["query"].method, x["query"].query, x["query"].urls_file)])

    for item in results:
        q: QueryInput = item["query"]
        slug: str = item["slug"]
        unique_ids_in_query: List[str] = item["unique_ids_in_query"]
        query_lid_to_url: Dict[str, str] = item["query_lid_to_url"]

        for lid in unique_ids_in_query:
            u = query_lid_to_url[lid]
            if lid not in global_map:
                global_map[lid] = {
                    "listing_id": lid,
                    "url": u,
                    "source_site": "rightmove",
                    "discovery_queries_by_method": {"station": [], "region": []},
                }
            rec = global_map[lid]
            if q.query not in rec["discovery_queries_by_method"][q.method]:
                rec["discovery_queries_by_method"][q.method].append(q.query)

        idx_path = os.path.join(query_indexes_dir, f"{slug}.jsonl")
        with open(idx_path, "w", encoding="utf-8") as f:
            for lid in unique_ids_in_query:
                f.write(
                    json.dumps(
                        {"listing_id": lid, "query_method": q.method, "query": q.query},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        summary_queries.append(
            {
                "slug": slug,
                "method": q.method,
                "query": q.query,
                "raw_url_count": item["raw_url_count"],
                "unique_listing_count": len(unique_ids_in_query),
                "index_file": idx_path,
            }
        )

    for rec in global_map.values():
        by_method = rec["discovery_queries_by_method"]
        rec["discovery_queries_by_method"] = {k: v for k, v in by_method.items() if v}

    global_simulated = os.path.join(run_dir, "global_listings_simulated.jsonl")
    dedup_urls = os.path.join(run_dir, "dedup_urls_to_crawl.txt")
    with open(global_simulated, "w", encoding="utf-8") as fout, open(dedup_urls, "w", encoding="utf-8") as fu:
        for lid in sorted(global_map.keys()):
            rec = global_map[lid]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fu.write(rec["url"] + "\n")

    summary = {
        "run_name": args.run_name,
        "query_count": len(queries),
        "global_unique_listing_count": len(global_map),
        "queries": summary_queries,
        "files": {
            "global_simulated": global_simulated,
            "dedup_urls_to_crawl": dedup_urls,
            "query_indexes_dir": query_indexes_dir,
        },
    }

    if args.crawl_details:
        raw_details = os.path.join(run_dir, "properties_raw_deduped.jsonl")
        ok, fail = crawl_urls(
            urls=[global_map[lid]["url"] for lid in sorted(global_map.keys())],
            out_jsonl=raw_details,
            source_name=args.source_name,
            sleep_sec=args.sleep_sec,
            workers=max(1, args.crawl_workers),
        )

        final_out = os.path.join(run_dir, "global_properties_with_discovery.jsonl")
        written = 0
        with open(raw_details, "r", encoding="utf-8") as fin, open(final_out, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                u = normalize_url(str(rec.get("url", "") or ""))
                lid = listing_id(u)
                rec["url"] = u
                rec["listing_id"] = lid
                rec["source_site"] = str(rec.get("source", args.source_name) or args.source_name)
                rec["discovery_queries_by_method"] = global_map.get(lid, {}).get(
                    "discovery_queries_by_method",
                    {},
                )
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

        summary["crawl"] = {
            "ok": ok,
            "fail": fail,
            "raw_details": raw_details,
            "global_properties_with_discovery": final_out,
            "written": written,
        }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
