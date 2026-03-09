import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEB_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "web_data")
if WEB_DATA_DIR not in sys.path:
    sys.path.insert(0, WEB_DATA_DIR)

from batch_crawl import crawl_urls  # noqa: E402
from get_urls import (  # noqa: E402
    DEFAULT_RESULTS_PER_PAGE,
    build_rightmove_search_url,
    collect_pages_parallel,
    resolve_location_identifiers,
)
from run_pipeline import (  # noqa: E402
    DEFAULT_DISPLAY_LOCATION_IDENTIFIER,
    DEFAULT_LOCATION_IDENTIFIER,
    slugify_area,
)


DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "artifacts", "web_data")


@dataclass
class QuerySpec:
    method: str
    query: str
    search_url: str = ""
    min_bedrooms: Optional[int] = None
    max_bedrooms: Optional[int] = None
    radius: float = 0.0
    sort_type: int = 6
    location_identifier: str = ""
    display_location_identifier: str = ""
    exclude_let_agreed: bool = False


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


def listing_key_from_url(url: str) -> str:
    pid = extract_property_id(url)
    if pid:
        return f"rightmove:{pid}"
    return f"rightmove:{normalize_url(url)}"


def load_query_specs(path: str) -> List[QuerySpec]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("queries-file must be a non-empty JSON array")

    out: List[QuerySpec] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"query[{i}] must be an object")
        method = str(item.get("method", "")).strip().lower()
        query = str(item.get("query", "")).strip()
        if method not in {"station", "region"}:
            raise ValueError(f"query[{i}].method must be station or region")
        if not query:
            raise ValueError(f"query[{i}].query is required")

        out.append(
            QuerySpec(
                method=method,
                query=query,
                search_url=str(item.get("search_url", "") or "").strip(),
                min_bedrooms=item.get("min_bedrooms"),
                max_bedrooms=item.get("max_bedrooms"),
                radius=float(item.get("radius", 0.0)),
                sort_type=int(item.get("sort_type", 6)),
                location_identifier=str(item.get("location_identifier", "") or "").strip(),
                display_location_identifier=str(item.get("display_location_identifier", "") or "").strip(),
                exclude_let_agreed=bool(item.get("exclude_let_agreed", False)),
            )
        )
    return out


def build_search_url(spec: QuerySpec) -> str:
    if spec.search_url:
        return spec.search_url

    location_identifier = spec.location_identifier
    display_location_identifier = spec.display_location_identifier

    if not location_identifier or not display_location_identifier:
        resolved_li, resolved_dli = resolve_location_identifiers(
            search_location=spec.query,
            radius=spec.radius,
        )
        location_identifier = location_identifier or (resolved_li or "")
        display_location_identifier = display_location_identifier or (resolved_dli or "")

        if not location_identifier or not display_location_identifier:
            if "canary wharf" in spec.query.lower():
                location_identifier = location_identifier or DEFAULT_LOCATION_IDENTIFIER
                display_location_identifier = display_location_identifier or DEFAULT_DISPLAY_LOCATION_IDENTIFIER
            else:
                raise RuntimeError(
                    f"could not resolve identifiers for query='{spec.query}'. "
                    "Pass search_url or explicit identifiers."
                )

    return build_rightmove_search_url(
        search_location=spec.query,
        min_bedrooms=spec.min_bedrooms,
        max_bedrooms=spec.max_bedrooms,
        radius=spec.radius,
        include_let_agreed=not spec.exclude_let_agreed,
        sort_type=spec.sort_type,
        location_identifier=location_identifier,
        display_location_identifier=display_location_identifier,
    )


def serialize_paths(path_set: Set[Tuple[str, str]]) -> List[Dict[str, str]]:
    rows = sorted(path_set, key=lambda x: (x[0], x[1]))
    return [{"method": method, "query": query} for method, query in rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect URLs from station/region queries, dedupe first, then crawl once and emit JSONL with discovery_paths.",
    )
    parser.add_argument("--queries-file", required=True, help="Path to query specs JSON file.")
    parser.add_argument("--run-name", default="discovery_run", help="Output folder name.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root output directory.")
    parser.add_argument("--start-page", type=int, default=0, help="Search start page (inclusive).")
    parser.add_argument("--pages", type=int, default=3, help="Number of pages per query.")
    parser.add_argument("--end-page", type=int, default=None, help="Search end page (exclusive).")
    parser.add_argument("--per-page", type=int, default=DEFAULT_RESULTS_PER_PAGE, help="Results per page.")
    parser.add_argument("--collect-workers", type=int, default=1, help="Workers for search pages.")
    parser.add_argument("--crawl-workers", type=int, default=4, help="Workers for detail crawl.")
    parser.add_argument("--sleep-sec", type=float, default=1.0, help="Delay between detail crawls.")
    parser.add_argument("--source-name", default="rightmove", help="Source label.")
    parser.add_argument("--max-listings-per-query", type=int, default=0, help="Cap URLs per query. 0 means no cap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_page < 0:
        raise ValueError("--start-page must be >= 0")
    end_page = args.end_page if args.end_page is not None else args.start_page + max(0, args.pages)
    if end_page < args.start_page:
        raise ValueError("--end-page must be >= --start-page")

    specs = load_query_specs(args.queries_file)
    run_dir = os.path.join(args.output_root, slugify_area(args.run_name))
    os.makedirs(run_dir, exist_ok=True)
    query_urls_dir = os.path.join(run_dir, "query_urls")
    os.makedirs(query_urls_dir, exist_ok=True)

    key_to_url: Dict[str, str] = {}
    key_to_paths: Dict[str, Set[Tuple[str, str]]] = {}
    per_query_stats: List[Dict[str, int]] = []

    for idx, spec in enumerate(specs, 1):
        search_url = build_search_url(spec)
        print(f"[{idx}/{len(specs)}] method={spec.method} query={spec.query}")
        print(f"search_url={search_url}")
        urls = collect_pages_parallel(
            search_url=search_url,
            start_page=args.start_page,
            end_page=end_page,
            per_page=args.per_page,
            workers=max(1, args.collect_workers),
        )

        if args.max_listings_per_query and args.max_listings_per_query > 0:
            urls = urls[: args.max_listings_per_query]

        query_slug = slugify_area(f"{spec.method}_{spec.query}")
        query_urls_file = os.path.join(query_urls_dir, f"{query_slug}.txt")

        raw_count = 0
        with open(query_urls_file, "w", encoding="utf-8") as f:
            for u in urls:
                nu = normalize_url(u)
                if not nu:
                    continue
                raw_count += 1
                f.write(nu + "\n")
                key = listing_key_from_url(nu)
                if key not in key_to_url:
                    key_to_url[key] = nu
                if key not in key_to_paths:
                    key_to_paths[key] = set()
                key_to_paths[key].add((spec.method, spec.query))

        per_query_stats.append({"raw_urls": raw_count})
        print(f"saved raw URLs: {query_urls_file} (count={raw_count})")

    unique_urls = [key_to_url[k] for k in sorted(key_to_url.keys())]
    unique_urls_file = os.path.join(run_dir, "listing_urls_deduped.txt")
    with open(unique_urls_file, "w", encoding="utf-8") as f:
        for u in unique_urls:
            f.write(u + "\n")

    summary = {
        "query_count": len(specs),
        "raw_urls_total": int(sum(x["raw_urls"] for x in per_query_stats)),
        "unique_listing_count": len(unique_urls),
        "output_dir": run_dir,
    }
    with open(os.path.join(run_dir, "dedupe_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"dedupe summary: {summary}")
    print(f"deduped URL file: {unique_urls_file}")

    raw_properties_file = os.path.join(run_dir, "properties_raw_deduped.jsonl")
    ok, fail = crawl_urls(
        urls=unique_urls,
        out_jsonl=raw_properties_file,
        source_name=args.source_name,
        sleep_sec=args.sleep_sec,
        workers=max(1, args.crawl_workers),
    )
    print(f"crawl done. OK={ok}, FAIL={fail}")

    final_jsonl = os.path.join(run_dir, "properties_with_discovery.jsonl")
    written = 0
    with open(raw_properties_file, "r", encoding="utf-8") as fin, open(final_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            normalized = normalize_url(str(rec.get("url", "") or ""))
            key = listing_key_from_url(normalized)
            rec["url"] = normalized
            rec["listing_id"] = key
            rec["source_site"] = str(rec.get("source", args.source_name) or args.source_name)
            rec["discovery_paths"] = serialize_paths(key_to_paths.get(key, set()))
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print(f"final jsonl: {final_jsonl} (rows={written})")


if __name__ == "__main__":
    main()
