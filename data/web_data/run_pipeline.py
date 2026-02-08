import argparse
import json
import math
import os
import re
from urllib.parse import parse_qs, urlparse
from typing import Any, Dict

from batch_crawl import crawl_urls
from get_urls import (
    DEFAULT_RESULTS_PER_PAGE,
    build_rightmove_search_url,
    collect_pages_parallel,
    DEFAULT_DISPLAY_LOCATION_IDENTIFIER,
    DEFAULT_LOCATION_IDENTIFIER,
    resolve_location_identifiers,
)

DEFAULT_LISTING_URLS_FILE = "listing_urls.txt"
DEFAULT_PROPERTIES_FILE = "properties.jsonl"
ASK_AGENT = "Ask agent"


def slugify_area(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "unknown_area"


def infer_area_name(search_url: str, search_location: str) -> str:
    if search_location and search_location.strip():
        return search_location.strip()
    if search_url:
        try:
            qs = parse_qs(urlparse(search_url).query)
            v = qs.get("searchLocation", [None])[0]
            if v:
                return str(v).strip()
        except Exception:
            pass
    return "unknown_area"


def is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def replace_nulls(obj: Any) -> Any:
    if obj is None:
        return ASK_AGENT
    if isinstance(obj, dict):
        return {k: replace_nulls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_nulls(v) for v in obj]
    return obj


def to_str(v: Any) -> str:
    if v is None or is_nan(v):
        return ASK_AGENT
    if isinstance(v, list):
        # Keep list-of-dict as JSON string so stations/schools remain structured in one field.
        if all(isinstance(x, dict) for x in v):
            try:
                s = json.dumps(v, ensure_ascii=False)
                return s if s.strip() else ASK_AGENT
            except Exception:
                return ASK_AGENT
        items = []
        for x in v:
            if x is None or is_nan(x):
                continue
            s = str(x).strip()
            if s:
                items.append(s)
        return " | ".join(items) if items else ASK_AGENT
    if isinstance(v, dict):
        try:
            s = json.dumps(v, ensure_ascii=False)
            return s if s.strip() else ASK_AGENT
        except Exception:
            return ASK_AGENT
    s = str(v).strip()
    return s if s else ASK_AGENT


def clean_one(rec: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in rec.items():
        out[k] = to_str(v)
    return out


def postprocess_jsonl(in_path: str, mode: str, out_path: str = None) -> str:
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return in_path

    if out_path:
        target = out_path
    else:
        root, ext = os.path.splitext(in_path)
        suffix = "_ask_agent" if mode == "ask-agent" else "_clean"
        target = f"{root}{suffix}{ext or '.jsonl'}"

    n = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(target, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if mode == "ask-agent":
                transformed = replace_nulls(rec)
            elif mode == "all-string":
                transformed = clean_one(rec)
            else:
                raise ValueError("--postprocess must be one of: none, ask-agent, all-string")
            fout.write(json.dumps(transformed, ensure_ascii=False) + "\n")
            n += 1

    print(f"Postprocessed {n} records with mode={mode} -> {target}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command pipeline: build search -> collect detail URLs -> crawl properties."
    )

    parser.add_argument("--search-url", default=None, help="Full Rightmove search URL. If set, keyword mode is skipped.")
    parser.add_argument("--search-location", default="Canary Wharf, East London", help="Location keywords for Rightmove search.")
    parser.add_argument(
        "--min-bedrooms",
        type=int,
        default=None,
        help="Minimum bedrooms filter. Leave unset to include all bedroom counts.",
    )
    parser.add_argument(
        "--max-bedrooms",
        type=int,
        default=None,
        help="Maximum bedrooms filter. Leave unset to include all bedroom counts.",
    )
    parser.add_argument("--radius", type=float, default=0.0, help="Search radius in miles.")
    parser.add_argument("--sort-type", type=int, default=6, help="Rightmove sortType value.")
    parser.add_argument(
        "--location-identifier",
        default=None,
        help="Rightmove locationIdentifier (e.g. REGION^85362).",
    )
    parser.add_argument(
        "--display-location-identifier",
        default=None,
        help="Rightmove displayLocationIdentifier (e.g. Canary-Wharf.html).",
    )
    parser.add_argument("--exclude-let-agreed", action="store_true", help="Exclude let-agreed listings.")

    parser.add_argument("--pages", type=int, default=3, help="How many search result pages to collect.")
    parser.add_argument("--start-page", type=int, default=0, help="Start page number (inclusive).")
    parser.add_argument("--end-page", type=int, default=None, help="End page number (exclusive). Overrides --pages when set.")
    parser.add_argument("--per-page", type=int, default=DEFAULT_RESULTS_PER_PAGE, help="Results per page; Rightmove default is 24.")
    parser.add_argument("--collect-workers", type=int, default=1, help="Parallel workers for search result page crawling.")
    parser.add_argument("--urls-out", default=None, help="Where to save collected listing URLs (defaults to area folder).")
    parser.add_argument("--max-listings", type=int, default=1000, help="Cap listing count before detail crawling. Use 0 to disable cap.")

    parser.add_argument("--properties-out", default=None, help="Output JSONL for crawled properties (defaults to area folder).")
    parser.add_argument("--output-root", default=".", help="Root folder where area-named result folders are created.")
    parser.add_argument(
        "--postprocess",
        default="none",
        choices=["none", "ask-agent", "all-string"],
        help="Postprocess mode: none, ask-agent (replace null), all-string (all values to string + Ask agent).",
    )
    parser.add_argument(
        "--postprocess-out",
        default=None,
        help="Optional output path for postprocessed JSONL. Defaults to auto-generated suffix.",
    )
    parser.add_argument("--source-name", default="rightmove", help="Source label stored in output.")
    parser.add_argument("--sleep-sec", type=float, default=1.0, help="Delay between detail page crawls.")
    parser.add_argument("--crawl-workers", type=int, default=4, help="Parallel workers for detail page crawling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_page < 0:
        raise ValueError("--start-page must be >= 0")
    end_page = args.end_page if args.end_page is not None else args.start_page + max(0, args.pages)
    if end_page < args.start_page:
        raise ValueError("--end-page must be >= --start-page")

    if args.search_url:
        search_url = args.search_url
    else:
        location_identifier = args.location_identifier
        display_location_identifier = args.display_location_identifier
        if not location_identifier or not display_location_identifier:
            resolved_li, resolved_dli = resolve_location_identifiers(
                search_location=args.search_location,
                radius=args.radius,
            )
            location_identifier = location_identifier or resolved_li
            display_location_identifier = display_location_identifier or resolved_dli
            if not location_identifier or not display_location_identifier:
                if "canary wharf" in args.search_location.lower():
                    location_identifier = location_identifier or DEFAULT_LOCATION_IDENTIFIER
                    display_location_identifier = (
                        display_location_identifier or DEFAULT_DISPLAY_LOCATION_IDENTIFIER
                    )
                    print(
                        "Identifier auto-resolve failed; using Canary Wharf defaults:"
                        f" locationIdentifier={location_identifier},"
                        f" displayLocationIdentifier={display_location_identifier}"
                    )
                else:
                    raise RuntimeError(
                        "Failed to resolve location identifiers. "
                        "Please pass --search-url or explicit --location-identifier and "
                        "--display-location-identifier."
                    )
            print(
                "Resolved identifiers:"
                f" locationIdentifier={location_identifier},"
                f" displayLocationIdentifier={display_location_identifier}"
            )

        search_url = build_rightmove_search_url(
            search_location=args.search_location,
            min_bedrooms=args.min_bedrooms,
            max_bedrooms=args.max_bedrooms,
            radius=args.radius,
            include_let_agreed=not args.exclude_let_agreed,
            sort_type=args.sort_type,
            location_identifier=location_identifier,
            display_location_identifier=display_location_identifier,
        )

    print(f"Using search URL:\n{search_url}")

    area_name = infer_area_name(search_url=search_url, search_location=args.search_location)
    area_slug = slugify_area(area_name)
    run_dir = os.path.join(args.output_root, area_slug)
    os.makedirs(run_dir, exist_ok=True)

    urls_out = args.urls_out or os.path.join(run_dir, DEFAULT_LISTING_URLS_FILE)
    properties_out = args.properties_out or os.path.join(run_dir, DEFAULT_PROPERTIES_FILE)
    postprocess_out = args.postprocess_out
    if not postprocess_out and args.postprocess != "none":
        suffix = "_ask_agent" if args.postprocess == "ask-agent" else "_clean"
        postprocess_out = os.path.join(run_dir, f"properties{suffix}.jsonl")

    print(f"Area folder: {run_dir}")
    print(
        "Collecting listing URLs from page range "
        f"[{args.start_page}, {end_page}) with collect-workers={max(1, args.collect_workers)}"
    )
    urls = collect_pages_parallel(
        search_url=search_url,
        start_page=args.start_page,
        end_page=end_page,
        per_page=args.per_page,
        workers=max(1, args.collect_workers),
    )
    if args.max_listings and args.max_listings > 0 and len(urls) > args.max_listings:
        urls = urls[: args.max_listings]
        print(f"Capped to first {len(urls)} listings by --max-listings")

    with open(urls_out, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")
    print(f"Saved {len(urls)} listing URLs to {urls_out}")

    ok, fail = crawl_urls(
        urls=urls,
        out_jsonl=properties_out,
        source_name=args.source_name,
        sleep_sec=args.sleep_sec,
        workers=max(1, args.crawl_workers),
    )
    print(f"Pipeline done. OK={ok}, FAIL={fail}")
    print(f"Properties JSONL: {properties_out}")
    final_out = postprocess_jsonl(
        in_path=properties_out,
        mode=args.postprocess,
        out_path=postprocess_out,
    )
    print(f"Final output: {final_out}")


if __name__ == "__main__":
    main()
