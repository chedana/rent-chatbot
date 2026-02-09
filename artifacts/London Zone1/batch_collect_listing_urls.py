import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import parse_qs, quote, urlencode, urlparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
WEB_DATA_DIR = os.path.join(REPO_ROOT, "data", "web_data")
if WEB_DATA_DIR not in sys.path:
    sys.path.insert(0, WEB_DATA_DIR)

from get_urls import DEFAULT_RESULTS_PER_PAGE, collect_pages_parallel


def slugify(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "unknown"


def build_probe_url(
    search_location: str,
    radius: float = 0.0,
    sort_type: int = 6,
    location_identifier: str = None,
) -> str:
    params = {
        "searchLocation": search_location,
        "radius": str(radius),
        "sortType": str(sort_type),
        "channel": "RENT",
        "transactionType": "LETTING",
        "_includeLetAgreed": "on",
    }
    if location_identifier:
        params["useLocationIdentifier"] = "true"
        params["locationIdentifier"] = location_identifier
    return "https://www.rightmove.co.uk/property-to-rent/find.html?" + urlencode(params)


def fetch_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    data = json.loads(raw)
    if isinstance(data, dict):
        return data
    return {"items": data}


def rightmove_tokenize_query(search_location: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", search_location or "").upper()
    parts = [cleaned[i : i + 2] for i in range(0, len(cleaned), 2) if cleaned[i : i + 2]]
    return "/".join(parts)


def fetch_typeahead_candidates(search_location: str) -> list[dict]:
    tokenized = rightmove_tokenize_query(search_location)
    urls = [
        f"https://www.rightmove.co.uk/typeAhead/uknostreet/{tokenized}/",
        f"https://www.rightmove.co.uk/typeAhead/uknostreet/{tokenized}.html",
        "https://www.rightmove.co.uk/typeAhead/uknostreet/" + quote(search_location) + ".html",
        "https://www.rightmove.co.uk/typeAhead/uknostreet/" + quote(search_location) + "/",
    ]
    for url in urls:
        try:
            payload = fetch_json(url)
            if isinstance(payload.get("typeAheadLocations"), list) and payload["typeAheadLocations"]:
                return payload["typeAheadLocations"]
            if isinstance(payload.get("items"), list) and payload["items"]:
                return payload["items"]
        except urllib.error.HTTPError:
            continue
        except Exception:
            continue
    raise RuntimeError("Typeahead lookup failed for query: " + search_location)


def pick_best_candidate(search_location: str, candidates: list[dict], prefer_region: bool = False) -> dict:
    if not candidates:
        raise RuntimeError("No typeahead candidates returned")

    q = search_location.strip().lower()
    core_hint = q.split(",")[0].strip()
    city_hint = ""
    if "," in q:
        city_hint = q.split(",")[-1].strip()
    if not city_hint and " london" in q:
        city_hint = "london"

    best = None
    best_score = -10**9

    for c in candidates:
        display = str(
            c.get("displayName")
            or c.get("name")
            or c.get("location")
            or c.get("text")
            or ""
        ).strip()
        location_id = str(c.get("locationIdentifier") or c.get("locationId") or "").strip()
        if not location_id:
            continue

        d = display.lower()
        score = 0
        if d == q:
            score += 120
        if q in d:
            score += 50
        if d.startswith(q.split(",")[0].strip()):
            score += 20
        score += max(0, 10 - abs(len(d) - len(q)))
        # Require core place text (before first comma) to match reasonably well.
        if core_hint:
            if core_hint in d:
                score += 180
            else:
                score -= 220
        if city_hint:
            city_check = city_hint
            if "london" in city_hint:
                city_check = "london"
            if city_check in d:
                score += 160
            else:
                score -= 140

        if prefer_region:
            if location_id.startswith("REGION^"):
                score += 120
            elif location_id.startswith("OUTCODE^"):
                score += 80
            elif location_id.startswith("STATION^"):
                score -= 80
        else:
            if location_id.startswith("STATION^"):
                score += 120
            elif location_id.startswith("REGION^"):
                score += 40
            elif location_id.startswith("OUTCODE^"):
                score += 20

        if score > best_score:
            best = c
            best_score = score

    if not best:
        raise RuntimeError("Candidates returned but none had locationIdentifier")
    return best


def resolve_with_typeahead(search_location: str, radius: float = 0.0, prefer_region: bool = False) -> dict:
    candidates = fetch_typeahead_candidates(search_location)
    chosen = pick_best_candidate(search_location, candidates, prefer_region=prefer_region)
    resolved_search_location = str(
        chosen.get("displayName")
        or chosen.get("name")
        or chosen.get("location")
        or chosen.get("text")
        or search_location
    ).strip()
    location_identifier = str(chosen.get("locationIdentifier") or chosen.get("locationId") or "").strip()
    return {
        "resolved_search_location": resolved_search_location,
        "location_identifier": location_identifier,
        "probe_url": build_probe_url(
            search_location=resolved_search_location,
            radius=radius,
            location_identifier=location_identifier,
        ),
        "candidate_count": len(candidates),
    }


def load_inputs(args: argparse.Namespace) -> List[Dict]:
    items: List[Dict] = []

    if args.queries:
        for q in args.queries:
            q = q.strip()
            if q:
                items.append({"query": q})

    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            for line in f:
                q = line.strip()
                if q:
                    items.append({"query": q})

    if args.station_results_json:
        with open(args.station_results_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("results", []):
            q = str(r.get("prompt") or "").strip()
            if q:
                items.append(
                    {
                        "query": q,
                        "preset_probe_url": r.get("final_url") or r.get("probe_url"),
                        "preset_location_identifier": r.get("location_identifier"),
                        "preset_resolved_search_location": r.get("resolved_search_location"),
                    }
                )

    if args.search_urls:
        for u in args.search_urls:
            u = (u or "").strip()
            if not u:
                continue
            qs = parse_qs(urlparse(u).query)
            q = (qs.get("searchLocation", [""])[0] or "").strip() or u
            items.append(
                {
                    "query": q,
                    "preset_probe_url": u,
                    "preset_location_identifier": (qs.get("locationIdentifier", [""])[0] or None),
                    "preset_resolved_search_location": q,
                }
            )

    if args.search_urls_file:
        with open(args.search_urls_file, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if not u:
                    continue
                qs = parse_qs(urlparse(u).query)
                q = (qs.get("searchLocation", [""])[0] or "").strip() or u
                items.append(
                    {
                        "query": q,
                        "preset_probe_url": u,
                        "preset_location_identifier": (qs.get("locationIdentifier", [""])[0] or None),
                        "preset_resolved_search_location": q,
                    }
                )

    seen = set()
    out = []
    for it in items:
        q = it["query"]
        u = (it.get("preset_probe_url") or "").strip().lower()
        k = f"{q.lower()}|{u}"
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def build_search_url_from_preset(
    url: str,
    radius: float,
    fallback_search_location: Optional[str] = None,
    fallback_location_identifier: Optional[str] = None,
) -> str:
    # Keep provided identifier/display fields; force rental channel defaults and chosen radius.
    from urllib.parse import parse_qs, urlparse

    qs = parse_qs(urlparse(url).query)
    search_location = (qs.get("searchLocation", [None])[0] or "").strip()
    location_identifier = (qs.get("locationIdentifier", [None])[0] or "").strip()
    if not search_location:
        search_location = (fallback_search_location or "").strip()
    if not location_identifier:
        location_identifier = (fallback_location_identifier or "").strip()
    if not search_location or not location_identifier:
        raise RuntimeError("preset probe_url missing searchLocation/locationIdentifier (and no usable fallback)")
    return build_probe_url(
        search_location=search_location,
        radius=radius,
        location_identifier=location_identifier,
    )


def process_one(item: Dict, args: argparse.Namespace) -> Dict:
    query = item["query"]
    search_url: Optional[str] = None
    resolved_search_location = item.get("preset_resolved_search_location")
    location_identifier = item.get("preset_location_identifier")

    # Priority 1: use preset URL directly from results JSON as requested.
    preset_probe_url = item.get("preset_probe_url")
    if preset_probe_url:
        search_url = str(preset_probe_url).strip() or None

    if not search_url:
        auto_prefer_region = "station" not in query.lower()
        prefer_region = True if args.prefer_region else auto_prefer_region
        resolved = resolve_with_typeahead(
            search_location=query,
            radius=args.radius,
            prefer_region=prefer_region,
        )
        search_url = resolved["probe_url"]
        resolved_search_location = resolved.get("resolved_search_location")
        location_identifier = resolved.get("location_identifier")

    end_page = args.end_page if args.end_page is not None else args.start_page + max(0, args.pages)
    urls = collect_pages_parallel(
        search_url=search_url,
        start_page=args.start_page,
        end_page=end_page,
        per_page=args.per_page,
        workers=max(1, args.collect_workers),
    )

    area = str(resolved_search_location or query)
    folder = os.path.join(args.out_root, slugify(query))
    os.makedirs(folder, exist_ok=True)
    out_file = os.path.join(folder, "listing_urls.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    return {
        "query": query,
        "resolved_search_location": resolved_search_location,
        "location_identifier": location_identifier,
        "search_url": search_url,
        "out_file": out_file,
        "url_count": len(urls),
        "candidate_count": resolved.get("candidate_count") if "resolved" in locals() else None,
        "ok": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch stage-1 collector: resolve query -> collect listing URLs only."
    )
    parser.add_argument("--queries", nargs="*", default=None, help="Queries to process.")
    parser.add_argument("--queries-file", default=None, help="Text file; one query per line.")
    parser.add_argument(
        "--search-urls",
        nargs="*",
        default=None,
        help="Full Rightmove search URLs to process directly.",
    )
    parser.add_argument(
        "--search-urls-file",
        default=None,
        help="Text file; one full Rightmove search URL per line.",
    )
    parser.add_argument(
        "--station-results-json",
        default=os.path.join(SCRIPT_DIR, "zone1_station_prompt_test_results.json"),
        help="Read prompts from zone1_station_prompt_test_results.json (results[].prompt).",
    )
    parser.add_argument(
        "--out-root",
        default=os.path.join(SCRIPT_DIR, "underground"),
        help="Output root directory.",
    )

    parser.add_argument("--jobs", type=int, default=4, help="Parallel query jobs.")
    parser.add_argument("--start-page", type=int, default=0, help="Start page number (inclusive).")
    parser.add_argument("--end-page", type=int, default=None, help="End page number (exclusive).")
    parser.add_argument("--pages", type=int, default=1, help="Used when --end-page is not provided.")
    parser.add_argument("--per-page", type=int, default=DEFAULT_RESULTS_PER_PAGE, help="Results per page.")
    parser.add_argument("--collect-workers", type=int, default=1, help="Per-query page collector workers.")
    parser.add_argument("--radius", type=float, default=0.0, help="Search radius in miles.")
    parser.add_argument(
        "--prefer-region",
        action="store_true",
        help="Prefer REGION candidates (for area-style queries). Default is station-friendly.",
    )
    parser.add_argument(
        "--summary-json",
        default=os.path.join(SCRIPT_DIR, "underground", "batch_listing_urls_summary.json"),
        help="Where to save summary result JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_inputs(args)
    if not items:
        raise RuntimeError("No queries found. Pass --queries or --queries-file or --station-results-json.")

    os.makedirs(args.out_root, exist_ok=True)
    results: List[Dict] = []

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        future_map = {pool.submit(process_one, it, args): it["query"] for it in items}
        for fut in as_completed(future_map):
            q = future_map[fut]
            try:
                row = fut.result()
                results.append(row)
                print(f"[OK] {q} -> {row['url_count']} urls | {row['out_file']}")
            except Exception as exc:
                row = {"query": q, "ok": False, "error": str(exc)}
                results.append(row)
                print(f"[FAIL] {q} -> {exc}")

    ok = sum(1 for r in results if r.get("ok"))
    fail = len(results) - ok
    summary = {
        "meta": {
            "total_queries": len(results),
            "ok_queries": ok,
            "failed_queries": fail,
            "jobs": args.jobs,
            "start_page": args.start_page,
            "end_page": args.end_page,
            "pages": args.pages,
            "per_page": args.per_page,
            "collect_workers": args.collect_workers,
            "radius": args.radius,
            "prefer_region": args.prefer_region,
        },
        "results": results,
    }

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("\n=== Summary ===")
    print(f"Total queries: {len(results)}")
    print(f"OK: {ok}")
    print(f"FAIL: {fail}")
    print(f"Summary saved: {args.summary_json}")


if __name__ == "__main__":
    main()
