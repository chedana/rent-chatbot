import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

DEFAULT_SEARCH_LOCATION = "Canary Wharf, East London"
DEFAULT_OUT_FILE = "listing_urls.txt"
DEFAULT_PAGES_TO_COLLECT = 3
DEFAULT_RESULTS_PER_PAGE = 24  # Rightmove usually 24
DEFAULT_MIN_BEDROOMS = None
DEFAULT_MAX_BEDROOMS = None
DEFAULT_RADIUS = 0.0
DEFAULT_SORT_TYPE = 6
DEFAULT_LOCATION_IDENTIFIER = "REGION^85362"
DEFAULT_DISPLAY_LOCATION_IDENTIFIER = "Canary-Wharf.html"

DETAIL_PATH_RE = re.compile(r"^/properties/(\d+)", re.IGNORECASE)


def dismiss_onetrust(page: Any) -> None:
    candidates = [
        "#onetrust-accept-btn-handler",
        "#onetrust-reject-all-handler",
        "#onetrust-pc-btn-handler",
        "button:has-text('Accept')",
        "button:has-text('Accept all')",
        "button:has-text('Allow all')",
        "button:has-text('Reject')",
        "button:has-text('Reject all')",
        "button:has-text('Confirm')",
        "button:has-text('Save')",
        "button:has-text('Continue')",
    ]
    for sel in candidates:
        try:
            btn = page.query_selector(sel)
            if btn and btn.is_visible():
                btn.click(timeout=1500)
                page.wait_for_timeout(200)
                break
        except Exception:
            pass

    # Remove overlays if still blocking interaction.
    try:
        page.evaluate(
            """
            () => {
              const ids = ['onetrust-consent-sdk', 'onetrust-banner-sdk'];
              for (const id of ids) {
                const el = document.getElementById(id);
                if (el) el.remove();
              }
              const overlays = document.querySelectorAll(
                '.onetrust-pc-dark-filter, .ot-sdk-container, .ot-fade-in, .ot-overlay, .ot-modal-backdrop'
              );
              overlays.forEach(el => el.remove());
            }
            """
        )
    except Exception:
        pass


def normalize_rightmove_detail_url(current_page_url: str, href: str) -> Optional[str]:
    """Return canonical detail URL or None if not a property detail link."""
    if not href:
        return None

    full = urljoin(current_page_url, href)
    parsed = urlparse(full)

    if "rightmove.co.uk" not in parsed.netloc.lower():
        return None

    m = DETAIL_PATH_RE.match(parsed.path)
    if not m:
        return None

    prop_id = m.group(1)
    return f"https://www.rightmove.co.uk/properties/{prop_id}#/?channel=RES_LET"


def set_index_param(url: str, index_value: int) -> str:
    """Set/overwrite query param `index` in the search URL."""
    p = urlparse(url)
    qs = parse_qs(p.query)
    qs["index"] = [str(index_value)]
    new_query = urlencode(qs, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def build_rightmove_search_url(
    search_location: str,
    min_bedrooms: Optional[int] = DEFAULT_MIN_BEDROOMS,
    max_bedrooms: Optional[int] = DEFAULT_MAX_BEDROOMS,
    radius: float = DEFAULT_RADIUS,
    include_let_agreed: bool = True,
    sort_type: int = DEFAULT_SORT_TYPE,
    location_identifier: Optional[str] = None,
    display_location_identifier: Optional[str] = None,
) -> str:
    """
    Build a Rightmove rental search URL from human-readable filters.
    Note: This mode does not require manual search-page copy/paste.
    """
    params = {
        "searchLocation": search_location,
        "radius": str(radius),
        "sortType": str(sort_type),
        "channel": "RENT",
        "transactionType": "LETTING",
    }
    if min_bedrooms is not None:
        params["minBedrooms"] = str(min_bedrooms)
    if max_bedrooms is not None:
        params["maxBedrooms"] = str(max_bedrooms)
    if include_let_agreed:
        params["_includeLetAgreed"] = "on"
    if location_identifier:
        params["useLocationIdentifier"] = "true"
        params["locationIdentifier"] = location_identifier
    if display_location_identifier:
        params["displayLocationIdentifier"] = display_location_identifier

    return "https://www.rightmove.co.uk/property-to-rent/find.html?" + urlencode(params)


def resolve_location_identifiers(search_location: str, radius: float = DEFAULT_RADIUS) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve Rightmove location identifiers from human-readable location text.
    This avoids requiring users to manually find locationIdentifier values.
    """
    from playwright.sync_api import sync_playwright

    probe_url = build_rightmove_search_url(
        search_location=search_location,
        min_bedrooms=None,
        max_bedrooms=None,
        radius=radius,
        include_let_agreed=True,
        sort_type=DEFAULT_SORT_TYPE,
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(probe_url, wait_until="domcontentloaded", timeout=45_000)
        try:
            page.wait_for_load_state("networkidle", timeout=8_000)
        except Exception:
            pass

        final_url = page.url
        browser.close()

    parsed = urlparse(final_url)
    qs = parse_qs(parsed.query)
    location_identifier = qs.get("locationIdentifier", [None])[0]
    display_location_identifier = qs.get("displayLocationIdentifier", [None])[0]
    return location_identifier, display_location_identifier


def collect_detail_urls_from_one_page(page: Any) -> list[str]:
    """Collect only /properties/<id> links from current loaded page."""
    urls = []
    anchors = page.query_selector_all("a[href]")
    for a in anchors:
        href = a.get_attribute("href")
        u = normalize_rightmove_detail_url(page.url, href or "")
        if u:
            urls.append(u)

    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def collect_detail_urls_from_html(html: str) -> list[str]:
    """
    Fallback extractor for cases where anchors are not directly queryable.
    """
    ids = re.findall(r"/properties/(\d+)", html)
    urls = [f"https://www.rightmove.co.uk/properties/{pid}#/?channel=RES_LET" for pid in ids]
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def split_page_ranges(start_page: int, end_page: int, workers: int) -> list[tuple[int, int]]:
    total = max(0, end_page - start_page)
    if total == 0:
        return []
    actual_workers = max(1, min(workers, total))
    base = total // actual_workers
    rem = total % actual_workers
    out: list[tuple[int, int]] = []
    cur = start_page
    for i in range(actual_workers):
        size = base + (1 if i < rem else 0)
        nxt = cur + size
        out.append((cur, nxt))
        cur = nxt
    return out


def collect_page_range(
    search_url: str,
    start_page: int,
    end_page: int,
    per_page: int = DEFAULT_RESULTS_PER_PAGE,
    worker_name: str = "worker",
) -> list[str]:
    from playwright.sync_api import sync_playwright

    all_urls: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for page_no in range(start_page, end_page):
            idx = page_no * per_page
            page_url = set_index_param(search_url, idx)

            print(f"[{worker_name}] Loading page={page_no} index={idx}")
            page.goto(page_url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(1200)
            dismiss_onetrust(page)

            try:
                page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass

            urls = collect_detail_urls_from_one_page(page)
            if not urls:
                # Fallback: parse raw HTML in case link elements are virtualized or obfuscated.
                html = page.content()
                urls = collect_detail_urls_from_html(html)
                if urls:
                    print(f"[{worker_name}] Fallback extracted {len(urls)} urls from HTML on page={page_no}")
                else:
                    try:
                        title = page.title()
                    except Exception:
                        title = "N/A"
                    print(f"[{worker_name}] No urls found on page={page_no}. url={page.url} title={title}")
            print(f"[{worker_name}] Collected {len(urls)} urls on page={page_no}")
            all_urls.extend(urls)

        browser.close()

    seen = set()
    return [u for u in all_urls if not (u in seen or seen.add(u))]


def collect_pages_parallel(
    search_url: str,
    start_page: int,
    end_page: int,
    per_page: int = DEFAULT_RESULTS_PER_PAGE,
    workers: int = 1,
) -> list[str]:
    ranges = split_page_ranges(start_page, end_page, workers)
    if not ranges:
        return []
    if len(ranges) == 1:
        s, e = ranges[0]
        return collect_page_range(search_url, s, e, per_page=per_page, worker_name="w1")

    futures = {}
    ordered: list[tuple[int, list[str]]] = []
    with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
        for idx, (s, e) in enumerate(ranges, start=1):
            name = f"w{idx}"
            fut = pool.submit(
                collect_page_range,
                search_url,
                s,
                e,
                per_page,
                name,
            )
            futures[fut] = (s, e, name)
        for fut in as_completed(futures):
            s, e, name = futures[fut]
            urls = fut.result()
            print(f"[{name}] Finished page range [{s}, {e}) with {len(urls)} urls")
            ordered.append((s, urls))

    ordered.sort(key=lambda x: x[0])
    combined: list[str] = []
    for _, urls in ordered:
        combined.extend(urls)

    seen = set()
    return [u for u in combined if not (u in seen or seen.add(u))]


def collect_first_n_pages(search_url: str, n_pages: int, per_page: int = DEFAULT_RESULTS_PER_PAGE) -> list[str]:
    return collect_pages_parallel(
        search_url=search_url,
        start_page=0,
        end_page=max(0, n_pages),
        per_page=per_page,
        workers=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-url", default=None, help="Full Rightmove search URL. If provided, it takes precedence.")
    parser.add_argument("--search-location", default=DEFAULT_SEARCH_LOCATION, help="Location keywords (used when --search-url is not set).")
    parser.add_argument(
        "--min-bedrooms",
        type=int,
        default=DEFAULT_MIN_BEDROOMS,
        help="Minimum bedrooms filter. Leave unset to include all bedroom counts.",
    )
    parser.add_argument(
        "--max-bedrooms",
        type=int,
        default=DEFAULT_MAX_BEDROOMS,
        help="Maximum bedrooms filter. Leave unset to include all bedroom counts.",
    )
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS, help="Search radius in miles.")
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGES_TO_COLLECT, help="How many result pages to crawl.")
    parser.add_argument("--start-page", type=int, default=0, help="Start page number (inclusive).")
    parser.add_argument("--end-page", type=int, default=None, help="End page number (exclusive). Overrides --pages when set.")
    parser.add_argument("--per-page", type=int, default=DEFAULT_RESULTS_PER_PAGE, help="Results per page (Rightmove default is 24).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for collecting listing URLs.")
    parser.add_argument("--sort-type", type=int, default=DEFAULT_SORT_TYPE, help="Rightmove sortType value.")
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
    parser.add_argument("--out-file", default=DEFAULT_OUT_FILE, help="Where to write collected detail URLs.")
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
    print(
        "Collecting pages in range "
        f"[{args.start_page}, {end_page}) with workers={max(1, args.workers)}"
    )
    urls = collect_pages_parallel(
        search_url=search_url,
        start_page=args.start_page,
        end_page=end_page,
        per_page=args.per_page,
        workers=max(1, args.workers),
    )

    with open(args.out_file, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    print(f"Saved {len(urls)} detail URLs to {args.out_file}")


if __name__ == "__main__":
    main()
