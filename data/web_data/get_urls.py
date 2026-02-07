

import re
from typing import Optional
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
from playwright.sync_api import sync_playwright, Page

SEARCH_URL = "https://www.rightmove.co.uk/property-to-rent/find.html?searchLocation=Canary+Wharf%2C+East+London&useLocationIdentifier=true&locationIdentifier=REGION%5E85362&radius=0.0&minBedrooms=1&maxBedrooms=1&_includeLetAgreed=on&index=24&sortType=6&channel=RENT&transactionType=LETTING&displayLocationIdentifier=Canary-Wharf.html"

OUT_FILE = "listing_urls.txt"
PAGES_TO_COLLECT = 3
RESULTS_PER_PAGE = 24  # Rightmove usually 24


DETAIL_PATH_RE = re.compile(r"^/properties/(\d+)", re.IGNORECASE)


def dismiss_onetrust(page: Page) -> None:
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

    # Hard remove overlays if still blocking
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

    # ensure index is set
    qs["index"] = [str(index_value)]

    # rebuild query (doseq keeps repeated params correct)
    new_query = urlencode(qs, doseq=True)

    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def collect_detail_urls_from_one_page(page: Page) -> list[str]:
    """Collect ONLY /properties/<id> links from current loaded page."""
    urls = []
    anchors = page.query_selector_all("a[href]")
    for a in anchors:
        href = a.get_attribute("href")
        u = normalize_rightmove_detail_url(page.url, href or "")
        if u:
            urls.append(u)

    # de-dup preserve order
    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]
    return urls


def collect_first_n_pages(search_url: str, n_pages: int, per_page: int = 24) -> list[str]:
    all_urls: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i in range(n_pages):
            idx = i * per_page
            page_url = set_index_param(search_url, idx)

            print(f"Loading page {i+1}/{n_pages} with index={idx}")
            page.goto(page_url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(1200)
            dismiss_onetrust(page)

            # Rightmove sometimes lazy-loads cards a bit after DOMContentLoaded
            try:
                page.wait_for_load_state("networkidle", timeout=10_000)
            except Exception:
                pass

            urls = collect_detail_urls_from_one_page(page)
            print(f"  Collected {len(urls)} detail urls on this page")

            all_urls.extend(urls)

        browser.close()

    # de-dup globally preserve order
    seen = set()
    all_urls = [u for u in all_urls if not (u in seen or seen.add(u))]
    return all_urls


if __name__ == "__main__":
    urls = collect_first_n_pages(SEARCH_URL, n_pages=PAGES_TO_COLLECT, per_page=RESULTS_PER_PAGE)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    print(f"Saved {len(urls)} detail URLs to {OUT_FILE}")