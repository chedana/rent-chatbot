# # # --- one-page extractor (fixed: full description, sqft with commas, address, and "Ask agent"/unknown values) ---

# # import json, re
# # from dataclasses import dataclass, asdict
# # from datetime import datetime, timezone
# # from typing import Dict, List, Optional, Tuple

# # from bs4 import BeautifulSoup
# # from playwright.sync_api import sync_playwright


# # # ----------------------------
# # # Utils
# # # ----------------------------
# # def now_utc_iso() -> str:
# #     return datetime.now(timezone.utc).isoformat()

# # def norm_key(s: str) -> str:
# #     s = (s or "").strip().replace("\u00a0", " ")
# #     s = re.sub(r"\s+", " ", s).replace(":", "")
# #     return s.lower()

# # def clean_text(s: str) -> str:
# #     s = (s or "").replace("\u00a0", " ")
# #     s = re.sub(r"\s+", " ", s).strip()
# #     return s

# # def parse_money(s: str) -> Optional[int]:
# #     """Parse '£3,750' -> 3750. Returns None if no £ amount exists."""
# #     if not s:
# #         return None
# #     m = re.search(r"£\s*([\d,]+)", s)
# #     return int(m.group(1).replace(",", "")) if m else None

# # def parse_date_ddmmyyyy(s: str) -> Optional[str]:
# #     """Parse '08/04/2026' -> '2026-04-08'."""
# #     s = (s or "").strip()
# #     m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", s)
# #     if not m:
# #         return None
# #     dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
# #     return f"{yyyy}-{mm}-{dd}"

# # # Unknown / Ask agent normalization
# # UNKNOWN_TOKENS = {
# #     "ask agent", "not provided", "not known", "unknown", "n/a", "na", "-", "—", "tbc"
# # }

# # def normalize_maybe_unknown(v: str) -> Optional[str]:
# #     """Return cleaned string; map known unknown tokens to 'Ask agent'; None if empty."""
# #     if v is None:
# #         return None
# #     s = clean_text(v)
# #     if not s:
# #         return None
# #     if s.lower() in UNKNOWN_TOKENS:
# #         return "Ask agent"
# #     return s


# # # ----------------------------
# # # FIX #1: robust sqft/sqm parsing (handles commas like 6,028)
# # # ----------------------------
# # def parse_int_with_commas(s: str) -> Optional[int]:
# #     if not s:
# #         return None
# #     s2 = s.replace(",", "").strip()
# #     return int(s2) if s2.isdigit() else None

# # def extract_sizes_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
# #     sqft_candidates = re.findall(
# #         r"(\d{1,3}(?:,\d{3})+|\d+)\s*sq\s*ft", text, flags=re.IGNORECASE
# #     )
# #     sqm_candidates = re.findall(
# #         r"(\d{1,3}(?:,\d{3})+|\d+)\s*sq\s*m", text, flags=re.IGNORECASE
# #     )
# #     sqft_vals = [parse_int_with_commas(x) for x in sqft_candidates]
# #     sqm_vals  = [parse_int_with_commas(x) for x in sqm_candidates]

# #     # choose largest to avoid accidental small matches like "28"
# #     sqft = max([v for v in sqft_vals if v is not None], default=None)
# #     sqm  = max([v for v in sqm_vals  if v is not None], default=None)
# #     return sqft, sqm


# # # ----------------------------
# # # FIX #2: FULL description extraction
# # # ----------------------------
# # def extract_description(soup: BeautifulSoup) -> Optional[str]:
# #     """
# #     Extract FULL property description from the page.
# #     Find a 'Description' section header and collect text until next major heading.
# #     """
# #     header = None
# #     for tag in soup.find_all(["h2", "h3", "h4"]):
# #         t = tag.get_text(" ", strip=True).lower()
# #         if t == "description" or "description" in t:
# #             header = tag
# #             break
# #     if not header:
# #         return None

# #     parts = []
# #     cur = header.find_next_sibling()
# #     while cur:
# #         if cur.name in ["h2", "h3", "h4"]:
# #             break
# #         text = cur.get_text(" ", strip=True)
# #         if text:
# #             parts.append(clean_text(text))
# #         cur = cur.find_next_sibling()

# #     desc = "\n\n".join([p for p in parts if p])
# #     return desc if desc else None


# # # ----------------------------
# # # Data model
# # # ----------------------------
# # @dataclass
# # class PropertyRecord:
# #     source: str
# #     url: str
# #     scraped_at: str

# #     address: Optional[str] = None
# #     title: Optional[str] = None
# #     added_date: Optional[str] = None

# #     price_pcm: Optional[int] = None
# #     price_pw: Optional[int] = None

# #     # deposit can be "£xxxx" OR "Ask agent" OR None (not shown)
# #     deposit: Optional[str] = None

# #     available_from: Optional[str] = None
# #     min_tenancy: Optional[str] = None
# #     let_type: Optional[str] = None
# #     furnish_type: Optional[str] = None
# #     council_tax: Optional[str] = None

# #     property_type: Optional[str] = None
# #     bedrooms: Optional[int] = None
# #     bathrooms: Optional[int] = None
# #     size_sqft: Optional[int] = None
# #     size_sqm: Optional[int] = None

# #     description: Optional[str] = None
# #     features: Optional[List[str]] = None


# # DTDD_MAP = {
# #     "let available date": "available_from",
# #     "deposit": "deposit",
# #     "min. tenancy": "min_tenancy",
# #     "min tenancy": "min_tenancy",
# #     "let type": "let_type",
# #     "furnish type": "furnish_type",
# #     "council tax": "council_tax",
# # }


# # # ----------------------------
# # # Fetch
# # # ----------------------------
# # def fetch_rendered_html(url: str, timeout_ms: int = 45_000) -> str:
# #     with sync_playwright() as p:
# #         browser = p.chromium.launch(headless=True)
# #         page = browser.new_page()
# #         page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
# #         page.wait_for_timeout(2000)  # allow hydration

# #         # If description is collapsed behind "Read more", try to expand (safe if not present)
# #         try:
# #             page.click("text=Read more", timeout=1500)
# #             page.wait_for_timeout(300)
# #         except Exception:
# #             pass

# #         html = page.content()
# #         browser.close()
# #         return html


# # # ----------------------------
# # # Extractors
# # # ----------------------------
# # def extract_address(soup: BeautifulSoup) -> Optional[str]:
# #     """
# #     For Rightmove-style layout: address line is typically right above the price.
# #     Anchor by locating the 'pcm'/'pw' node and walking backwards to find a plausible address.
# #     """
# #     price_node = soup.find(string=re.compile(r"\bpcm\b", re.IGNORECASE)) or \
# #                  soup.find(string=re.compile(r"\bpw\b", re.IGNORECASE))
# #     if not price_node:
# #         return None

# #     price_el = getattr(price_node, "parent", None)
# #     if not price_el:
# #         return None

# #     container = price_el
# #     for _ in range(4):
# #         if getattr(container, "parent", None):
# #             container = container.parent

# #     prev = container
# #     for _ in range(300):
# #         prev = prev.find_previous()
# #         if not prev:
# #             break
# #         if getattr(prev, "name", None) not in ["div", "span", "p", "h1", "h2", "h3"]:
# #             continue

# #         txt = clean_text(prev.get_text(" ", strip=True))
# #         if not txt:
# #             continue

# #         # Reject price lines
# #         if re.search(r"£\s*[\d,]+\s*(pcm|pw)", txt, re.IGNORECASE):
# #             continue
# #         if "tenancy info" in txt.lower():
# #             continue

# #         # Accept plausible address: has comma OR postcode-like token (E14, SW1, etc.)
# #         if "," in txt or re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\b", txt):
# #             return txt

# #     return None


# # def extract_dt_dd_pairs(soup: BeautifulSoup) -> Dict[str, str]:
# #     out: Dict[str, str] = {}
# #     for dt in soup.find_all("dt"):
# #         label = dt.get_text(" ", strip=True)
# #         if not label:
# #             continue
# #         dd = dt.find_next_sibling("dd") or dt.find_next("dd")
# #         if not dd:
# #             continue
# #         value = dd.get_text(" ", strip=True)
# #         if value:
# #             out[label] = value
# #     return out


# # def extract_title_and_added_date(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
# #     h1 = soup.find("h1") or soup.find("h2")
# #     title = clean_text(h1.get_text(" ", strip=True)) if h1 else None

# #     text = soup.get_text(" ", strip=True)
# #     m = re.search(r"Added on\s+(\d{2})/(\d{2})/(\d{4})", text)
# #     added = f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None
# #     return title, added


# # def extract_price(soup: BeautifulSoup) -> Tuple[Optional[int], Optional[int]]:
# #     text = soup.get_text(" ", strip=True)
# #     m_pcm = re.search(r"£\s*[\d,]+\s*pcm", text, re.IGNORECASE)
# #     m_pw  = re.search(r"£\s*[\d,]+\s*pw",  text, re.IGNORECASE)
# #     return (
# #         parse_money(m_pcm.group(0)) if m_pcm else None,
# #         parse_money(m_pw.group(0)) if m_pw else None
# #     )


# # def extract_core_specs(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int]]:
# #     text_nl = soup.get_text("\n", strip=True)

# #     prop_type = None
# #     m = re.search(r"PROPERTY TYPE\s*\n([^\n]+)", text_nl, re.IGNORECASE)
# #     if m:
# #         prop_type = clean_text(m.group(1))

# #     bedrooms = None
# #     m = re.search(r"BEDROOMS\s*\n(\d+)", text_nl, re.IGNORECASE)
# #     if m:
# #         bedrooms = int(m.group(1))

# #     bathrooms = None
# #     m = re.search(r"BATHROOMS\s*\n(\d+)", text_nl, re.IGNORECASE)
# #     if m:
# #         bathrooms = int(m.group(1))

# #     text_sp = soup.get_text(" ", strip=True)
# #     size_sqft, size_sqm = extract_sizes_from_text(text_sp)

# #     return prop_type, bedrooms, bathrooms, size_sqft, size_sqm


# # def extract_features(soup: BeautifulSoup) -> Optional[List[str]]:
# #     heading = None
# #     for tag in soup.find_all(["h2", "h3", "h4"]):
# #         t = tag.get_text(" ", strip=True).lower()
# #         if t == "key features" or "key features" in t:
# #             heading = tag
# #             break
# #     if not heading:
# #         return None

# #     container = heading.find_next()
# #     lis = container.find_all("li") if container else []
# #     feats = [clean_text(li.get_text(" ", strip=True)) for li in lis if li.get_text(strip=True)]
# #     feats = [f for f in feats if f and len(f) <= 200]
# #     # de-dup keep order
# #     dedup = []
# #     seen = set()
# #     for f in feats:
# #         if f not in seen:
# #             seen.add(f)
# #             dedup.append(f)
# #     return dedup or None


# # # ----------------------------
# # # Build record
# # # ----------------------------
# # def build_record_from_html(html: str, url: str, source: str = "rightmove") -> PropertyRecord:
# #     soup = BeautifulSoup(html, "lxml")

# #     title, added_date = extract_title_and_added_date(soup)
# #     price_pcm, price_pw = extract_price(soup)

# #     rec = PropertyRecord(
# #         source=source,
# #         url=url,
# #         scraped_at=now_utc_iso(),
# #         title=title,
# #         added_date=added_date,
# #         price_pcm=price_pcm,
# #         price_pw=price_pw,
# #     )

# #     # address
# #     rec.address = extract_address(soup)

# #     # dt/dd letting details
# #     pairs = extract_dt_dd_pairs(soup)
# #     for raw_k, raw_v in pairs.items():
# #         k = norm_key(raw_k)
# #         field = DTDD_MAP.get(k)
# #         if not field:
# #             continue

# #         v = normalize_maybe_unknown(raw_v)

# #         if field == "available_from":
# #             if v is None:
# #                 rec.available_from = None
# #             else:
# #                 rec.available_from = parse_date_ddmmyyyy(v) or v

# #         elif field == "deposit":
# #             # keep text (e.g. "Ask agent" or "£3,000")
# #             rec.deposit = v

# #         else:
# #             setattr(rec, field, v)

# #     # specs + sizes
# #     prop_type, bedrooms, bathrooms, size_sqft, size_sqm = extract_core_specs(soup)
# #     rec.property_type = prop_type
# #     rec.bedrooms = bedrooms
# #     rec.bathrooms = bathrooms
# #     rec.size_sqft = size_sqft
# #     rec.size_sqm = size_sqm

# #     # description + features
# #     rec.description = extract_description(soup)
# #     rec.features = extract_features(soup)

# #     return rec


# # # ----------------------------
# # # Example: run one page
# # # ----------------------------
# # if __name__ == "__main__":
# #     test_url = "PASTE_RIGHTMOVE_PROPERTY_URL_HERE"
# #     html = fetch_rendered_html(test_url)
# #     rec = build_record_from_html(html, url=test_url, source="rightmove")
# #     print(json.dumps(asdict(rec), ensure_ascii=False, indent=2))


# # extract_one_page.py
# # Rightmove one-page extractor
# # Fixes:
# #  - Click "Read full description" (and variants) so full text is visible
# #  - Remove "Read full description"/"Read more" UI text from extracted description
# #  - Robust sqft/sqm parsing (handles commas like 6,028; picks largest)
# #  - Address extraction (line above price)
# #  - Letting detail values can be real values or "Ask agent" (strict normalization)
# #
# # Install:
# #   pip install playwright beautifulsoup4 lxml
# #   python -m playwright install chromium
# #
# # Run:
# #   python extract_one_page.py --url "https://www.rightmove.co.uk/properties/171879728#/?channel=RES_LET"

# import argparse
# import json
# import re
# from dataclasses import dataclass, asdict
# from datetime import datetime, timezone
# from typing import Dict, List, Optional, Tuple

# from bs4 import BeautifulSoup
# from playwright.sync_api import sync_playwright


# # ----------------------------
# # Utils
# # ----------------------------
# def now_utc_iso() -> str:
#     return datetime.now(timezone.utc).isoformat()

# def clean_text(s: str) -> str:
#     s = (s or "").replace("\u00a0", " ")
#     s = re.sub(r"\s+", " ", s).strip()
#     return s

# def norm_key(s: str) -> str:
#     return clean_text(s).replace(":", "").lower()

# def parse_money(s: str) -> Optional[int]:
#     """Parse '£3,750' -> 3750. Returns None if no £ amount exists."""
#     if not s:
#         return None
#     m = re.search(r"£\s*([\d,]+)", s)
#     return int(m.group(1).replace(",", "")) if m else None

# def parse_date_ddmmyyyy(s: str) -> Optional[str]:
#     """Parse '08/04/2026' -> '2026-04-08'."""
#     s = (s or "").strip()
#     m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", s)
#     if not m:
#         return None
#     dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
#     return f"{yyyy}-{mm}-{dd}"

# # strict unknown mapping: only exact tokens become "Ask agent"; empty stays None
# UNKNOWN_TOKENS = {
#     "ask agent", "ask the agent",
#     "not provided", "not known", "unknown",
#     "n/a", "na", "-", "—", "tbc"
# }

# def normalize_maybe_unknown(v: Optional[str]) -> Optional[str]:
#     if v is None:
#         return None
#     s = clean_text(v)
#     if not s:
#         return None
#     if s.lower() in UNKNOWN_TOKENS:
#         return "Ask agent"
#     return s


# # ----------------------------
# # Size parsing (handles commas like 6,028)
# # ----------------------------
# def parse_int_with_commas(s: str) -> Optional[int]:
#     if not s:
#         return None
#     s2 = s.replace(",", "").strip()
#     return int(s2) if s2.isdigit() else None

# def extract_sizes_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
#     sqft_candidates = re.findall(
#         r"(\d{1,3}(?:,\d{3})+|\d+)\s*sq\s*ft",
#         text,
#         flags=re.IGNORECASE,
#     )
#     sqm_candidates = re.findall(
#         r"(\d{1,3}(?:,\d{3})+|\d+)\s*sq\s*m",
#         text,
#         flags=re.IGNORECASE,
#     )

#     sqft_vals = [parse_int_with_commas(x) for x in sqft_candidates]
#     sqm_vals = [parse_int_with_commas(x) for x in sqm_candidates]

#     # choose largest to avoid accidental small matches like "28"
#     sqft = max([v for v in sqft_vals if v is not None], default=None)
#     sqm = max([v for v in sqm_vals if v is not None], default=None)
#     return sqft, sqm


# # ----------------------------
# # Data model
# # ----------------------------
# @dataclass
# class PropertyRecord:
#     source: str
#     url: str
#     scraped_at: str

#     address: Optional[str] = None
#     title: Optional[str] = None
#     added_date: Optional[str] = None

#     price_pcm: Optional[int] = None
#     price_pw: Optional[int] = None

#     deposit: Optional[str] = None  # "£xxxx" or "Ask agent" or None

#     available_from: Optional[str] = None
#     min_tenancy: Optional[str] = None
#     let_type: Optional[str] = None
#     furnish_type: Optional[str] = None
#     council_tax: Optional[str] = None

#     property_type: Optional[str] = None
#     bedrooms: Optional[int] = None
#     bathrooms: Optional[int] = None
#     size_sqft: Optional[int] = None
#     size_sqm: Optional[int] = None

#     description: Optional[str] = None
#     features: Optional[List[str]] = None

#     # you can add later if you need
#     # stations: Optional[List[str]] = None
#     # schools: Optional[List[str]] = None


# DTDD_MAP = {
#     "let available date": "available_from",
#     "deposit": "deposit",
#     "min. tenancy": "min_tenancy",
#     "min tenancy": "min_tenancy",
#     "let type": "let_type",
#     "furnish type": "furnish_type",
#     "council tax": "council_tax",
# }


# # ----------------------------
# # Playwright fetch (click expand description)
# # ----------------------------
# def _click_first_available(page, selectors: List[str], timeout_ms: int = 2500) -> bool:
#     for sel in selectors:
#         try:
#             page.click(sel, timeout=timeout_ms)
#             page.wait_for_timeout(400)
#             return True
#         except Exception:
#             continue
#     return False


# def fetch_rendered_html(url: str, timeout_ms: int = 45_000) -> str:
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         page = browser.new_page()
#         page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
#         page.wait_for_timeout(2000)  # allow hydration

#         # ✅ Click expand description if present
#         _click_first_available(
#             page,
#             selectors=[
#                 'text=Read full description',
#                 'text=Read Full Description',
#                 'text=Read more',
#                 'text=Read More',
#                 'text=Show more',
#                 'text=Show More',
#                 'role=button[name="Read full description"]',
#                 'role=link[name="Read full description"]',
#                 'role=button[name="Read more"]',
#                 'role=link[name="Read more"]',
#             ],
#             timeout_ms=2500,
#         )

#         page.wait_for_timeout(600)
#         html = page.content()
#         browser.close()
#         return html


# # ----------------------------
# # Extractors
# # ----------------------------
# def extract_title_and_added_date(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
#     h1 = soup.find("h1") or soup.find("h2")
#     title = clean_text(h1.get_text(" ", strip=True)) if h1 else None

#     text = soup.get_text(" ", strip=True)
#     m = re.search(r"Added on\s+(\d{2})/(\d{2})/(\d{4})", text)
#     added = f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None
#     return title, added


# def extract_price(soup: BeautifulSoup) -> Tuple[Optional[int], Optional[int]]:
#     text = soup.get_text(" ", strip=True)
#     m_pcm = re.search(r"£\s*[\d,]+\s*pcm", text, re.IGNORECASE)
#     m_pw = re.search(r"£\s*[\d,]+\s*pw", text, re.IGNORECASE)
#     return (
#         parse_money(m_pcm.group(0)) if m_pcm else None,
#         parse_money(m_pw.group(0)) if m_pw else None,
#     )


# def extract_address(soup: BeautifulSoup) -> Optional[str]:
#     """
#     Address is typically the line right above the price on Rightmove.
#     Anchor by locating 'pcm'/'pw' then walk backwards to find a plausible address-like line.
#     """
#     price_node = soup.find(string=re.compile(r"\bpcm\b", re.IGNORECASE)) or soup.find(
#         string=re.compile(r"\bpw\b", re.IGNORECASE)
#     )
#     if not price_node:
#         return None

#     price_el = getattr(price_node, "parent", None)
#     if not price_el:
#         return None

#     container = price_el
#     for _ in range(4):
#         if getattr(container, "parent", None):
#             container = container.parent

#     prev = container
#     for _ in range(350):
#         prev = prev.find_previous()
#         if not prev:
#             break
#         if getattr(prev, "name", None) not in ["div", "span", "p", "h1", "h2", "h3"]:
#             continue

#         txt = clean_text(prev.get_text(" ", strip=True))
#         if not txt:
#             continue

#         if re.search(r"£\s*[\d,]+\s*(pcm|pw)", txt, re.IGNORECASE):
#             continue
#         if "tenancy info" in txt.lower():
#             continue

#         if "," in txt or re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\b", txt):
#             return txt

#     return None


# def extract_dt_dd_pairs(soup: BeautifulSoup) -> Dict[str, str]:
#     out: Dict[str, str] = {}
#     for dt in soup.find_all("dt"):
#         label = dt.get_text(" ", strip=True)
#         if not label:
#             continue
#         dd = dt.find_next_sibling("dd") or dt.find_next("dd")
#         if not dd:
#             continue
#         value = dd.get_text(" ", strip=True)
#         if value:
#             out[label] = value
#     return out


# def extract_core_specs(
#     soup: BeautifulSoup,
# ) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int]]:
#     text_nl = soup.get_text("\n", strip=True)

#     prop_type = None
#     m = re.search(r"PROPERTY TYPE\s*\n([^\n]+)", text_nl, re.IGNORECASE)
#     if m:
#         prop_type = clean_text(m.group(1))

#     bedrooms = None
#     m = re.search(r"BEDROOMS\s*\n(\d+)", text_nl, re.IGNORECASE)
#     if m:
#         bedrooms = int(m.group(1))

#     bathrooms = None
#     m = re.search(r"BATHROOMS\s*\n(\d+)", text_nl, re.IGNORECASE)
#     if m:
#         bathrooms = int(m.group(1))

#     text_sp = soup.get_text(" ", strip=True)
#     size_sqft, size_sqm = extract_sizes_from_text(text_sp)
#     return prop_type, bedrooms, bathrooms, size_sqft, size_sqm


# DESCRIPTION_UI_NOISE = {
#     "read full description",
#     "read more",
#     "show more",
#     "show less",
#     "read less",
#     "collapse",
#     "expand",
# }

# def extract_description(soup: BeautifulSoup) -> Optional[str]:
#     """
#     Extract FULL property description. Filters out UI button text like 'Read full description'.
#     """
#     header = None
#     for tag in soup.find_all(["h2", "h3", "h4"]):
#         t = tag.get_text(" ", strip=True).lower()
#         if t == "description" or "description" in t:
#             header = tag
#             break
#     if not header:
#         return None

#     parts: List[str] = []
#     cur = header.find_next_sibling()

#     while cur:
#         if cur.name in ["h2", "h3", "h4"]:
#             break

#         txt = clean_text(cur.get_text(" ", strip=True))
#         if txt:
#             # If a sibling is just the button label, skip it
#             if txt.lower() in DESCRIPTION_UI_NOISE:
#                 cur = cur.find_next_sibling()
#                 continue

#             # Remove embedded noise tokens
#             for noise in DESCRIPTION_UI_NOISE:
#                 txt = re.sub(rf"\b{re.escape(noise)}\b", "", txt, flags=re.IGNORECASE).strip()

#             if txt:
#                 parts.append(txt)

#         cur = cur.find_next_sibling()

#     desc = "\n\n".join([p for p in parts if p]).strip()
#     return desc or None


# def extract_features(soup: BeautifulSoup) -> Optional[List[str]]:
#     heading = None
#     for tag in soup.find_all(["h2", "h3", "h4"]):
#         t = tag.get_text(" ", strip=True).lower()
#         if t == "key features" or "key features" in t:
#             heading = tag
#             break
#     if not heading:
#         return None

#     container = heading.find_next()
#     lis = container.find_all("li") if container else []
#     feats = [clean_text(li.get_text(" ", strip=True)) for li in lis if li.get_text(strip=True)]
#     feats = [f for f in feats if f and len(f) <= 200]

#     out, seen = [], set()
#     for f in feats:
#         if f not in seen:
#             seen.add(f)
#             out.append(f)
#     return out or None


# # ----------------------------
# # Build record
# # ----------------------------
# def build_record_from_html(html: str, url: str, source: str = "rightmove") -> PropertyRecord:
#     soup = BeautifulSoup(html, "lxml")

#     title, added_date = extract_title_and_added_date(soup)
#     price_pcm, price_pw = extract_price(soup)

#     rec = PropertyRecord(
#         source=source,
#         url=url,
#         scraped_at=now_utc_iso(),
#         title=title,
#         added_date=added_date,
#         price_pcm=price_pcm,
#         price_pw=price_pw,
#     )

#     rec.address = extract_address(soup)

#     # letting details dt/dd
#     pairs = extract_dt_dd_pairs(soup)
#     for raw_k, raw_v in pairs.items():
#         k = norm_key(raw_k)
#         field = DTDD_MAP.get(k)
#         if not field:
#             continue

#         v = normalize_maybe_unknown(raw_v)

#         if field == "available_from":
#             if v is None:
#                 rec.available_from = None
#             else:
#                 rec.available_from = parse_date_ddmmyyyy(v) or v
#         elif field == "deposit":
#             rec.deposit = v
#         else:
#             setattr(rec, field, v)

#     # specs + sizes
#     prop_type, bedrooms, bathrooms, size_sqft, size_sqm = extract_core_specs(soup)
#     rec.property_type = prop_type
#     rec.bedrooms = bedrooms
#     rec.bathrooms = bathrooms
#     rec.size_sqft = size_sqft
#     rec.size_sqm = size_sqm

#     # description + features
#     rec.description = extract_description(soup)
#     rec.features = extract_features(soup)

#     return rec


# # ----------------------------
# # CLI
# # ----------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--url", required=True, help="Rightmove property URL")
#     parser.add_argument("--source", default="rightmove", help="Source label stored in output")
#     args = parser.parse_args()

#     html = fetch_rendered_html(args.url)
#     rec = build_record_from_html(html, url=args.url, source=args.source)
#     print(json.dumps(asdict(rec), ensure_ascii=False, indent=2))


# if __name__ == "__main__":
#     main()

# extract_one_page.py
# Rightmove one-page extractor (FULL)
# - Click "Read full description" to get full description
# - Click "Stations" tab and extract nearest stations (name + miles)
# - Click "Schools" tab and extract nearest schools (name + miles)
# - Robust sqft/sqm parsing (handles commas)
# - Address extraction (line above price)
# - Letting detail values can be real values or "Ask agent" (strict normalization)

# extract_one_page.py
# Rightmove one-page extractor (FULL, includes Stations & Schools via click+wait)
#
# Features:
# - Click "Read full description" (and variants) to expand description
# - Extract description without UI noise text (e.g., "Read full description")
# - Click "Stations" tab -> wait until "NEAREST STATIONS" appears -> parse (name, miles)
# - Click "Schools" tab  -> wait until "NEAREST SCHOOLS" appears  -> parse (name, miles)
# - Robust sqft/sqm parsing (handles commas like 6,028 and picks largest)
# - Address extraction (line above price)
# - Letting details values may be real values or "Ask agent" (strict normalization)
#
# Install:
#   pip install playwright beautifulsoup4 lxml
#   python -m playwright install chromium
#
# Run:
#   python extract_one_page.py --url "https://www.rightmove.co.uk/properties/171879728#/?channel=RES_LET"

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


# ----------------------------
# Utils
# ----------------------------
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clean_text(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_key(s: str) -> str:
    return clean_text(s).replace(":", "").lower()

def parse_money(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"£\s*([\d,]+)", s)
    return int(m.group(1).replace(",", "")) if m else None

def parse_date_ddmmyyyy(s: str) -> Optional[str]:
    s = (s or "").strip()
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", s)
    if not m:
        return None
    dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
    return f"{yyyy}-{mm}-{dd}"

# strict unknown mapping: only exact tokens become "Ask agent"; empty stays None
UNKNOWN_TOKENS = {
    "ask agent", "ask the agent",
    "not provided", "not known", "unknown",
    "n/a", "na", "-", "—", "tbc"
}
def normalize_maybe_unknown(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None

    s = clean_text(v)
    if not s:
        return None

    # ✅ HARD TRUNCATE: remove everything from Rightmove deposit boilerplate onward
    s = re.split(
        r"A deposit provides security",
        s,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    if not s:
        return None

    if s.lower() in UNKNOWN_TOKENS:
        return "Ask agent"

    return s


# ----------------------------
# Size parsing (handles commas like 6,028)
# ----------------------------
def parse_int_with_commas(s: str) -> Optional[int]:
    if not s:
        return None
    s2 = s.replace(",", "").strip()
    return int(s2) if s2.isdigit() else None

def extract_sizes_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    sqft_candidates = re.findall(
        r"(\d{1,3}(?:,\d{3})+|\d+)\s*sq\s*ft",
        text,
        flags=re.IGNORECASE,
    )
    sqm_candidates = re.findall(
        r"(\d{1,3}(?:,\d{3})+|\d+)\s*sq\s*m",
        text,
        flags=re.IGNORECASE,
    )

    sqft_vals = [parse_int_with_commas(x) for x in sqft_candidates]
    sqm_vals  = [parse_int_with_commas(x) for x in sqm_candidates]

    # choose largest to avoid accidental small matches like "28"
    sqft = max([v for v in sqft_vals if v is not None], default=None)
    sqm  = max([v for v in sqm_vals  if v is not None], default=None)
    return sqft, sqm


# ----------------------------
# Data model
# ----------------------------
@dataclass
class PropertyRecord:
    source: str
    url: str
    scraped_at: str

    address: Optional[str] = None
    title: Optional[str] = None
    added_date: Optional[str] = None

    price_pcm: Optional[int] = None
    price_pw: Optional[int] = None

    deposit: Optional[str] = None  # "£xxxx" or "Ask agent" or None

    available_from: Optional[str] = None
    min_tenancy: Optional[str] = None
    let_type: Optional[str] = None
    furnish_type: Optional[str] = None
    council_tax: Optional[str] = None

    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    size_sqft: Optional[int] = None
    size_sqm: Optional[int] = None

    description: Optional[str] = None
    features: Optional[List[str]] = None

    stations: Optional[List[Dict[str, Any]]] = None  # [{"name": "...", "miles": 0.4}, ...]
    schools: Optional[List[Dict[str, Any]]] = None   # [{"name": "...", "miles": 0.6}, ...]


DTDD_MAP = {
    "let available date": "available_from",
    "deposit": "deposit",
    "min. tenancy": "min_tenancy",
    "min tenancy": "min_tenancy",
    "let type": "let_type",
    "furnish type": "furnish_type",
    "council tax": "council_tax",
}


# ----------------------------
# Playwright helpers
# ----------------------------
def _click_first_available(page, selectors: List[str], timeout_ms: int = 2500) -> bool:
    for sel in selectors:
        try:
            page.locator(sel).first.click(timeout=timeout_ms)
            page.wait_for_timeout(400)
            return True
        except Exception:
            continue
    return False

def click_tab(page, name: str, timeout_ms: int = 5000) -> None:
    selectors = [
        f'role=tab[name="{name}"]',
        f'button:has-text("{name}")',
        f'a:has-text("{name}")',
        f'[role="button"]:has-text("{name}")',
    ]
    last = None
    deadline = time.time() + max(timeout_ms, 500) / 1000.0
    for sel in selectors:
        while time.time() < deadline:
            try:
                loc = page.locator(sel)
                cnt = loc.count()
                if cnt == 0:
                    page.wait_for_timeout(150)
                    continue
                for i in range(min(cnt, 12)):
                    item = loc.nth(i)
                    try:
                        if item.is_visible():
                            item.click(timeout=1500)
                            page.wait_for_timeout(300)
                            return
                    except Exception as e:
                        last = e
                        continue
                page.wait_for_timeout(150)
            except Exception as e:
                last = e
                page.wait_for_timeout(150)
                continue
    if last:
        raise last
    raise RuntimeError(f"Tab not clickable: {name}")


def wait_tab_active(page, name: str, timeout_ms: int = 7000) -> None:
    page.wait_for_function(
        """
        (tabName) => {
          const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
          const want = norm(tabName);
          const tabs = Array.from(document.querySelectorAll('[role="tab"], button, a'));
          for (const el of tabs) {
            const txt = norm(el.innerText || el.textContent || '');
            if (!txt) continue;
            if (txt !== want && !txt.includes(want)) continue;
            const rect = el.getBoundingClientRect();
            if (!(rect.width > 0 && rect.height > 0)) continue;

            const aria = (el.getAttribute('aria-selected') || '').toLowerCase();
            const cls = (el.className || '').toString().toLowerCase();
            if (aria === 'true' || cls.includes('active') || cls.includes('selected')) {
              return true;
            }
          }
          return false;
        }
        """,
        arg=name,
        timeout=timeout_ms,
    )

def wait_nearest_header(page, header_text: str, timeout_ms: int = 9000) -> None:
    page.wait_for_function(
        """
        (hdr) => {
          const t = (document.body.innerText || '').toLowerCase();
          return t.includes((hdr || '').toLowerCase());
        }
        """,
        arg=header_text,
        timeout=timeout_ms,
    )


def _extract_nearby_with_retry(page, header_text: str, timeout_ms: int = 10_000) -> List[Dict[str, Any]]:
    """
    Retry extractor because the panel content often hydrates after tab click.
    """
    deadline = time.time() + max(timeout_ms, 500) / 1000.0
    last_rows: List[Dict[str, Any]] = []
    while time.time() < deadline:
        try:
            rows = extract_nearby_by_header(page, header_text)
            if rows:
                return rows
            last_rows = rows
        except Exception:
            pass
        try:
            rows2 = extract_nearby_from_text_fallback(page, header_text)
            if rows2:
                return rows2
        except Exception:
            pass
        page.wait_for_timeout(250)
    return last_rows


def click_tab_js_fallback(page, name: str) -> bool:
    """
    JS fallback click for tabs when Playwright locator-based clicks are flaky.
    """
    try:
        ok = page.evaluate(
            """
            (tabName) => {
              const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const want = norm(tabName);
              const candidates = Array.from(document.querySelectorAll('[role="tab"], button, a, [role="button"]'));
              for (const el of candidates) {
                const txt = norm(el.innerText || el.textContent || '');
                if (!txt) continue;
                if (txt !== want && !txt.includes(want)) continue;
                const rect = el.getBoundingClientRect();
                const visible = rect.width > 0 && rect.height > 0;
                if (!visible) continue;
                try { el.scrollIntoView({block: 'center', inline: 'center'}); } catch (e) {}
                el.click();
                return true;
              }
              return false;
            }
            """,
            name,
        )
        return bool(ok)
    except Exception:
        return False


def activate_tab(page, name: str, timeout_ms: int = 7000) -> None:
    last = None
    try:
        click_tab(page, name, timeout_ms=timeout_ms)
        wait_tab_active(page, name, timeout_ms=timeout_ms)
        return
    except Exception as e:
        last = e

    try:
        if click_tab_js_fallback(page, name):
            page.wait_for_timeout(300)
            wait_tab_active(page, name, timeout_ms=timeout_ms)
            return
    except Exception as e:
        last = e

    if last:
        raise last
    raise RuntimeError(f"Failed to activate tab: {name}")


def extract_nearby_by_header(page, header_text: str) -> List[Dict[str, Any]]:
    """
    Parse nearby rows by DOM structure:
    - locate the target panel by header_text
    - for each "<num> miles" node, resolve its row container
    - pick the primary title line from that row as name
    """
    data = page.evaluate(
        """
        (hdr) => {
          const norm = (s) => (s || '').replace(/\\u00a0/g,' ').replace(/\\s+/g,' ').trim();
          const hdrLower = (hdr || '').toLowerCase();
          const mileRe = /^(\\d+(?:\\.\\d+)?)\\s*miles?$/i;

          const els = Array.from(document.querySelectorAll('*'));
          let headerEl = null;

          for (const el of els) {
            const t = norm(el.innerText);
            if (!t) continue;
            const low = t.toLowerCase();
            if (low === hdrLower || low.includes(hdrLower)) {
              headerEl = el;
              break;
            }
          }
          if (!headerEl) return [];

          let container = headerEl;
          for (let i = 0; i < 12; i++) {
            if (!container.parentElement) break;
            const cand = container.parentElement;
            const lines = norm(cand.innerText).split(/\\n+/).map(norm).filter(Boolean);
            const mileCount = lines.filter(x => mileRe.test(x)).length;
            if (mileCount >= 1) container = cand;
            else break;
          }

          const out = [];
          const seen = new Set();

          const invalidLine = (line) => {
            const low = norm(line).toLowerCase();
            if (!low) return true;
            if (mileRe.test(low)) return true;
            if (low.includes('nearest stations') || low.includes('nearest schools')) return true;
            if (low === 'stations' || low === 'schools' || low === 'my places') return true;
            if (low.startsWith('type:') || low.startsWith('rating:') || low.startsWith('ofsted:')) return true;
            if (low === 'state school' || low === 'independent school') return true;
            if (low.startsWith('state school') || low.startsWith('independent school')) return true;
            if (low.includes(' | ')) return true;
            return false;
          };

          const parseRow = (rowEl) => {
            const rowLines = norm(rowEl.innerText).split(/\\n+/).map(norm).filter(Boolean);
            if (!rowLines.length) return null;

            let miles = null;
            let mileIdx = -1;
            for (let i = 0; i < rowLines.length; i++) {
              const mm = rowLines[i].match(mileRe);
              if (mm) {
                miles = parseFloat(mm[1]);
                mileIdx = i;
                break;
              }
            }
            if (miles === null || Number.isNaN(miles)) return null;

            let name = null;
            // Prefer explicit title-like nodes (a/h*) over free text lines.
            const anchors = Array.from(rowEl.querySelectorAll('a, h3, h2, [role="link"]'));
            for (const a of anchors) {
              const t = norm(a.innerText || a.textContent || '');
              if (!invalidLine(t)) {
                name = t;
                break;
              }
            }

            if (!name && mileIdx > 0) {
              for (let i = mileIdx - 1; i >= 0; i--) {
                const cand = rowLines[i];
                if (!invalidLine(cand)) {
                  name = cand;
                  break;
                }
              }
            }

            if (!name || invalidLine(name)) return null;
            return { name, miles };
          };

          // Strategy A (preferred): icon-anchored row extraction.
          // Schools rows have a leading icon; using it is more stable than text heuristics.
          const iconSelectors = [
            'svg',
            'img',
            '[data-testid*="icon"]',
            '[class*="icon"]',
          ];
          const iconNodes = [];
          for (const sel of iconSelectors) {
            for (const n of Array.from(container.querySelectorAll(sel))) {
              iconNodes.push(n);
            }
          }

          for (const icon of iconNodes) {
            let row = icon;
            let found = false;
            for (let i = 0; i < 8; i++) {
              if (!row) break;
              const txt = norm(row.innerText);
              if (txt && txt.split(/\\n+/).some(line => mileRe.test(norm(line)))) {
                found = true;
                break;
              }
              row = row.parentElement;
            }
            if (!found || !row) continue;

            const parsed = parseRow(row);
            if (!parsed) continue;
            const key = parsed.name + '|' + parsed.miles;
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(parsed);
          }

          if (out.length > 0) {
            out.sort((a, b) => a.miles - b.miles);
            return out;
          }

          // Strategy B: distance-node anchored fallback.
          const all = Array.from(container.querySelectorAll('*'));
          const mileNodes = all.filter(el => mileRe.test(norm(el.innerText)));

          for (const node of mileNodes) {
            const milesText = norm(node.innerText);
            const mm = milesText.match(mileRe);
            if (!mm) continue;

            let row = node;
            for (let i = 0; i < 6; i++) {
              if (!row.parentElement) break;
              row = row.parentElement;
              const txt = norm(row.innerText);
              if (!txt) continue;
              const lines = txt.split(/\\n+/).map(norm).filter(Boolean);
              if (lines.length >= 2) break;
            }

            const parsed = parseRow(row);
            if (!parsed) continue;
            const key = parsed.name + '|' + parsed.miles;
            if (seen.has(key)) continue;
            seen.add(key);
            out.push(parsed);
          }

          out.sort((a, b) => a.miles - b.miles);
          return out;
        }
        """,
        header_text,
    )
    return data if isinstance(data, list) else []


def extract_nearby_from_text_fallback(page, header_text: str) -> List[Dict[str, Any]]:
    """
    Text-based fallback parser for Rightmove tabs.
    Works when DOM selectors are unstable but the tab text is visible.
    """
    body_text = page.inner_text("body")
    lines = [clean_text(x) for x in body_text.splitlines()]
    lines = [x for x in lines if x]
    hdr = header_text.lower()

    start = -1
    for i, line in enumerate(lines):
        if hdr in line.lower():
            start = i
            break
    if start < 0:
        return []

    out: List[Dict[str, Any]] = []
    miles_only_re = re.compile(r"^(\d+(?:\.\d+)?)\s*miles?$", re.IGNORECASE)
    full_re = re.compile(r"^(.+?)\s+(\d+(?:\.\d+)?)\s*miles?$", re.IGNORECASE)

    stop_tokens = {
        "show more on map",
        "my places",
        "stations",
        "schools",
        "nearest stations",
        "nearest schools",
    }
    meta_prefixes = ("type:", "rating:", "ofsted:", "state school", "independent school")

    for line in lines[start + 1 : start + 200]:
        low = line.lower()
        if low in stop_tokens:
            continue
        if "ofsted information displayed" in low:
            break
        if "show more on map" in low:
            break
        if low.startswith("to check broadband") or low.startswith("council tax"):
            break
        if low.startswith(meta_prefixes):
            continue
        if " | " in line and ("ofsted" in low or "state school" in low):
            continue

        m_full = full_re.match(line)
        if m_full:
            name = clean_text(m_full.group(1))
            miles = float(m_full.group(2))
            out.append({"name": name, "miles": miles})
            continue

        m_miles = miles_only_re.match(line)
        if m_miles and out:
            # Distance may appear on next line after school name.
            if out[-1].get("miles") is None:
                out[-1]["miles"] = float(m_miles.group(1))
            continue

        # Candidate school/station name line without distance on same line.
        if (
            not low.startswith(meta_prefixes)
            and " miles" not in low
            and " | " not in line
            and ":" not in line
            and len(line) > 2
            and len(line) < 120
            and not low.startswith("nearest ")
        ):
            out.append({"name": line, "miles": None})

    # Keep only rows with name and miles.
    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for row in out:
        name = clean_text(str(row.get("name") or ""))
        miles = row.get("miles")
        low_name = name.lower()
        if (
            not name
            or miles is None
            or low_name.startswith(meta_prefixes)
            or low_name == "state school"
            or low_name == "independent school"
            or low_name in stop_tokens
        ):
            continue
        key = f"{name}|{miles}"
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"name": name, "miles": float(miles)})

    cleaned.sort(key=lambda x: x["miles"])
    return cleaned


def dismiss_onetrust(page) -> None:
    # Optional: cookie overlay can block clicks
    candidates = [
        "#onetrust-accept-btn-handler",
        "#onetrust-reject-all-handler",
        "button:has-text('Accept')",
        "button:has-text('Accept all')",
        "button:has-text('Reject')",
        "button:has-text('Reject all')",
        "button:has-text('Continue')",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if loc.count() and loc.is_visible():
                loc.click(timeout=1500)
                page.wait_for_timeout(300)
                break
        except Exception:
            pass


# ----------------------------
# Fetch page + click tabs to get stations/schools
# ----------------------------
def fetch_rendered_html_and_nearby(url: str, timeout_ms: int = 45_000) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_timeout(2000)

        dismiss_onetrust(page)

        # Expand description if present
        _click_first_available(
            page,
            selectors=[
                'text=Read full description',
                'text=Read Full Description',
                'text=Read more',
                'text=Read More',
                'text=Show more',
                'text=Show More',
                'role=button[name="Read full description"]',
                'role=link[name="Read full description"]',
                'role=button[name="Read more"]',
                'role=link[name="Read more"]',
            ],
            timeout_ms=3000,
        )
        page.wait_for_timeout(500)

        stations: List[Dict[str, Any]] = []
        schools: List[Dict[str, Any]] = []

        # Stations and schools are optional enrichments: extraction failure should not fail the listing.
        try:
            activate_tab(page, "Stations", timeout_ms=7000)
            wait_nearest_header(page, "NEAREST STATIONS", timeout_ms=12000)
            stations = _extract_nearby_with_retry(page, "NEAREST STATIONS", timeout_ms=8000)
        except Exception as e:
            print(f"Warn: station extraction skipped for {url}: {e}")

        try:
            activate_tab(page, "Schools", timeout_ms=7000)
            try:
                wait_nearest_header(page, "NEAREST SCHOOLS", timeout_ms=12000)
            except Exception:
                # Some pages render schools rows first and header later.
                page.wait_for_function(
                    """
                    () => {
                      const t = (document.body.innerText || '').toLowerCase();
                      return t.includes('type:') || t.includes('rating:') || t.includes('nearest schools');
                    }
                    """,
                    timeout=10000,
                )
            schools = _extract_nearby_with_retry(page, "NEAREST SCHOOLS", timeout_ms=10000)
        except Exception as e:
            print(f"Warn: school extraction skipped for {url}: {e}")

        html = page.content()
        browser.close()
        return html, stations, schools


# ----------------------------
# Soup extractors
# ----------------------------
def extract_title_and_added_date(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[str]]:
    h1 = soup.find("h1") or soup.find("h2")
    title = clean_text(h1.get_text(" ", strip=True)) if h1 else None

    text = soup.get_text(" ", strip=True)
    m = re.search(r"Added on\s+(\d{2})/(\d{2})/(\d{4})", text)
    added = f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None
    return title, added

def extract_price(soup: BeautifulSoup) -> Tuple[Optional[int], Optional[int]]:
    text = soup.get_text(" ", strip=True)
    m_pcm = re.search(r"£\s*[\d,]+\s*pcm", text, re.IGNORECASE)
    m_pw  = re.search(r"£\s*[\d,]+\s*pw",  text, re.IGNORECASE)
    return (
        parse_money(m_pcm.group(0)) if m_pcm else None,
        parse_money(m_pw.group(0)) if m_pw else None,
    )

def extract_address(soup: BeautifulSoup) -> Optional[str]:
    price_node = soup.find(string=re.compile(r"\bpcm\b", re.IGNORECASE)) or soup.find(
        string=re.compile(r"\bpw\b", re.IGNORECASE)
    )
    if not price_node:
        return None

    price_el = getattr(price_node, "parent", None)
    if not price_el:
        return None

    container = price_el
    for _ in range(4):
        if getattr(container, "parent", None):
            container = container.parent

    prev = container
    for _ in range(350):
        prev = prev.find_previous()
        if not prev:
            break
        if getattr(prev, "name", None) not in ["div", "span", "p", "h1", "h2", "h3"]:
            continue

        txt = clean_text(prev.get_text(" ", strip=True))
        if not txt:
            continue

        if re.search(r"£\s*[\d,]+\s*(pcm|pw)", txt, re.IGNORECASE):
            continue
        if "tenancy info" in txt.lower():
            continue

        if "," in txt or re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\b", txt):
            return txt

    return None

def extract_dt_dd_pairs(soup: BeautifulSoup) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for dt in soup.find_all("dt"):
        label = dt.get_text(" ", strip=True)
        if not label:
            continue
        dd = dt.find_next_sibling("dd") or dt.find_next("dd")
        if not dd:
            continue
        value = dd.get_text(" ", strip=True)
        if value:
            out[label] = value
    return out

def extract_core_specs(soup: BeautifulSoup) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int], Optional[int]]:
    text_nl = soup.get_text("\n", strip=True)

    prop_type = None
    m = re.search(r"PROPERTY TYPE\s*\n([^\n]+)", text_nl, re.IGNORECASE)
    if m:
        prop_type = clean_text(m.group(1))

    bedrooms = None
    m = re.search(r"BEDROOMS\s*\n(\d+)", text_nl, re.IGNORECASE)
    if m:
        bedrooms = int(m.group(1))

    bathrooms = None
    m = re.search(r"BATHROOMS\s*\n(\d+)", text_nl, re.IGNORECASE)
    if m:
        bathrooms = int(m.group(1))

    text_sp = soup.get_text(" ", strip=True)
    size_sqft, size_sqm = extract_sizes_from_text(text_sp)

    return prop_type, bedrooms, bathrooms, size_sqft, size_sqm


DESCRIPTION_UI_NOISE = {
    "read full description",
    "read more",
    "show more",
    "show less",
    "read less",
    "collapse",
    "expand",
}

PARA_SEP = " <PARA> "
def _merge_paren_lines(lines: List[str]) -> List[str]:
    """
    Ensure fragments like:
      ["(", "ChestertonsLettings", ...] -> ["(ChestertonsLettings)", ...]
      ["(", "ChestertonsLettings)"]     -> ["(ChestertonsLettings)", ...]
      ["(", "ChestertonsLettings", ")"] -> ["(ChestertonsLettings)", ...]
    So final join becomes: "(ChestertonsLettings) <PARA> ..."
    """
    out: List[str] = []
    i = 0

    def norm(x: str) -> str:
        return (x or "").strip()

    while i < len(lines):
        a = norm(lines[i])

        # Core: if we see a lone "(", absorb the next non-empty token and force a closing ")"
        if a == "(":
            j = i + 1
            # find next non-empty token
            while j < len(lines) and not norm(lines[j]):
                j += 1
            if j >= len(lines):
                # dangling "(" at end -> drop it
                i += 1
                continue

            b = norm(lines[j])

            # If b is just ")", then "(" ")" -> drop both
            if b == ")":
                i = j + 1
                continue

            # If b already endswith ")", just prepend "("
            if b.endswith(")"):
                merged = "(" + b
                out.append(merged)
                i = j + 1
                continue

            # If next token after b is a standalone ")", consume it; else we will add ")"
            k = j + 1
            while k < len(lines) and not norm(lines[k]):
                k += 1

            if k < len(lines) and norm(lines[k]) == ")":
                merged = f"({b})"
                out.append(merged)
                i = k + 1
            else:
                # force closing
                merged = f"({b})"
                out.append(merged)
                i = j + 1
            continue

        # If we see a standalone ")", attach to previous (rare)
        if a == ")" and out:
            out[-1] = out[-1].rstrip() + ")"
            i += 1
            continue

        out.append(lines[i])
        i += 1

    # normalize spaces inside parentheses
    cleaned = []
    for x in out:
        x = re.sub(r"\(\s+", "(", x)
        x = re.sub(r"\s+\)", ")", x)
        x = x.strip()
        if x:
            cleaned.append(x)
    return cleaned
def extract_description(soup: BeautifulSoup) -> Optional[str]:
    header = None
    for tag in soup.find_all(["h2", "h3", "h4"]):
        t = tag.get_text(" ", strip=True).lower()
        if t == "description" or "description" in t:
            header = tag
            break
    if not header:
        return None

    parts: List[str] = []
    cur = header.find_next_sibling()
    while cur:
        if cur.name in ["h2", "h3", "h4"]:
            break

        # ✅ change 1: keep newlines as structure signal
        txt = cur.get_text("\n", strip=True)
        txt = txt.replace("\u00a0", " ")

        if txt:
            # split into paragraph-ish chunks, then clean each line
            lines = [clean_text(x) for x in txt.split("\n")]
            lines = [x for x in lines if x]

            # remove UI noise lines + embedded noise
            cleaned_lines = []
            for line in lines:
                if line.lower() in DESCRIPTION_UI_NOISE:
                    continue
                for noise in DESCRIPTION_UI_NOISE:
                    line = re.sub(rf"\b{re.escape(noise)}\b", "", line, flags=re.IGNORECASE).strip()
                line = line.strip()
                if line:
                    cleaned_lines.append(line)

            if cleaned_lines:
                # ✅ change 2: encode paragraphs with your special symbol
                parts.append(PARA_SEP.join(cleaned_lines))


            # if cleaned_lines:
            #     cleaned_lines = _merge_paren_lines(cleaned_lines)
            #     if cleaned_lines:
            #         parts.append(PARA_SEP.join(cleaned_lines))

        cur = cur.find_next_sibling()

    desc = PARA_SEP.join([p for p in parts if p]).strip()

    # your existing truncation
    desc = re.split(r"\bBrochures?\b", desc, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    return desc or None


def extract_features(soup: BeautifulSoup) -> Optional[List[str]]:
    heading = None
    for tag in soup.find_all(["h2", "h3", "h4"]):
        t = tag.get_text(" ", strip=True).lower()
        if t == "key features" or "key features" in t:
            heading = tag
            break
    if not heading:
        return None

    container = heading.find_next()
    lis = container.find_all("li") if container else []
    feats = [clean_text(li.get_text(" ", strip=True)) for li in lis if li.get_text(strip=True)]
    feats = [f for f in feats if f and len(f) <= 200]

    out, seen = [], set()
    for f in feats:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out or None


# ----------------------------
# Build record
# ----------------------------
def build_record_from_html(
    html: str,
    url: str,
    source: str = "rightmove",
    stations: Optional[List[Dict[str, Any]]] = None,
    schools: Optional[List[Dict[str, Any]]] = None,
) -> PropertyRecord:
    soup = BeautifulSoup(html, "lxml")

    title, added_date = extract_title_and_added_date(soup)
    price_pcm, price_pw = extract_price(soup)

    rec = PropertyRecord(
        source=source,
        url=url,
        scraped_at=now_utc_iso(),
        title=title,
        added_date=added_date,
        price_pcm=price_pcm,
        price_pw=price_pw,
    )

    rec.address = extract_address(soup)

    pairs = extract_dt_dd_pairs(soup)
    for raw_k, raw_v in pairs.items():
        k = norm_key(raw_k)
        field = DTDD_MAP.get(k)
        if not field:
            continue

        v = normalize_maybe_unknown(raw_v)

        if field == "available_from":
            rec.available_from = (parse_date_ddmmyyyy(v) or v) if v else None
        elif field == "deposit":
            rec.deposit = v
        else:
            setattr(rec, field, v)

    def check_val(val):
        if val is None:
            return "Ask agent"
        s = str(val).strip()
        return s if s else "Ask agent"

    prop_type, bedrooms, bathrooms, size_sqft, size_sqm = extract_core_specs(soup)

    # 使用函数处理每一个字段
    rec.property_type = check_val(prop_type)
    rec.bedrooms      = check_val(bedrooms)
    rec.bathrooms     = check_val(bathrooms)
    rec.size_sqft     = check_val(size_sqft)
    rec.size_sqm      = check_val(size_sqm)
    rec.let_type      = check_val(rec.let_type)
    rec.furnish_type  = check_val(rec.furnish_type)
    rec.min_tenancy   = check_val(rec.min_tenancy)

    # description + features
    # 对于文本，如果提取出的是空字符串 ""，你可能也希望显示 "Ask agent"
    # 所以这里也可以用 check_val，或者视情况调整
    rec.description = extract_description(soup) or "Ask agent" 
    rec.features    = extract_features(soup) or "Ask agent"
    rec.stations = stations if stations else "Ask agent"
    rec.schools = schools if schools else "Ask agent"

    return rec


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Rightmove property URL")
    parser.add_argument("--source", default="rightmove", help="Source label stored in output")
    args = parser.parse_args()

    html, stations, schools = fetch_rendered_html_and_nearby(args.url)
    rec = build_record_from_html(html, url=args.url, source=args.source, stations=stations, schools=schools)

    print(json.dumps(asdict(rec), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
