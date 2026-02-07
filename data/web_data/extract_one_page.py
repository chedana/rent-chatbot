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
        f'text={name}',
    ]
    last = None
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            loc.wait_for(state="visible", timeout=timeout_ms)
            loc.click(timeout=timeout_ms)
            page.wait_for_timeout(300)
            return
        except Exception as e:
            last = e
            continue
    if last:
        raise last

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
def extract_nearby_by_header(page, header_text: str) -> List[Dict[str, Any]]:
    """
    Parse rows like: "<name> 0.2 miles" from the container that includes header_text.
    """
    data = page.evaluate(
        """
        (hdr) => {
          const norm = (s) => (s || '').replace(/\\u00a0/g,' ').replace(/\\s+/g,' ').trim();
          const hdrLower = (hdr || '').toLowerCase();
          const mileRe = /(\\d+(?:\\.\\d+)?)\\s*miles?/i;

          const els = Array.from(document.querySelectorAll('*'));
          let headerEl = null;

          // Find header element by text match (exact or contains)
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

          // Climb up to a container that likely holds the list
          let container = headerEl;
          for (let i=0; i<12; i++) {
            if (!container.parentElement) break;
            const cand = container.parentElement;
            const txt = norm(cand.innerText);
            const matches = (txt.match(new RegExp(mileRe.source, 'ig')) || []).length;
            if (matches >= 1) container = cand;
            else break;
          }

          const lines = (container.innerText || '').split(/\\n+/).map(norm).filter(Boolean);

          const out = [];
          const seen = new Set();

          for (const line of lines) {
            const m = line.match(/^(.+?)\\s+(\\d+(?:\\.\\d+)?)\\s*miles?$/i);
            if (!m) continue;

            const name = norm(m[1]);
            const miles = parseFloat(m[2]);
            if (!name || Number.isNaN(miles)) continue;

            const low = name.toLowerCase();
            if (low.includes('open map') || low.includes('street view')) continue;
            if (low.includes('nearest stations') || low.includes('nearest schools')) continue;
            if (low === 'stations' || low === 'schools' || low === 'my places') continue;

            const key = name + '|' + miles;
            if (seen.has(key)) continue;
            seen.add(key);

            out.push({ name, miles });
          }

          out.sort((a,b) => a.miles - b.miles);
          return out;
        }
        """,
        header_text,
    )
    return data if isinstance(data, list) else []


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

        # Stations tab -> wait -> extract
        click_tab(page, "Stations", timeout_ms=6000)
        wait_nearest_header(page, "NEAREST STATIONS", timeout_ms=12000)
        stations = extract_nearby_by_header(page, "NEAREST STATIONS")

        # Schools tab -> wait -> extract
        click_tab(page, "Schools", timeout_ms=6000)
        wait_nearest_header(page, "NEAREST SCHOOLS", timeout_ms=12000)
        schools = extract_nearby_by_header(page, "NEAREST SCHOOLS")

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
        return val if val is not None else "Ask agent"

    prop_type, bedrooms, bathrooms, size_sqft, size_sqm = extract_core_specs(soup)

    # 使用函数处理每一个字段
    rec.property_type = check_val(prop_type)
    rec.bedrooms      = check_val(bedrooms)
    rec.bathrooms     = check_val(bathrooms)
    rec.size_sqft     = check_val(size_sqft)
    rec.size_sqm      = check_val(size_sqm)

    # description + features
    # 对于文本，如果提取出的是空字符串 ""，你可能也希望显示 "Ask agent"
    # 所以这里也可以用 check_val，或者视情况调整
    rec.description = extract_description(soup) or "Ask agent" 
    rec.features    = extract_features(soup) or "Ask agent"

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