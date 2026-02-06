import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from bs4 import BeautifulSoup


PROPERTY_ID_RE = re.compile(r"/properties/(\d+)")
NEXT_DATA_RE = re.compile(r'__NEXT_DATA__"\s*type="application/json">\s*(\{.*?\})\s*</script>', re.S)
LD_JSON_RE = re.compile(r'<script[^>]+type="application/ld\+json"[^>]*>\s*(\{.*?\})\s*</script>', re.S)


@dataclass
class NormalizedListing:
    property_id: str
    url: str
    title: Optional[str] = None
    address: Optional[str] = None
    postcode: Optional[str] = None

    price_pcm: Optional[int] = None
    price_pw: Optional[int] = None
    rent_frequency: Optional[str] = None  # pcm/pw etc
    deposit: Optional[str] = None
    available_from: Optional[str] = None
    added_on: Optional[str] = None

    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    property_type: Optional[str] = None
    furnish_type: Optional[str] = None

    size_sqft: Optional[float] = None
    size_sqm: Optional[float] = None

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    description: Optional[str] = None
    key_features: Optional[List[str]] = None

    agent_name: Optional[str] = None
    agent_phone: Optional[str] = None

    images: Optional[List[str]] = None
    floorplan: Optional[List[str]] = None
    epc: Optional[List[str]] = None

    # 兜底：保留原始 JSON 的关键片段（便于你后面补字段）
    raw_keys_present: Optional[List[str]] = None


def _ua_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }


def fetch_html(client: httpx.Client, url: str) -> str:
    r = client.get(url, headers=_ua_headers(), timeout=30)
    r.raise_for_status()
    return r.text


def extract_property_ids_from_search(html: str) -> List[str]:
    ids: Set[str] = set()
    for m in PROPERTY_ID_RE.finditer(html):
        ids.add(m.group(1))
    return sorted(ids)


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_embedded_json(html: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    返回:
      - next_data: 详情页常见的 __NEXT_DATA__ JSON（如果存在）
      - ld_json_list: 页面里可能有多个 application/ld+json（SEO schema）
    """
    next_data = None
    m = NEXT_DATA_RE.search(html)
    if m:
        next_data = _try_parse_json(m.group(1))

    ld_json_list: List[Dict[str, Any]] = []
    for m2 in LD_JSON_RE.finditer(html):
        obj = _try_parse_json(m2.group(1))
        if isinstance(obj, dict):
            ld_json_list.append(obj)

    return next_data, ld_json_list


def _to_int_price(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = s.replace("£", "").replace(",", "").strip()
    if not s.isdigit():
        # 有些价格会是 "1,250" 或带文本
        m = re.search(r"(\d[\d,]*)", s)
        if not m:
            return None
        s = m.group(1).replace(",", "")
    try:
        return int(s)
    except Exception:
        return None


def _extract_postcode(address: Optional[str]) -> Optional[str]:
    if not address:
        return None
    # 英国 postcode 粗略正则（够用）
    m = re.search(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b", address.upper())
    return m.group(1).strip() if m else None


def normalize_from_next_data(next_data: Dict[str, Any], property_id: str) -> NormalizedListing:
    """
    Rightmove 内部 JSON 结构会变，所以这里做“尽可能多抽取 + 不崩”的宽松解析。
    关键：把图片、floorplan、EPC、经纬度、描述、key features、agent 等拉出来。
    """
    url = f"https://www.rightmove.co.uk/properties/{property_id}"
    n = NormalizedListing(property_id=property_id, url=url)

    # 1) 尝试从 next_data 里找“最大最像详情数据”的 dict
    # 常见路径：props.pageProps 或者 props.initialReduxState / apolloState 等
    props = next_data.get("props", {}) if isinstance(next_data, dict) else {}
    page_props = props.get("pageProps", {}) if isinstance(props, dict) else {}

    # 用启发式：在 page_props 里找含有 images/description/price 的块
    candidate_blocks: List[Dict[str, Any]] = []
    def walk(obj: Any):
        if isinstance(obj, dict):
            keys = set(obj.keys())
            score = 0
            for k in ["images", "description", "price", "propertyId", "bedrooms", "bathrooms", "location"]:
                if k in keys:
                    score += 1
            if score >= 2:
                candidate_blocks.append(obj)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(page_props)

    detail = max(candidate_blocks, key=lambda d: len(d.keys()), default={})

    # 2) 标题/地址
    n.title = detail.get("title") or detail.get("propertyType") or detail.get("summary")
    n.address = detail.get("displayAddress") or detail.get("address") or detail.get("location", {}).get("displayAddress")
    n.postcode = _extract_postcode(n.address)

    # 3) 价格
    price = detail.get("price") or {}
    if isinstance(price, dict):
        n.price_pcm = _to_int_price(price.get("amount") or price.get("pcm") or price.get("perMonth"))
        n.price_pw = _to_int_price(price.get("pw") or price.get("perWeek"))
        n.rent_frequency = price.get("frequency") or price.get("rentFrequency")
    else:
        # 有时是字符串
        n.price_pcm = _to_int_price(str(price))

    n.added_on = detail.get("addedOn") or detail.get("firstVisibleDate") or detail.get("addedOrReduced")
    n.available_from = detail.get("availableFrom") or detail.get("availabilityDate")
    n.deposit = detail.get("deposit")

    # 4) 基本属性
    n.bedrooms = detail.get("bedrooms")
    n.bathrooms = detail.get("bathrooms")
    n.property_type = detail.get("propertyType")
    n.furnish_type = detail.get("furnishType") or detail.get("furnishing")

    # 5) 面积
    # 可能出现在 size / floorArea / keyFeatures 文本里
    size = detail.get("size") or detail.get("floorArea")
    if isinstance(size, dict):
        n.size_sqft = size.get("sqft")
        n.size_sqm = size.get("sqm")

    # 6) 经纬度
    loc = detail.get("location") or detail.get("latitudeLongitude") or {}
    if isinstance(loc, dict):
        n.latitude = loc.get("latitude") or loc.get("lat")
        n.longitude = loc.get("longitude") or loc.get("lng") or loc.get("lon")

    # 7) 描述 & key features
    n.description = detail.get("description")
    kf = detail.get("keyFeatures") or detail.get("keyfeatures") or detail.get("highlights")
    if isinstance(kf, list):
        n.key_features = [str(x).strip() for x in kf if str(x).strip()]

    # 8) Agent 信息
    agent = detail.get("customer") or detail.get("agent") or {}
    if isinstance(agent, dict):
        n.agent_name = agent.get("branchName") or agent.get("name")
        n.agent_phone = agent.get("telephone") or agent.get("phone")

    # 9) 媒体：images / floorplan / epc
    def extract_urls(arr: Any) -> List[str]:
        out = []
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    for k in ["url", "imageUrl", "src"]:
                        if it.get(k):
                            out.append(it[k])
                elif isinstance(it, str):
                    out.append(it)
        return out

    n.images = extract_urls(detail.get("images"))
    n.floorplan = extract_urls(detail.get("floorplans") or detail.get("floorplan"))
    n.epc = extract_urls(detail.get("epc") or detail.get("epcGraphs") or detail.get("energyPerformanceCertificate"))

    # 兜底：记录这个块里有哪些字段（方便你补齐“所有信息”）
    if isinstance(detail, dict):
        n.raw_keys_present = sorted(list(detail.keys()))

    return n


def normalize_from_ldjson(ld_list: List[Dict[str, Any]], property_id: str) -> NormalizedListing:
    """
    兜底：如果 __NEXT_DATA__ 找不到，至少从 ld+json 拿到 name/address/price 等。
    """
    url = f"https://www.rightmove.co.uk/properties/{property_id}"
    n = NormalizedListing(property_id=property_id, url=url)

    # 找最像房源的 schema（通常 @type=Residence/Apartment/Offer 等）
    best = None
    for obj in ld_list:
        t = obj.get("@type")
        if t:
            best = obj
            break
    if not best and ld_list:
        best = ld_list[0]

    if not best:
        return n

    n.title = best.get("name")
    addr = best.get("address")
    if isinstance(addr, dict):
        n.address = " ".join([str(addr.get(k, "")).strip() for k in ["streetAddress", "addressLocality", "postalCode"]]).strip() or None
    elif isinstance(addr, str):
        n.address = addr
    n.postcode = _extract_postcode(n.address)

    offers = best.get("offers")
    if isinstance(offers, dict):
        n.price_pcm = _to_int_price(str(offers.get("price")))
        n.rent_frequency = offers.get("priceSpecification", {}).get("unitText") if isinstance(offers.get("priceSpecification"), dict) else None

    geo = best.get("geo")
    if isinstance(geo, dict):
        n.latitude = geo.get("latitude")
        n.longitude = geo.get("longitude")

    return n


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def crawl_canary_wharf_1bed(
    search_url: str,
    limit_listings: int = 50,
    sleep_s: float = 1.5,
    out_dir: str = "rightmove_canarywharf_1bed"
) -> List[NormalizedListing]:
    with httpx.Client(follow_redirects=True) as client:
        search_html = fetch_html(client, search_url)
        property_ids = extract_property_ids_from_search(search_html)[:limit_listings]

        normalized: List[NormalizedListing] = []

        for pid in property_ids:
            detail_url = f"https://www.rightmove.co.uk/properties/{pid}"
            html = fetch_html(client, detail_url)

            next_data, ld_list = extract_embedded_json(html)

            # 保存原始数据（实现“所有信息”最靠谱的一步）
            if next_data:
                save_json(os.path.join(out_dir, "raw_next_data", f"{pid}.json"), next_data)
            if ld_list:
                save_json(os.path.join(out_dir, "raw_ldjson", f"{pid}.json"), ld_list)

            # 标准化输出
            if next_data:
                n = normalize_from_next_data(next_data, pid)
            else:
                n = normalize_from_ldjson(ld_list, pid)

            normalized.append(n)

            time.sleep(sleep_s)

        # 保存结构化结果
        save_json(os.path.join(out_dir, "normalized_listings.json"), [asdict(x) for x in normalized])
        return normalized


if __name__ == "__main__":
    # 你把这里替换成：Rightmove 上 Canary Wharf + 1 bed 的搜索结果 URL
    SEARCH_URL = "https://www.rightmove.co.uk/property-to-rent/find.html?searchLocation=Canary+Wharf%2C+East+London&useLocationIdentifier=true&locationIdentifier=REGION%5E85362&radius=0.0&minBedrooms=1&maxBedrooms=1&_includeLetAgreed=on"

    results = crawl_canary_wharf_1bed(
        search_url=SEARCH_URL,
        limit_listings=30,   # 先小一点，跑通再加
        sleep_s=1.8,
        out_dir="rightmove_canarywharf_1bed"
    )

    # 打印前 3 个看看
    for r in results[:3]:
        print(asdict(r))