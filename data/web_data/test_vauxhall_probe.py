import argparse
import json
import re
import urllib.error
import urllib.request
from urllib.parse import parse_qs, quote, urlencode, urlparse


def build_probe_url(
    search_location: str,
    radius: float = 0.0,
    sort_type: int = 6,
    location_identifier: str = None,
    display_location_identifier: str = None,
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
    if display_location_identifier:
        params["displayLocationIdentifier"] = display_location_identifier
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
    candidates: list[dict] = []
    tried: list[str] = []
    urls = [
        f"https://www.rightmove.co.uk/typeAhead/uknostreet/{tokenized}/",
        f"https://www.rightmove.co.uk/typeAhead/uknostreet/{tokenized}.html",
        "https://www.rightmove.co.uk/typeAhead/uknostreet/" + quote(search_location) + ".html",
        "https://www.rightmove.co.uk/typeAhead/uknostreet/" + quote(search_location) + "/",
    ]

    for url in urls:
        tried.append(url)
        try:
            payload = fetch_json(url)
            if isinstance(payload.get("typeAheadLocations"), list):
                candidates = payload["typeAheadLocations"]
            elif isinstance(payload.get("items"), list):
                candidates = payload["items"]
            if candidates:
                return candidates
        except urllib.error.HTTPError:
            continue
        except Exception:
            continue

    raise RuntimeError("Typeahead lookup failed for all URL patterns: " + " | ".join(tried))


def pick_best_candidate(search_location: str, candidates: list[dict], prefer_region: bool = True) -> dict:
    if not candidates:
        raise RuntimeError("No Rightmove typeahead candidates returned")

    q = search_location.strip().lower()
    city_hint = ""
    if "," in q:
        city_hint = q.split(",")[-1].strip()
    if not city_hint and " london" in q:
        city_hint = "london"
    best = None
    best_score = -1

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
            score += 100
        if q in d:
            score += 50
        if d.startswith(q.split(",")[0].strip()):
            score += 20
        score += max(0, 10 - abs(len(d) - len(q)))
        if city_hint:
            if city_hint in d:
                score += 160
            else:
                score -= 140
        # Prefer area-like identifiers over transport nodes by default.
        if prefer_region:
            if location_id.startswith("REGION^"):
                score += 120
            elif location_id.startswith("OUTCODE^"):
                score += 80
            elif location_id.startswith("STATION^"):
                score -= 80

        if score > best_score:
            best = c
            best_score = score

    if not best:
        raise RuntimeError("Candidates returned but none had locationIdentifier")
    return best


def fetch_final_url(url: str, timeout: int = 40) -> str:
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
        return resp.geturl()


def resolve_with_typeahead(
    search_location: str,
    radius: float = 0.0,
    verify_final_url: bool = True,
    prefer_region: bool = True,
) -> dict:
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

    probe_url = build_probe_url(
        search_location=resolved_search_location,
        radius=radius,
        location_identifier=location_identifier,
    )
    final_url = probe_url
    display_location_identifier = None

    if verify_final_url:
        try:
            final_url = fetch_final_url(probe_url)
            qs = parse_qs(urlparse(final_url).query)
            display_location_identifier = qs.get("displayLocationIdentifier", [None])[0]
            if qs.get("searchLocation", [None])[0]:
                resolved_search_location = qs["searchLocation"][0]
            if qs.get("locationIdentifier", [None])[0]:
                location_identifier = qs["locationIdentifier"][0]
        except Exception:
            pass

    return {
        "input_query": search_location,
        "probe_url": probe_url,
        "final_url": final_url,
        "resolved_search_location": resolved_search_location,
        "location_identifier": location_identifier,
        "display_location_identifier": display_location_identifier,
        "candidate_count": len(candidates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Rightmove canonical location resolution via typeahead.")
    parser.add_argument("--query", default="Vauxhall, London", help="Input location query")
    parser.add_argument("--radius", type=float, default=0.0, help="Search radius in miles")
    parser.add_argument(
        "--no-verify-final-url",
        action="store_true",
        help="Skip final URL fetch; only resolve from typeahead candidates.",
    )
    parser.add_argument(
        "--allow-station-first",
        action="store_true",
        help="Disable region-first scoring and allow station candidates to rank first.",
    )
    args = parser.parse_args()

    result = resolve_with_typeahead(
        search_location=args.query,
        radius=args.radius,
        verify_final_url=not args.no_verify_final_url,
        prefer_region=not args.allow_station_first,
    )

    print("=== Rightmove Probe Result ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    if args.no_verify_final_url:
        ok = bool(result["location_identifier"])
    else:
        ok = bool(result["location_identifier"] and result["display_location_identifier"])

    if ok:
        print("\nPASS: identifiers resolved")
    else:
        print("\nFAIL: identifiers not resolved")


if __name__ == "__main__":
    main()
