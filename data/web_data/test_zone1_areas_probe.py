import argparse
import json
import os
import time
from typing import Any, Dict, List

from test_vauxhall_probe import resolve_with_typeahead


LONDON_ZONE1_AREAS = [
    "Soho",
    "Covent Garden",
    "Mayfair",
    "Marylebone",
    "Fitzrovia",
    "Bloomsbury",
    "Holborn",
    "Chinatown",
    "Marble Arch",
    "Belgravia",
    "Knightsbridge",
    "Chelsea",
    "South Kensington",
    "Westminster",
    "Pimlico",
    "St James's",
    "Victoria",
    "Millbank",
    "Paddington",
    "Bayswater",
    "King's Cross",
    "St Pancras",
    "Euston",
    "Regent's Park",
    "Angel",
    "Clerkenwell",
    "Farringdon",
    "Old Street",
    "Shoreditch",
    "Barbican",
    "Liverpool Street",
    "Moorgate",
    "Aldgate",
    "Spitalfields",
    "Temple",
    "Blackfriars",
    "Tower Hill",
    "St Paul's",
    "Waterloo",
    "South Bank",
    "Bankside",
    "London Bridge",
    "Borough",
    "Bermondsey",
    "Elephant & Castle",
    "Vauxhall",
]


def probe_one(area: str, radius: float, verify_final_url: bool) -> Dict[str, Any]:
    query = f"{area}, London"
    result = resolve_with_typeahead(
        search_location=query,
        radius=radius,
        verify_final_url=verify_final_url,
        prefer_region=True,
    )
    resolved = str(result.get("resolved_search_location") or "")
    location_identifier = str(result.get("location_identifier") or "")
    location_type = location_identifier.split("^", 1)[0] if "^" in location_identifier else ""
    is_london = "london" in resolved.lower()
    is_area_like = location_type in {"REGION", "OUTCODE"}
    ok = bool(location_identifier and is_london and is_area_like)
    return {
        "area": area,
        "query": query,
        "ok": ok,
        "is_london": is_london,
        "location_type": location_type,
        "resolved_search_location": result.get("resolved_search_location"),
        "location_identifier": result.get("location_identifier"),
        "display_location_identifier": result.get("display_location_identifier"),
        "candidate_count": result.get("candidate_count"),
        "probe_url": result.get("probe_url"),
        "final_url": result.get("final_url"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch test Rightmove area resolution for London Zone 1 areas.")
    parser.add_argument("--radius", type=float, default=0.0, help="Search radius in miles.")
    parser.add_argument("--sleep-sec", type=float, default=0.15, help="Delay between requests.")
    parser.add_argument(
        "--no-verify-final-url",
        action="store_true",
        help="Skip final URL fetch and only use typeahead result.",
    )
    parser.add_argument(
        "--out-json",
        default="data/web_data/zone1_probe_results.json",
        help="Where to save detailed batch results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_final_url = not args.no_verify_final_url

    rows: List[Dict[str, Any]] = []
    total = len(LONDON_ZONE1_AREAS)
    success = 0

    for idx, area in enumerate(LONDON_ZONE1_AREAS, start=1):
        try:
            row = probe_one(area=area, radius=args.radius, verify_final_url=verify_final_url)
        except Exception as exc:
            row = {
                "area": area,
                "query": f"{area}, London",
                "ok": False,
                "error": str(exc),
            }
        rows.append(row)
        if row.get("ok"):
            success += 1

        status = "OK" if row.get("ok") else "FAIL"
        resolved = row.get("resolved_search_location", "N/A")
        identifier = row.get("location_identifier", "N/A")
        print(f"[{idx:02d}/{total}] {status} {area} -> {resolved} | {identifier}")

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "total": total,
                    "success": success,
                    "failed": total - success,
                    "verify_final_url": verify_final_url,
                    "radius": args.radius,
                },
                "results": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print("\n=== Summary ===")
    print(f"Total: {total}")
    print(f"Success: {success}")
    print(f"Failed: {total - success}")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
