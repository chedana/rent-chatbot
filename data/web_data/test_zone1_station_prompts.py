import argparse
import json
import os
import time
from typing import Any, Dict, List

from test_vauxhall_probe import resolve_with_typeahead


# "St Pancras International" intentionally excluded per user decision.
ZONE1_STATION_PROMPTS = [
    "Aldgate Station",
    "Aldgate East Station",
    "Angel Station",
    "Baker Street Station",
    "Bank Station",
    "Barbican Station",
    "Bayswater Station",
    "Blackfriars Station",
    "Bond Street Station",
    "Borough Station",
    "Cannon Street Station",
    "Chancery Lane Station",
    "Charing Cross Station",
    "City Thameslink Station",
    "Covent Garden Station",
    "Edgware Road Station",
    "Elephant & Castle Station",
    "Embankment Station",
    "Euston Station",
    "Euston Square Station",
    "Farringdon Station",
    "Fenchurch Street Station",
    "Gloucester Road Station",
    "Goodge Street Station",
    "Great Portland Street Station",
    "Green Park Station",
    "High Street Kensington Station",
    "Holborn Station",
    "Hyde Park Corner Station",
    "Kennington Station",
    "King's Cross St. Pancras Station",
    "Knightsbridge Station",
    "Lambeth North Station",
    "Lancaster Gate Station",
    "Leicester Square Station",
    "Liverpool Street Station",
    "London Bridge Station",
    "Mansion House Station",
    "Marble Arch Station",
    "Marylebone Station",
    "Monument Station",
    "Moorgate Station",
    "Old Street Station",
    "Oxford Circus Station",
    "Paddington Station",
    "Piccadilly Circus Station",
    "Pimlico Station",
    "Queensway Station",
    "Regent's Park Station",
    "Russell Square Station",
    "Sloane Square Station",
    "South Kensington Station",
    "Southwark Station",
    "St James's Park Station",
    "St Paul's Station",
    "Temple Station",
    "Tottenham Court Road Station",
    "Tower Hill Station",
    "Vauxhall Station",
    "Victoria Station",
    "Warren Street Station",
    "Waterloo Station",
    "Waterloo East Station",
    "Westminster Station",
]


def test_one(prompt: str, radius: float, verify_final_url: bool) -> Dict[str, Any]:
    try:
        r = resolve_with_typeahead(
            search_location=prompt,
            radius=radius,
            verify_final_url=verify_final_url,
            prefer_region=False,  # station prompts should allow STATION hits first
        )
        location_identifier = str(r.get("location_identifier") or "")
        resolved = str(r.get("resolved_search_location") or "")
        usable = bool(location_identifier)
        return {
            "prompt": prompt,
            "usable": usable,
            "reason": "" if usable else "empty location_identifier",
            "resolved_search_location": resolved,
            "location_identifier": location_identifier or None,
            "display_location_identifier": r.get("display_location_identifier"),
            "candidate_count": r.get("candidate_count"),
            "probe_url": r.get("probe_url"),
            "final_url": r.get("final_url"),
        }
    except Exception as exc:
        return {
            "prompt": prompt,
            "usable": False,
            "reason": str(exc),
            "resolved_search_location": None,
            "location_identifier": None,
            "display_location_identifier": None,
            "candidate_count": None,
            "probe_url": None,
            "final_url": None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch test Zone 1 station prompts and report which prompts are unusable."
    )
    parser.add_argument("--radius", type=float, default=0.0, help="Search radius in miles.")
    parser.add_argument("--sleep-sec", type=float, default=0.05, help="Delay between requests.")
    parser.add_argument(
        "--no-verify-final-url",
        action="store_true",
        help="Skip final URL fetch; only use typeahead resolution.",
    )
    parser.add_argument(
        "--out-json",
        default="data/web_data/zone1_station_prompt_test_results.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_final_url = not args.no_verify_final_url

    results: List[Dict[str, Any]] = []
    total = len(ZONE1_STATION_PROMPTS)

    for i, prompt in enumerate(ZONE1_STATION_PROMPTS, start=1):
        row = test_one(prompt=prompt, radius=args.radius, verify_final_url=verify_final_url)
        results.append(row)
        status = "OK" if row["usable"] else "FAIL"
        print(f"[{i:02d}/{total}] {status} {prompt} -> {row.get('location_identifier')}")
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

    usable = [r for r in results if r["usable"]]
    unusable = [r for r in results if not r["usable"]]

    out = {
        "meta": {
            "total": total,
            "usable": len(usable),
            "unusable": len(unusable),
            "verify_final_url": verify_final_url,
            "radius": args.radius,
        },
        "unusable_prompts": [r["prompt"] for r in unusable],
        "usable_prompts": [r["prompt"] for r in usable],
        "results": results,
    }

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("\n=== Summary ===")
    print(f"Total: {total}")
    print(f"Usable: {len(usable)}")
    print(f"Unusable: {len(unusable)}")
    print(f"Saved: {args.out_json}")


if __name__ == "__main__":
    main()
