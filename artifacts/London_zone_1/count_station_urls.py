import argparse
import csv
import json
import os
from typing import Dict, List, Tuple


def count_urls_in_file(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def collect_counts(root_dir: str, filename: str = "listing_urls.txt") -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(root_dir)

    for entry in sorted(os.listdir(root_dir)):
        sub = os.path.join(root_dir, entry)
        if not os.path.isdir(sub):
            continue
        fpath = os.path.join(sub, filename)
        url_count = count_urls_in_file(fpath)
        rows.append(
            {
                "station_slug": entry,
                "url_count": url_count,
                "file": fpath,
                "exists": os.path.exists(fpath),
            }
        )
    return rows


def summary(rows: List[Dict]) -> Dict:
    total_stations = len(rows)
    with_file = sum(1 for r in rows if r["exists"])
    total_urls = sum(int(r["url_count"]) for r in rows)
    zero_count = sum(1 for r in rows if int(r["url_count"]) == 0)
    return {
        "total_stations": total_stations,
        "stations_with_listing_file": with_file,
        "stations_without_listing_file": total_stations - with_file,
        "total_urls": total_urls,
        "stations_with_zero_urls": zero_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count listing URLs per station folder.")
    parser.add_argument(
        "--root-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "underground"),
        help="Root directory containing station subfolders.",
    )
    parser.add_argument(
        "--out-json",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "underground", "station_url_counts.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "underground", "station_url_counts.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    args = parse_args()
    rows = collect_counts(root_dir=args.root_dir, filename="listing_urls.txt")
    stats = summary(rows)

    # Sort for human-readability: most URLs first.
    rows_sorted = sorted(rows, key=lambda x: int(x["url_count"]), reverse=True)

    ensure_parent(args.out_json)
    payload = {"meta": stats, "results": rows_sorted}
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    ensure_parent(args.out_csv)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["station_slug", "url_count", "exists", "file"])
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)

    print("=== Station URL Count Summary ===")
    print(f"Root dir: {args.root_dir}")
    print(f"Total stations: {stats['total_stations']}")
    print(f"With listing_urls.txt: {stats['stations_with_listing_file']}")
    print(f"Without listing_urls.txt: {stats['stations_without_listing_file']}")
    print(f"Total urls: {stats['total_urls']}")
    print(f"Stations with 0 urls: {stats['stations_with_zero_urls']}")
    print(f"JSON: {args.out_json}")
    print(f"CSV: {args.out_csv}")

    print("\nTop 15 stations by URL count:")
    for r in rows_sorted[:15]:
        print(f"- {r['station_slug']}: {r['url_count']}")


if __name__ == "__main__":
    main()
