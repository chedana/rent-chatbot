import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


DEFAULT_FILL_FIELDS = ["let_type", "furnish_type", "min_tenancy"]
DEFAULT_FILL_VALUE = "Ask agent"


def listing_id_from_url(url: str) -> str:
    m = re.search(r"/properties/(\d+)", url or "")
    return f"rightmove:{m.group(1)}" if m else f"rightmove:{url}"


def is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        s = v.strip().lower()
        return s == "" or s == "null" or s == "none"
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill discovery_queries_by_method and fill missing fields with Ask agent.",
    )
    parser.add_argument("--raw-jsonl", required=True, help="Merged raw details JSONL (e.g. properties_all_raw.jsonl).")
    parser.add_argument("--map-jsonl", required=True, help="Global simulated mapping JSONL (global_listings_simulated.jsonl).")
    parser.add_argument("--out-jsonl", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--fill-fields",
        default=",".join(DEFAULT_FILL_FIELDS),
        help="Comma-separated fields to fill when missing/null/empty.",
    )
    parser.add_argument("--fill-value", default=DEFAULT_FILL_VALUE, help="Replacement value for missing fields.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_jsonl = Path(args.raw_jsonl)
    map_jsonl = Path(args.map_jsonl)
    out_jsonl = Path(args.out_jsonl)
    fill_fields: List[str] = [x.strip() for x in args.fill_fields.split(",") if x.strip()]

    mapping: Dict[str, Dict] = {}
    with map_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lid = str(rec.get("listing_id") or "").strip()
            if not lid:
                continue
            mapping[lid] = rec.get("discovery_queries_by_method", {})

    rows = 0
    filled_counts = {k: 0 for k in fill_fields}
    with raw_jsonl.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["listing_id"] = listing_id_from_url(str(rec.get("url", "")))
            rec["source_site"] = rec.get("source", "rightmove")
            rec["discovery_queries_by_method"] = mapping.get(rec["listing_id"], {})

            for f in fill_fields:
                if is_missing(rec.get(f)):
                    rec[f] = args.fill_value
                    filled_counts[f] += 1

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows += 1

    print(
        json.dumps(
            {
                "rows_written": rows,
                "out_jsonl": str(out_jsonl),
                "fill_fields": fill_fields,
                "fill_value": args.fill_value,
                "filled_counts": filled_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
