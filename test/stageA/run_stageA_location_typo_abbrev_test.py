#!/usr/bin/env python3
import json
import os
import re
import sys
from difflib import SequenceMatcher

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from helpers import expand_location_keyword_candidates

DATA_FILE = os.path.join(ROOT, "test", "stageA", "stageA_location_typo_abbrev_test_cases.json")


def is_match(top1: str, canonical: str, folder: str, category: str) -> bool:
    t = (top1 or "").strip().lower()
    c = (canonical or "").strip().lower()
    if not t:
        return False
    if t == c:
        return True
    alias_family = {
        "king s cross st pancras": {"king s", "kings cross", "king s cross", "st pancras"},
    }
    fam = alias_family.get(c)
    if fam and t in fam:
        return True

    folder_base = folder
    if category == "station" and folder_base.endswith("_station"):
        folder_base = folder_base[: -len("_station")]
    if category == "mental_region" and folder_base.endswith("_london"):
        folder_base = folder_base[: -len("_london")]
    folder_plain = folder_base.replace("_", " ")

    if t == folder_plain:
        return True

    t_compact = re.sub(r"[^a-z0-9]", "", t)
    c_compact = re.sub(r"[^a-z0-9]", "", c)
    f_compact = re.sub(r"[^a-z0-9]", "", folder_plain)
    if t_compact in {c_compact, f_compact}:
        return True
    if c_compact in t_compact or f_compact in t_compact:
        return True

    if t_compact and c_compact:
        if SequenceMatcher(None, t_compact, c_compact).ratio() >= 0.86:
            return True

    t_tokens = set(re.findall(r"[a-z0-9]+", t))
    c_tokens = set(re.findall(r"[a-z0-9]+", c))
    if len(t_tokens.intersection(c_tokens)) >= max(2, min(len(c_tokens), 3)):
        return True

    return False


def main() -> int:
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        obj = json.load(f)

    total = 0
    passed = 0
    failed = []

    for loc in obj.get("locations", []):
        category = loc.get("category", "")
        canonical = loc.get("canonical_name", "")
        gt_folder = loc.get("gt_folder", "")
        for v in loc.get("variants", []):
            total += 1
            raw = v.get("input", "")
            top3 = expand_location_keyword_candidates(raw, limit=3, min_score=0.80)
            top1 = top3[0] if top3 else ""
            ok = is_match(top1, canonical, gt_folder, category)
            if ok:
                passed += 1
            else:
                failed.append(
                    {
                        "category": category,
                        "gt_folder": gt_folder,
                        "canonical_name": canonical,
                        "input": raw,
                        "actual_top1": top1,
                        "actual_top3": top3,
                    }
                )

    print(json.dumps(
        {
            "suite": obj.get("suite"),
            "total": total,
            "passed": passed,
            "failed": len(failed),
            "pass_rate": round((passed / total) if total else 0.0, 4),
        },
        ensure_ascii=False,
        indent=2,
    ))

    if failed:
        print("\\nTop failures:")
        for row in failed[:20]:
            print(json.dumps(row, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
