#!/usr/bin/env python3
import json
import os
import re
import sys
from difflib import SequenceMatcher
from datetime import datetime
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from helpers import expand_location_keyword_candidates
MENTAL_DIR = os.path.join(ROOT, "artifacts", "London_zone_1", "mental_region")
STATION_DIR = os.path.join(ROOT, "artifacts", "London_zone_1", "underground")
OUT_FILE = os.path.join(ROOT, "test", "stageA", "stageA_location_typo_abbrev_test_cases.json")


def _is_data_folder(name: str) -> bool:
    if not name or name.startswith("."):
        return False
    if name.endswith(".json"):
        return False
    return os.path.isdir(name)


def _tokenize_slug(slug: str) -> List[str]:
    return [t for t in slug.split("_") if t]


def _mental_canonical_from_folder(folder: str) -> str:
    slug = folder
    if slug.endswith("_london"):
        slug = slug[: -len("_london")]
    for suffix in ("_south_east", "_south_west", "_north_west", "_north_east", "_central"):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
    return slug.replace("_", " ").strip()


def _station_canonical_from_folder(folder: str) -> str:
    slug = folder
    if slug.endswith("_station"):
        slug = slug[: -len("_station")]
    return slug.replace("_", " ").strip()


def _initialism(tokens: List[str]) -> str:
    if not tokens:
        return ""
    return "".join(t[0] for t in tokens if t and t[0].isalnum())


def _drop_vowels(word: str) -> str:
    if len(word) <= 4:
        return word
    out = word[0] + re.sub(r"[aeiou]", "", word[1:])
    return out if len(out) >= 3 else word


def _delete_one_char(word: str) -> str:
    if len(word) <= 4:
        return word
    i = max(1, len(word) // 2)
    return word[:i] + word[i + 1 :]


def _transpose_one(word: str) -> str:
    if len(word) <= 4:
        return word
    i = min(2, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def _shorten_multi(tokens: List[str]) -> str:
    parts = []
    for t in tokens:
        if t == "s":
            parts.append(t)
            continue
        if len(t) <= 4:
            parts.append(t)
        else:
            parts.append(t[:3])
    return " ".join(parts)


def _candidate_pool(canonical: str, category: str) -> List[Tuple[str, str]]:
    toks = canonical.split()
    first = toks[0] if toks else canonical
    cands: List[Tuple[str, str]] = []

    manual: Dict[str, List[str]] = {
        "tottenham court road": ["tcr", "tot ct rd", "tottenham crt rd"],
        "king s cross st pancras": ["kx", "kings x st pancras", "kx st p"],
        "london bridge": ["lbg", "ldn bridge", "ldn brg"],
        "waterloo": ["wloo", "wtaerloo", "waterlo"],
        "victoria": ["vic", "victora", "vict stn"],
        "south kensington": ["south ken", "s ken", "s kensngtn"],
        "covent garden": ["cov gdn", "covnt garden", "covent grdn"],
        "st james s park": ["st jamess park", "st james pk", "sjp"],
        "st james s": ["st jamess", "st james", "st jms"],
        "elephant castle": ["e&c", "elephant n castle", "elephnt castle"],
        "oxford circus": ["oxf circ", "oxford crcs", "oxc"],
        "green park": ["gpk", "green pk", "grn park"],
        "leicester square": ["lc sq", "leicster sq", "leicester sq"],
        "regent s park": ["regents pk", "regent park", "regnts park"],
        "great portland street": ["great portland st", "grt portland street", "portland st"],
        "high street kensington": ["high street kens", "high st kensington", "high st ken"],
        "waterloo east": ["waterloo e", "wloo east", "waterloo east stn"],
        "king s cross st pancras": [
            "kings x",
            "kingx",
            "kingsx",
            "king cross st pancras",
            "kings cross",
            "kings x st pancras",
        ],
        "st paul s": ["st pauls", "st pauls stn", "st pauls station"],
    }
    for m in manual.get(canonical, []):
        risk = "high_risk_abbrev" if canonical == "king s cross st pancras" and ("x" in m) else "manual_abbrev_or_typo"
        cands.append((m, risk))

    if len(toks) >= 2:
        init = _initialism([t for t in toks if t not in {"s", "st"}])
        if len(init) >= 3:
            cands.append((init, "initialism"))
        cands.append((_shorten_multi(toks), "shorten_words"))
        cands.append((" ".join([_delete_one_char(t) if len(t) >= 6 else t for t in toks]), "delete_char_multi"))

    typo_base = first
    if len(typo_base) <= 3:
        for t in toks:
            if len(t) >= 5 and t not in {"station", "street", "park", "road"}:
                typo_base = t
                break
    if len(typo_base) >= 4:
        cands.append((_delete_one_char(typo_base), "delete_char_first_token"))
        cands.append((_transpose_one(typo_base), "transpose_first_token"))
        cands.append((_drop_vowels(typo_base), "drop_vowels_first_token"))
    if len(typo_base) >= 6:
        cands.append((typo_base[:5], "prefix_first_token"))

    if category == "station" and len(first) >= 4:
        cands.append((f"{first} stn", "station_suffix"))

    seen = set()
    out: List[Tuple[str, str]] = []
    for text, reason in cands:
        t = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not t or len(t) < 2:
            continue
        if t == canonical:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append((t, reason))
    return out


def _is_match(top1: str, canonical: str, folder: str, category: str) -> bool:
    t = (top1 or "").strip().lower()
    c = canonical.strip().lower()
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

    # Allow close aliases like "regents park" -> "regent s park".
    t_compact = re.sub(r"[^a-z0-9]", "", t)
    c_compact = re.sub(r"[^a-z0-9]", "", c)
    f_compact = re.sub(r"[^a-z0-9]", "", folder_plain)
    if t_compact in {c_compact, f_compact}:
        return True
    if c_compact in t_compact or f_compact in t_compact:
        return True
    if t_compact and c_compact:
        ratio = SequenceMatcher(None, t_compact, c_compact).ratio()
        if ratio >= 0.86:
            return True
    # Shared token overlap for alias-like forms, e.g. "cross st pancras" vs
    # "king s cross st pancras".
    t_tokens = set(re.findall(r"[a-z0-9]+", t))
    c_tokens = set(re.findall(r"[a-z0-9]+", c))
    if len(t_tokens.intersection(c_tokens)) >= max(2, min(len(c_tokens), 3)):
        return True
    return False


def _build_for_one(folder: str, category: str) -> Dict:
    canonical = (
        _station_canonical_from_folder(folder)
        if category == "station"
        else _mental_canonical_from_folder(folder)
    )
    pool = _candidate_pool(canonical, category)

    chosen = []
    rejected = []
    for variant, reason in pool:
        tops = expand_location_keyword_candidates(variant, limit=3, min_score=0.80)
        top1 = tops[0] if tops else ""
        if _is_match(top1, canonical, folder, category):
            chosen.append(
                {
                    "input": variant,
                    "expected_corrected": top1,
                    "reason": reason,
                    "top3_preview": tops,
                }
            )
        else:
            rejected.append(
                {
                    "input": variant,
                    "reason": reason,
                    "top3_preview": tops,
                }
            )
        if len(chosen) >= 5:
            break

    # Force-include high-risk abbreviations for King's Cross family to keep
    # stress cases visible in the dataset.
    if category == "station" and canonical == "king s cross st pancras":
        forced = ["kings x", "kingx", "kingsx"]
        seen_inputs = {x["input"] for x in chosen}
        for raw in forced:
            if raw in seen_inputs:
                continue
            tops = expand_location_keyword_candidates(raw, limit=3, min_score=0.80)
            top1 = tops[0] if tops else ""
            chosen.append(
                {
                    "input": raw,
                    "expected_corrected": top1,
                    "reason": "high_risk_abbrev_forced",
                    "top3_preview": tops,
                }
            )
            seen_inputs.add(raw)

    if len(chosen) > 5:
        forced_items = [x for x in chosen if x.get("reason") == "high_risk_abbrev_forced"]
        normal_items = [x for x in chosen if x.get("reason") != "high_risk_abbrev_forced"]
        chosen = (forced_items + normal_items)[:5]

    # Keep 2-5 variants per name. If not enough stable matches, keep the best available even if unresolved.
    if len(chosen) < 2:
        need = 2 - len(chosen)
        chosen.extend(
            {
                "input": r["input"],
                "expected_corrected": (r["top3_preview"][0] if r.get("top3_preview") else ""),
                "reason": f"fallback_{r['reason']}",
                "top3_preview": r.get("top3_preview") or [],
            }
            for r in rejected[:need]
        )

    return {
        "category": category,
        "canonical_name": canonical,
        "gt_folder": folder,
        "variants": chosen[:5],
        "stats": {
            "candidate_pool_size": len(pool),
            "accepted": len(chosen),
            "rejected": len(rejected),
        },
    }


def _list_folders(path: str) -> List[str]:
    rows = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if _is_data_folder(full):
            rows.append(name)
    return rows


def main() -> None:
    mental_folders = _list_folders(MENTAL_DIR)
    station_folders = _list_folders(STATION_DIR)

    locations = []
    for f in mental_folders:
        locations.append(_build_for_one(f, "mental_region"))
    for f in station_folders:
        locations.append(_build_for_one(f, "station"))

    bad = []
    total_variants = 0
    for loc in locations:
        for v in loc["variants"]:
            total_variants += 1
            top = (v.get("expected_corrected") or "").strip().lower()
            if not _is_match(top, loc["canonical_name"], loc["gt_folder"], loc["category"]):
                bad.append(
                    {
                        "category": loc["category"],
                        "gt_folder": loc["gt_folder"],
                        "canonical_name": loc["canonical_name"],
                        "input": v.get("input"),
                        "expected_corrected": v.get("expected_corrected"),
                        "top3_preview": v.get("top3_preview"),
                    }
                )

    out = {
        "suite": "stageA_location_typo_abbrev_test_cases",
        "description": "Location typo and abbreviation test data for StageA prefilter correction and extraction.",
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "constraints": {
            "variants_per_name": "2-5",
            "match_function": "helpers.expand_location_keyword_candidates(raw, limit=3, min_score=0.80)",
            "target_scope": ["mental_region", "station"],
        },
        "summary": {
            "location_count": len(locations),
            "total_variants": total_variants,
            "unmatched_variants": len(bad),
            "mental_region_count": len(mental_folders),
            "station_count": len(station_folders),
        },
        "locations": locations,
        "unmatched_examples": bad[:30],
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {OUT_FILE}")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
