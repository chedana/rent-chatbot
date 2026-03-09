# clean_properties.py
import json
import math
from typing import Any, Dict

IN_PATH  = "properties.jsonl"
OUT_PATH = "properties_clean.jsonl"

ASK = "Ask agent"

def is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)

def to_str(v: Any) -> str:
    """
    强制把任何值变成 string
    """
    if v is None or is_nan(v):
        return ASK

    # list -> join
    if isinstance(v, list):
        items = []
        for x in v:
            if x is None or is_nan(x):
                continue
            s = str(x).strip()
            if s:
                items.append(s)
        return " | ".join(items) if items else ASK

    # dict -> json string
    if isinstance(v, dict):
        try:
            s = json.dumps(v, ensure_ascii=False)
            return s if s.strip() else ASK
        except Exception:
            return ASK

    # everything else
    s = str(v).strip()
    return s if s else ASK


def clean_one(rec: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in rec.items():
        out[k] = to_str(v)
    return out


def main():
    n = 0
    with open(IN_PATH, "r", encoding="utf-8") as fin, \
         open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cleaned = clean_one(rec)
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            n += 1

    print(f"Done. Cleaned {n} records -> {OUT_PATH}")


if __name__ == "__main__":
    main()