#!/usr/bin/env python3
"""
Replace all JSON null values in a .jsonl file with the string "Ask agent".

Usage:
  python none_to_ask_agent.py properties.jsonl
  python none_to_ask_agent.py properties.jsonl -o properties.cleaned.jsonl
  python none_to_ask_agent.py properties.jsonl --inplace
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import tempfile
from typing import Any

ASK_AGENT = "Ask agent"

def replace_nulls(obj: Any) -> Any:
    """Recursively replace None (JSON null) with 'Ask agent' in dicts/lists."""
    if obj is None:
        return ASK_AGENT
    if isinstance(obj, dict):
        return {k: replace_nulls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_nulls(v) for v in obj]
    return obj

def process_jsonl(in_path: str, out_path: str) -> None:
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no} in {in_path}: {e}") from e

            cleaned = replace_nulls(record)
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to properties.jsonl")
    ap.add_argument("-o", "--output", default=None, help="Output path (default: <input>.ask_agent.jsonl)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite the input file safely")
    args = ap.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"Error: file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    if args.inplace and args.output:
        print("Error: use either --inplace or --output, not both.", file=sys.stderr)
        sys.exit(1)

    if args.inplace:
        # Safe in-place overwrite via temp file + atomic replace
        dir_name = os.path.dirname(os.path.abspath(in_path)) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="tmp_", suffix=".jsonl", dir=dir_name)
        os.close(fd)
        try:
            process_jsonl(in_path, tmp_path)
            os.replace(tmp_path, in_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        print(f"Updated in place: {in_path}")
        return

    out_path = args.output or (os.path.splitext(in_path)[0] + "_clean.jsonl")
    process_jsonl(in_path, out_path)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()