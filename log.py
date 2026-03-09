import json
import os
from typing import Any, Dict, List

LOG_LEVEL = os.environ.get("RENT_LOG_LEVEL", "INFO").strip().upper()
RANKING_LOG_DETAIL = os.environ.get("RENT_RANKING_LOG_DETAIL", "summary").strip().lower()
RANKING_LOG_MAX_CANDIDATES = int(os.environ.get("RENT_RANKING_LOG_MAX_CANDIDATES", "6"))
RANKING_LOG_TEXT_LIMIT = int(os.environ.get("RENT_RANKING_LOG_TEXT_LIMIT", "320"))

_LOG_LEVEL_ORDER = {"ERROR": 40, "WARN": 30, "INFO": 20, "DEBUG": 10}


def _should_log(level: str) -> bool:
    current = _LOG_LEVEL_ORDER.get(LOG_LEVEL, 20)
    target = _LOG_LEVEL_ORDER.get(level.upper(), 20)
    return target >= current


def log_message(level: str, msg: str) -> None:
    lvl = level.upper()
    if _should_log(lvl):
        print(f"[{lvl}] {msg}")


def _truncate_text(v: Any, limit: int) -> str:
    s = str(v or "")
    if len(s) <= limit:
        return s
    return s[:limit] + "...(truncated)"


def _compact_candidate_list(items: List[Dict[str, Any]], max_items: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for x in (items or [])[:max_items]:
        out.append(
            {
                "rank": x.get("rank"),
                "url": x.get("url"),
                "title": x.get("title"),
                "score": x.get("score"),
                "score_formula": x.get("score_formula"),
                "hard_pass": x.get("hard_pass"),
                "hard_fail_reasons": x.get("hard_fail_reasons"),
                "hits": x.get("hits"),
            }
        )
    return out


def append_ranking_log_entry(path: str, obj: Dict[str, Any]) -> None:
    try:
        payload = obj
        if RANKING_LOG_DETAIL != "full":
            from helpers import compact_constraints_view

            counts = obj.get("counts", {}) if isinstance(obj, dict) else {}
            stage_d = obj.get("stage_d", {}) if isinstance(obj, dict) else {}
            payload = {
                "timestamp": obj.get("timestamp"),
                "log_path": obj.get("log_path"),
                "user_query": obj.get("user_query"),
                "stage_a_query": obj.get("stage_a_query"),
                "counts": counts,
                "constraints": compact_constraints_view(obj.get("constraints") or {}),
                "structured_conflict_count": int((obj.get("structured_audit") or {}).get("conflict_count", 0)),
                "semantic_parse_source": (obj.get("signals") or {}).get("semantic_debug", {}).get("parse_source"),
                "stage_a_candidates": _compact_candidate_list(
                    obj.get("stage_a_candidates") or [],
                    RANKING_LOG_MAX_CANDIDATES,
                ),
                "stage_b_pass_candidates": _compact_candidate_list(
                    obj.get("stage_b_pass_candidates") or [],
                    RANKING_LOG_MAX_CANDIDATES,
                ),
                "stage_c_candidates": _compact_candidate_list(
                    obj.get("stage_c_candidates") or [],
                    RANKING_LOG_MAX_CANDIDATES,
                ),
                "stage_d": {
                    "enabled": stage_d.get("enabled"),
                    "error": stage_d.get("error"),
                    "output_preview": _truncate_text(stage_d.get("output"), RANKING_LOG_TEXT_LIMIT),
                },
            }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as e:
        log_message("WARN", f"failed to write ranking log: {e}")


def append_jsonl(path: str, obj: Dict[str, Any], log_name: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        log_message("WARN", f"failed to write {log_name}: {e}")
