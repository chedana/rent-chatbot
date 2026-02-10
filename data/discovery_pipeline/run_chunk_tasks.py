import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_CRAWL = PROJECT_ROOT / "data" / "web_data" / "batch_crawl.py"


@dataclass
class Task:
    chunk_file: Path
    out_jsonl: Path
    log_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all chunk crawl tasks automatically with bounded concurrency.",
    )
    parser.add_argument("--chunks-dir", required=True, help="Directory containing chunk files (e.g., chunk_000).")
    parser.add_argument("--out-dir", required=True, help="Directory to write chunk result JSONL files.")
    parser.add_argument("--logs-dir", default=None, help="Directory to write per-task logs.")
    parser.add_argument("--chunk-glob", default="chunk_*", help="Glob pattern for chunk files.")
    parser.add_argument("--max-parallel", type=int, default=2, help="How many chunk tasks to run in parallel.")
    parser.add_argument("--crawl-workers", type=int, default=8, help="batch_crawl workers per chunk task.")
    parser.add_argument("--sleep-sec", type=float, default=0.5, help="batch_crawl sleep-sec.")
    parser.add_argument("--source-name", default="rightmove", help="Source label.")
    parser.add_argument("--python-bin", default="python3", help="Python executable.")
    parser.add_argument("--batch-crawl-script", default=str(DEFAULT_BATCH_CRAWL), help="Path to batch_crawl.py.")
    parser.add_argument("--rerun-existing", action="store_true", help="Rerun tasks even if output JSONL exists.")
    parser.add_argument("--stop-on-fail", action="store_true", help="Stop submitting new tasks after first failure.")
    return parser.parse_args()


def discover_tasks(args: argparse.Namespace) -> List[Task]:
    chunks_dir = Path(args.chunks_dir)
    out_dir = Path(args.out_dir)
    logs_dir = Path(args.logs_dir) if args.logs_dir else out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted([p for p in chunks_dir.glob(args.chunk_glob) if p.is_file()])
    tasks: List[Task] = []
    for c in chunk_files:
        name = c.name
        out_jsonl = out_dir / f"properties_{name}.jsonl"
        log_file = logs_dir / f"{name}.log"
        tasks.append(Task(chunk_file=c, out_jsonl=out_jsonl, log_file=log_file))
    return tasks


def run_one_task(task: Task, args: argparse.Namespace) -> Dict:
    if task.out_jsonl.exists() and task.out_jsonl.stat().st_size > 0 and not args.rerun_existing:
        return {
            "chunk": task.chunk_file.name,
            "status": "skipped",
            "out_jsonl": str(task.out_jsonl),
            "log_file": str(task.log_file),
            "reason": "output exists",
        }

    cmd = [
        args.python_bin,
        args.batch_crawl_script,
        "--urls-file",
        str(task.chunk_file),
        "--out-jsonl",
        str(task.out_jsonl),
        "--source-name",
        args.source_name,
        "--workers",
        str(max(1, int(args.crawl_workers))),
        "--sleep-sec",
        str(args.sleep_sec),
    ]

    with task.log_file.open("w", encoding="utf-8") as log:
        log.write("CMD: " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=log, text=True)

    return {
        "chunk": task.chunk_file.name,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "out_jsonl": str(task.out_jsonl),
        "log_file": str(task.log_file),
    }


def main() -> None:
    args = parse_args()
    tasks = discover_tasks(args)
    if not tasks:
        raise RuntimeError(f"No chunk files found in {args.chunks_dir} with pattern {args.chunk_glob}")

    summary: List[Dict] = []
    max_parallel = max(1, int(args.max_parallel))
    stop_submit = False

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {}
        task_iter = iter(tasks)

        # Seed initial workers
        for _ in range(max_parallel):
            try:
                t = next(task_iter)
            except StopIteration:
                break
            futures[pool.submit(run_one_task, t, args)] = t

        while futures:
            done = []
            for fut in as_completed(list(futures.keys())):
                t = futures.pop(fut)
                result = fut.result()
                summary.append(result)
                print(f"[{result['status'].upper()}] {t.chunk_file.name} -> {result['out_jsonl']}")
                if result["status"] == "failed" and args.stop_on_fail:
                    stop_submit = True
                done.append(True)
                break

            if not stop_submit:
                try:
                    nxt = next(task_iter)
                    futures[pool.submit(run_one_task, nxt, args)] = nxt
                except StopIteration:
                    pass

    ok = sum(1 for x in summary if x["status"] == "ok")
    failed = sum(1 for x in summary if x["status"] == "failed")
    skipped = sum(1 for x in summary if x["status"] == "skipped")

    out_dir = Path(args.out_dir)
    summary_path = out_dir / "chunk_run_summary.json"
    payload = {
        "meta": {
            "total_tasks": len(tasks),
            "ok": ok,
            "failed": failed,
            "skipped": skipped,
            "max_parallel": max_parallel,
            "crawl_workers_per_task": int(args.crawl_workers),
            "sleep_sec": float(args.sleep_sec),
        },
        "results": summary,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
