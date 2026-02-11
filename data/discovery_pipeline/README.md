# Zone1 Pipeline (RunPod)

This README describes the exact flow used in RunPod, starting from:

- `artifacts/London_zone_1/zone1_station_prompt_test_results.json`
- `artifacts/London_zone_1/zone1_probe_results.json`

## Step 1: Collect station URLs

```bash
python3 /workspace/rent-chatbot/artifacts/London_zone_1/batch_collect_listing_urls.py \
  --station-results-json /workspace/rent-chatbot/artifacts/London_zone_1/zone1_station_prompt_test_results.json \
  --out-root /workspace/rent-chatbot/artifacts/London_zone_1/underground \
  --summary-json /workspace/rent-chatbot/artifacts/London_zone_1/underground/batch_listing_urls_summary.json \
  --jobs 8 \
  --pages 3 \
  --collect-workers 1
```

## Step 2: Collect region URLs

```bash
python3 /workspace/rent-chatbot/artifacts/London_zone_1/batch_collect_listing_urls.py \
  --station-results-json /workspace/rent-chatbot/artifacts/London_zone_1/zone1_probe_results.json \
  --out-root /workspace/rent-chatbot/artifacts/London_zone_1/mental_region \
  --summary-json /workspace/rent-chatbot/artifacts/London_zone_1/mental_region/batch_listing_urls_summary.json \
  --jobs 8 \
  --pages 3 \
  --collect-workers 1
```

## Step 3: Build global query config from all station/region folders

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/workspace/rent-chatbot/artifacts/London_zone_1")
out = Path("/workspace/rent-chatbot/data/discovery_pipeline/queries_zone1_all_runpod.json")
rows = []

for d in sorted((root / "underground").iterdir()):
    f = d / "listing_urls.txt"
    if d.is_dir() and f.exists():
        rows.append({"method":"station","query":d.name,"urls_file":str(f),"slug":d.name})

for d in sorted((root / "mental_region").iterdir()):
    f = d / "listing_urls.txt"
    if d.is_dir() and f.exists():
        rows.append({"method":"region","query":d.name,"urls_file":str(f),"slug":d.name})

out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", len(rows), "queries ->", out)
PY
```

## Step 4: Global dedupe only (no detail crawl)

```bash
python3 /workspace/rent-chatbot/data/discovery_pipeline/build_global_dataset.py \
  --queries-file /workspace/rent-chatbot/data/discovery_pipeline/queries_zone1_all_runpod.json \
  --run-name zone1_global_dedup_only_runpod \
  --index-workers 16
```

Main outputs:

- `/workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/global_listings_simulated.jsonl`
- `/workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/dedup_urls_to_crawl.txt`
- `/workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/query_indexes/`

## Step 5: Split dedup URL list into 200-size tasks

This creates many task files, each with at most 200 URLs (`chunk_000`, `chunk_001`, ...).

```bash
mkdir -p /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunks
split -d -a 3 -l 200 \
  /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/dedup_urls_to_crawl.txt \
  /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunks/chunk_
```

Check task count:

```bash
ls -1 /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunks | wc -l
```

## Step 6: Run one crawl task (example: chunk_000)

```bash
mkdir -p /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunk_results
python3 /workspace/rent-chatbot/data/web_data/batch_crawl.py \
  --urls-file /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunks/chunk_000 \
  --out-jsonl /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunk_results/properties_chunk_000.jsonl \
  --source-name rightmove \
  --workers 8 \
  --sleep-sec 0.5
```

Repeat Step 6 for each chunk file.

## Step 6B: One-command run for all chunk tasks

Instead of running 20 tasks manually, run all chunks automatically:

```bash
python3 /workspace/rent-chatbot/data/discovery_pipeline/run_chunk_tasks.py \
  --chunks-dir /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunks \
  --out-dir /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunk_results \
  --crawl-workers 8 \
  --sleep-sec 0.5
```

Outputs:

- `/workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunk_results/properties_chunk_*.jsonl`
- `/workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunk_results/chunk_run_summary.json`

## Step 7: Merge chunk JSONLs

```bash
cat /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/chunk_results/properties_chunk_*.jsonl \
  > /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/properties_all_raw.jsonl
```

## Step 8: Backfill discovery mapping into merged details

```bash
python3 /workspace/rent-chatbot/data/discovery_pipeline/backfill_discovery.py \
  --raw-jsonl /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/properties_all_raw.jsonl \
  --map-jsonl /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/global_listings_simulated.jsonl \
  --out-jsonl /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/global_properties_with_discovery.jsonl \
  --fill-fields let_type,furnish_type,min_tenancy \
  --fill-value "Ask agent"
```

Final file for retrieval/indexing:

- `/workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/global_properties_with_discovery.jsonl`

## Step 9: Build HNSW index from final JSONL

Use the final JSONL generated in Step 8 as input:

```bash
python3 /workspace/rent-chatbot/data/HNSW/build_hnsw_indexes.py \
  --in-jsonl /workspace/rent-chatbot/artifacts/web_data/zone1_global_dedup_only_runpod/global_properties_with_discovery.jsonl \
  --out-dir /workspace/rent-chatbot/artifacts/hnsw/zone1_global
```

Main outputs:

- `/workspace/rent-chatbot/artifacts/hnsw/zone1_global/listings_hnsw.faiss`
- `/workspace/rent-chatbot/artifacts/hnsw/zone1_global/listings_meta.parquet`
- `/workspace/rent-chatbot/artifacts/hnsw/zone1_global/evidence_hnsw.faiss`
- `/workspace/rent-chatbot/artifacts/hnsw/zone1_global/evidence_meta.parquet`

Note:

- If you copied an older command with `--web-input` / `--output-root`, switch to `--in-jsonl` / `--out-dir` for the current `build_hnsw_indexes.py`.
