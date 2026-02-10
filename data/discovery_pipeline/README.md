# Discovery Pipeline (Station + Region)

This pipeline is separate from `data/web_data/` and keeps your existing flow unchanged.

## What it does

1. Collect listing URLs from multiple queries (`station` / `region`).
2. Dedupe by listing key (`rightmove:<property_id>`) before detail crawling.
3. Crawl each unique listing only once.
4. Output one JSONL with `discovery_paths` per listing.

## Input format

Use a JSON array file like `queries_example.json`:

```json
[
  {"method": "station", "query": "South Quay Station"},
  {"method": "region", "query": "Canary Wharf, East London"}
]
```

## Run

```bash
python3 data/discovery_pipeline/run_discovery_pipeline.py \
  --queries-file data/discovery_pipeline/queries_example.json \
  --run-name canary_wharf_station_region \
  --pages 3 \
  --collect-workers 1 \
  --crawl-workers 4
```

## Main outputs

- `artifacts/web_data/<run_name>/listing_urls_deduped.txt`
- `artifacts/web_data/<run_name>/dedupe_summary.json`
- `artifacts/web_data/<run_name>/properties_raw_deduped.jsonl`
- `artifacts/web_data/<run_name>/properties_with_discovery.jsonl`

## Agent-Enhanced Variant

If you also want `agent_name`, `agent_address`, `agent_phone` in crawled output, use:

```bash
python3 data/discovery_pipeline/build_global_dataset_with_agent.py \
  --queries-file data/discovery_pipeline/queries_aldgate_three.json \
  --run-name aldgate_global_with_agent \
  --index-workers 8 \
  --crawl-details \
  --crawl-workers 8 \
  --sleep-sec 0.5
```

This writes:

- `artifacts/web_data/<run_name>/global_properties_with_discovery_and_agent.jsonl`
