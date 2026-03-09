# Stage C Design Notes

## 1) Goal
Stage C is a soft rerank stage on top of Stage B pass set.
It should:
- keep unknown-pass behavior from Stage B (do not hard drop unknown rows)
- rank by preference quality + listing quality signals
- stay explainable in debug/summary outputs

## 2) Scope (Current)

### P0 implemented
- `deposit_score`
- `freshness_score` from `added_date`
- merged into `final_score`
- summary/debug explanations updated

### Preference pipeline (implemented)
- preference terms are extracted by LLM (`general_semantic`)
- preference matching uses sidecar vectors (no legacy text fallback)
- sidecar source fields:
  - `features`: each bullet encoded separately
  - `description`: split by `<PARA>`, each segment encoded separately

## 3) Final Score Formula

Current Stage C score:

`final = w_transit*transit + w_school*school + w_preference*preference - w_penalty*penalty + w_deposit*deposit + w_freshness*freshness`

Notes:
- `deposit` and `freshness` are soft boosts
- `penalty` is subtractive
- `transit/school` may be zero if no intent terms

## 4) Deposit/Freshness Logic

### Deposit
- parse `£number`, `£0`, and unknown tokens (`Ask agent`, etc.)
- unknown/missing follows policy (`light_penalty` default)
- uses smooth decay: `deposit_score = exp(-deposit / tau)`
- `tau` configurable via `DEPOSIT_SCORE_TAU`

### Freshness
- parse `added_date` and decay by age days
- half-life configurable
- missing/unparseable follows missing policy
- current default weights:
  - `w_deposit = 0.05`
  - `w_freshness = 0.06`

## 5) Preference Matching Design

### Sidecar-only behavior
- `preference_source = sidecar_vectors` when sidecar hit
- `preference_source = sidecar_missing_no_fallback` when sidecar miss
- no fallback to legacy text scoring

### Aggregation (implemented)
- per-intent field score:
  - take top-2 on `features`, then `features_field_score = 0.7*top1 + 0.3*top2`
  - take top-2 on `description`, then `description_field_score = 0.7*top1 + 0.3*top2`
  - weighted combine by field weights
- multi-intent group score:
  - simple mean across intent scores:
  - `group_score = mean(intent_scores)`

## 6) Debug/Explainability

In debug mode, each listing shows:
- component scores and weighted contributions
- `preference_calc` details:
  - per-intent score
  - `features_field_score`
  - `description_field_score`
  - `field_agg=0.7*top1+0.3*top2`
  - top2 matched lines (with similarity + text)
- `preference_top_matches`:
  - top2 per preference signal (`pref='...'`)
  - includes matched field (`features`/`description`) and text snippet

## 7) RULE_FIRST Clarification

Structured constraints use dual extraction (LLM + rule).
Expected policy:
- if conflict: prefer rule
- if rule missing and LLM present: allow LLM

Fix applied for `let_type` drift:
- prevent invalid value like `student` from landing as hard `let_type`
- keep `let_type` aligned with rule-intended domain (`short term` / `long term`)

## 8) Availability Handling

`available_from = Now` behavior:
- hard filter treats `Now` as immediate availability pass (`now_pass`)
- no date-fail penalty for this case

## 9) Sidecar Data Location

Current runtime path is controlled by:
- `RENT_PREF_VECTOR_PATH`

Current `run.sh` default:
- `/workspace/rent-chatbot/artifacts/features/pref_vectors.parquet`

## 10) Tuning Checklist

When results look wrong:
1. Check `preference_source` (sidecar hit or miss)
2. Check per-intent top2 matched lines in debug
3. Compare preference contribution vs deposit/freshness contribution
4. Tune in this order:
   - field aggregation (`top1/top2` weights)
   - `w_deposit`, `w_freshness`
   - `DEPOSIT_SCORE_TAU`, `FRESHNESS_HALF_LIFE_DAYS`

## 11) Implementation Status (Completed)

### A. Score pipeline
- `final_score` includes:
  - `transit_score`
  - `school_score`
  - `preference_score`
  - `deposit_score`
  - `freshness_score`
  - `penalty_score` (subtractive)

### B. Deposit score
- Parses `£number`, `£0`, `Ask agent/unknown`.
- Uses smooth decay:
  - `deposit_score = exp(-deposit / tau)`
- Config:
  - `W_DEPOSIT` (default `0.05`)
  - `DEPOSIT_SCORE_TAU` (default `3000`)
  - `DEPOSIT_MISSING_POLICY`

### C. Freshness score
- Based on `added_date` age decay with half-life.
- Missing/unparseable values follow policy.
- Config:
  - `W_FRESHNESS` (default `0.06`)
  - `FRESHNESS_HALF_LIFE_DAYS` (default `14`)
  - `FRESHNESS_MISSING_POLICY`

### D. Preference pipeline (sidecar)
- Preference terms come from LLM (`general_semantic`).
- Sidecar-only matching (no legacy text fallback).
- Sidecar encoding:
  - `features`: one vector per bullet
  - `description`: one vector per `<PARA>` segment
- Per-field aggregation:
  - `field_score = 0.7*top1 + 0.3*top2`
- Multi-intent aggregation:
  - `group_score = mean(intent_scores)`

### E. Sidecar runtime behavior
- Path from `RENT_PREF_VECTOR_PATH`.
- Key matching by:
  - primary: `url`
  - fallback: `listing_id`
- Miss behavior:
  - `preference_source = sidecar_missing_no_fallback`
  - `preference_score = 0`

### F. Explainability output
- Summary mode:
  - `Because matched ...`
  - `Because boosted by deposit/freshness ...` when triggered
- Debug mode:
  - component scores + weighted contributions + `final_calc`
  - `preference_source`
  - `preference_calc` with top2 lines and field scores
  - `preference_top_matches`: top2 per preference signal

### G. Rule consistency and parser fixes
- `let_type` drift guardrail fixed (invalid values like `student` no longer become hard `let_type`).
- Rule parser updated to support hyphen patterns like `1-bed` / `2-bedroom`.
- `available_from = Now` supports hard-filter `now_pass`.

### H. Artifacts and test assets
- Stage C test cases:
  - `/workspace/rent-chatbot/test/stageC/stageC_cases.json`
- Stage C notes:
  - `/workspace/rent-chatbot/docs/stagec_notes.md`
- `run.sh` sidecar default:
  - `/workspace/rent-chatbot/artifacts/features/pref_vectors.parquet`
