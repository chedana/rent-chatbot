# Stage A Design Notes

## 1) Goal
Stage A is the retrieval stage. It should:
- maximize recall under active constraints,
- avoid noisy prefilter over-pruning,
- provide traceable filtering behavior for debugging.

---

## 2) Current Retrieval Flow
1. Pre Stage A produces normalized constraints (`location_keywords`, `layout_options`, etc.).
2. Stage A optionally applies Qdrant prefilter (`QDRANT_ENABLE_PREFILTER`).
3. Vector search runs on remaining candidates (`recall` top-N).

Main code:
- `/workspace/rent-chatbot/qdrant_search.py`
- `/workspace/rent-chatbot/helpers.py` (location normalization/correction in Pre Stage A)

---

## 3) Location Prefilter Data Sources
Qdrant payload location-related fields (index build side):
- `location_postcode_tokens`
- `location_station_tokens`
- `location_region_tokens`
- `location_tokens` (compat fallback)

Index build source:
- `/workspace/rent-chatbot/data/qdrant/build_qdrant_from_source.py`

Current token policy:
- full `address_norm` removed from prefilter tokens (to reduce noise),
- postcode/station/region kept.

---

## 4) Pre Stage A Location Normalization & Correction
Location correction is performed before Stage A retrieval, using dictionary entries built from:
- Qdrant local storage: `storage.sqlite` payloads (primary source).

Main code:
- `/workspace/rent-chatbot/helpers.py`

Normalization keys:
- `plain_key`
- `slug_key`
- `compact_key`

Matching strategy:
1. `exact` on `plain/slug/compact`
2. `contains` on `compact`
3. `distance` (Damerau-Levenshtein) with adaptive threshold + sliding-window similarity

Decision gate:
- auto-rewrite only when:
  - `best_score >= 0.80`
- rewrite target is the top-1 **alias token** from dictionary entries (not canonical merge).
- no hardcoded station family merge (e.g., no special `king cross` canonicalization).

---

## 5) Scoring Formula (Location Candidate Rewrite)
Per candidate alias token, local score is max of:

1. Contains score:
- if `q_compact in cand_compact` or reverse:
- `score_contains = 0.88 + 0.10 * (shorter_len / longer_len)`

2. Distance score:
- `sim = window_best_similarity(q_compact, cand_compact)`
- pass only if `sim >= min_sim` where:
  - `min_sim = 1 - adaptive_max_ed / len(q_compact)`
- if passed:
  - `len_ratio = min(len(q_compact), len(cand_compact)) / max(len(q_compact), len(cand_compact))`
  - `score_dist = 0.68 + 0.20 * sim + 0.12 * len_ratio`

Global rewrite decision:
- choose top-1 (`best`) candidate score across aliases.
- rewrite to top-1 alias only if gate in section 4 passes.

---

## 6) Adaptive Distance
`adaptive_max_ed(n)`:
- `n <= 6` -> `1`
- `7 <= n <= 12` -> `2`
- `n > 12` -> `round(0.2 * n)` with lower bound `2`

Distance metric:
- Damerau-Levenshtein (supports adjacent transposition).
- Example:
  - `waterloo` vs `wtaerloo0` is more tolerant than plain Levenshtein.

---

## 7) Known Boundaries
- Stage A prefilter uses token equality (`MatchAny`), not full substring search in Qdrant payload.
- Pre Stage A currently rewrites each user location term to a single top-1 alias. Stage A does not consume top-N fuzzy candidates.
- If rewritten token and indexed token forms are misaligned (e.g., short alias vs other long station alias), prefilter may still miss.
- Mitigation path:
  - enrich station aliases/subphrases in index tokens,
  - keep compatibility fallback (`location_tokens`),
  - optional future change: expand each query keyword to top-N alias candidates before `MatchAny`.

---

## 8) Debug Checklist
When Stage A returns 0 candidates unexpectedly:
1. Check Pre Stage A output `location_keywords` after normalization.
2. Check Stage A trace:
   - grouped location tokens (`postcode/station/region`)
   - prefilter count
3. Verify indexed payload fields exist after rebuild.
4. Confirm `storage.sqlite` path and collection are correct.

---

## 9) Rebuild Triggers
Rebuild Qdrant index when:
- location token extraction logic changes,
- payload schema changes,
- source dataset changes.

Rebuild command reference:
- `/workspace/rent-chatbot/RUNPOD_SESSION_SETUP.md`
