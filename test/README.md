# Test Suite Registry (Rent Chatbot)

This document tracks **what each test set is for**, **what it validates**, and **how to use/maintain it**.

Primary workspace path (RunPod): `/workspace/rent-chatbot/test`

---

## 1) Purpose

The test directory is used to validate behavior consistency across pipeline stages, especially:

- Hard-constraint state updates across multi-turn conversations.
- Deterministic merge behavior (`patch` vs `replace_all`).
- Field-level update semantics (`replace` / `append` / `remove` / `keep`).
- Regression protection for previously fixed bugs.

---

## 2) Directory Standard

Recommended structure:

- `test/stageA/` -> retrieval-focused tests
- `test/stageB/` -> hard-filter and constraint update tests
- `test/stageC/` -> soft preference rerank tests
- `test/stageD/` -> explanation/grounding tests
- `test/stageE/` -> listing-level QA tests

File naming guideline:

- `<stage>_<topic>_test_cases.json`
- Example: `stageB_hard_filter_update_test_cases.json`

---

## 3) Test Set Registry

### 3.1 `test/stageB/stageB_hard_filter_update_summary.json`

- Stage: `Stage B`
- Type: Sequential state-transition dataset
- Status: Active
- Main objective:
  - Verify multi-turn hard-constraint update behavior.
  - Ensure location does **not** accidentally accumulate unless append intent is explicit.
  - Validate consistent behavior for:
    - `update_scope` (`patch` / `replace_all`)
    - `location_update_mode` (`replace` / `append` / `keep` / clear)
    - `layout_update_mode` (`replace` / `append`) and layout removal selectors

What this dataset specifically protects:

- Fix regression where:
  - non-location fields were overwritten,
  - but location was unintentionally merged as union.
- Ensure second query like:
  - `Change to Vauxhall instead.`
  - replaces old location instead of keeping both.

Schema summary:

- Top-level:
  - `suite`
  - `description`
  - `purpose`
  - `expected_outcome`
  - `notes`
  - `cases[]`
- Each case:
  - `id`
  - `query`
  - `expected_update`
  - `expected_state_after`

Execution semantics:

- Cases must be run **in order**.
- `expected_state_after` is evaluated against cumulative state after applying current query.
- Case runner should reset only at suite start (not between cases).

Known boundaries (current):

- Designed for hard constraints and merge intent routing.
- Not intended to validate ranking quality or explanation text quality.
- Some “remove scalar constraint” intents (e.g., remove tenancy) may require dedicated remove-rule expansion in parser logic.

---

## 4) How to Use This Dataset

Recommended runner behavior:

1. Load initial empty constraint state.
2. For each case in ascending `id`:
   - run extraction + repair + merge
   - compare resulting state with `expected_state_after`
   - store pass/fail and diff
3. Emit final report with:
   - pass rate
   - failing case ids
   - per-field diffs

Minimum pass condition:

- All cases must pass.
- Any mismatch in `location_keywords`, `layout_options`, or scope/mode behavior is considered regression.

---

## 5) Add a New Test Set (Checklist)

When adding a new dataset, include:

1. Goal statement:
   - One sentence describing the behavior under test.
2. Stage scope:
   - One stage only unless explicitly marked cross-stage.
3. Input format:
   - Query-level, listing-level, or end-to-end.
4. Expected outputs:
   - Deterministic fields only.
5. Edge cases:
   - At least 3 edge cases per new behavior.
6. This registry update:
   - Add a new subsection under **Test Set Registry**.

---

## 6) Maintenance Rules

- Keep datasets deterministic (no randomness).
- Do not mix product requirements and model creativity checks in one file.
- If schema changes:
  - update dataset fields,
  - update this README section,
  - record migration notes in commit message.

