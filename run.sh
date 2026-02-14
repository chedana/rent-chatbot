#!/usr/bin/env bash
set -euo pipefail

# Prefer RunPod workspace path; fall back to script directory.
APP_DIR="/workspace/rent-chatbot"
if [[ ! -d "${APP_DIR}" ]]; then
  APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "${APP_DIR}"

# Qdrant
export RENT_QDRANT_PATH="${RENT_QDRANT_PATH:-/workspace/rent-chatbot/artifacts/qdrant_local}"
export RENT_QDRANT_COLLECTION="${RENT_QDRANT_COLLECTION:-rent_listings}"
export RENT_QDRANT_ENABLE_PREFILTER="${RENT_QDRANT_ENABLE_PREFILTER:-1}"
export RENT_PREF_VECTOR_PATH="${RENT_PREF_VECTOR_PATH:-/workspace/rent-chatbot/artifacts/features/pref_vectors.parquet}"

# Retrieval / structured policy
export RENT_RECALL="${RENT_RECALL:-1000}"
export RENT_STRUCTURED_POLICY="${RENT_STRUCTURED_POLICY:-RULE_FIRST}"
export RENT_STRUCTURED_CONFLICT_LOG="${RENT_STRUCTURED_CONFLICT_LOG:-1}"
export RENT_STRUCTURED_CONFLICT_LOG_PATH="${RENT_STRUCTURED_CONFLICT_LOG_PATH:-/workspace/rent-chatbot/artifacts/debug/structured_conflicts.jsonl}"
export RENT_ENABLE_STAGE_D_EXPLAIN="${RENT_ENABLE_STAGE_D_EXPLAIN:-0}"

mkdir -p "/workspace/rent-chatbot/artifacts/debug" || true

echo "[run] APP_DIR=${APP_DIR}"
echo "[run] RENT_QDRANT_PATH=${RENT_QDRANT_PATH}"
echo "[run] RENT_QDRANT_COLLECTION=${RENT_QDRANT_COLLECTION}"
echo "[run] RENT_QDRANT_ENABLE_PREFILTER=${RENT_QDRANT_ENABLE_PREFILTER}"
echo "[run] RENT_PREF_VECTOR_PATH=${RENT_PREF_VECTOR_PATH}"
echo "[run] RENT_RECALL=${RENT_RECALL}"
echo "[run] RENT_STRUCTURED_POLICY=${RENT_STRUCTURED_POLICY}"
echo "[run] RENT_STRUCTURED_CONFLICT_LOG=${RENT_STRUCTURED_CONFLICT_LOG}"
echo "[run] RENT_STRUCTURED_CONFLICT_LOG_PATH=${RENT_STRUCTURED_CONFLICT_LOG_PATH}"
echo "[run] RENT_ENABLE_STAGE_D_EXPLAIN=${RENT_ENABLE_STAGE_D_EXPLAIN}"

exec python3 chat_bot.py
