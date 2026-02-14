import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

QDRANT_LOCAL_PATH = os.environ.get(
    "RENT_QDRANT_PATH",
    os.path.join(ROOT_DIR, "artifacts", "qdrant_local"),
)
QDRANT_COLLECTION = os.environ.get("RENT_QDRANT_COLLECTION", "rent_listings")
QDRANT_ENABLE_PREFILTER = os.environ.get("RENT_QDRANT_ENABLE_PREFILTER", "1") != "0"

VERBOSE_STATE_LOG = os.environ.get("RENT_VERBOSE_STATE_LOG", "0") == "1"
STAGEA_TRACE = os.environ.get("RENT_STAGEA_TRACE", "1") != "0"

EMBED_MODEL = os.environ.get("RENT_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BATCH = int(os.environ.get("RENT_EMBED_BATCH", "256"))

DEFAULT_K = int(os.environ.get("RENT_K", "5"))
DEFAULT_RECALL = int(os.environ.get("RENT_RECALL", "1000"))

# Stage C: unknown-pass penalties for active hard constraints.
# These are intentionally conservative defaults and should be tuned with offline eval.
UNKNOWN_PENALTY_WEIGHTS = {
    "price": 0.20,
    "bedrooms": 0.16,
    "bathrooms": 0.12,
    "available_from": 0.14,
    "furnish_type": 0.08,
    "let_type": 0.10,
    "min_tenancy_months": 0.08,
    "min_size_sqm": 0.10,
    "property_type": 0.12,
}
UNKNOWN_PENALTY_CAP = 0.60

RANKING_LOG_PATH = os.environ.get(
    "RENT_RANKING_LOG_PATH",
    os.path.join(ROOT_DIR, "artifacts", "debug", "ranking_log.jsonl"),
)

STRUCTURED_POLICY = os.environ.get("RENT_STRUCTURED_POLICY", "RULE_FIRST").strip().upper()
if STRUCTURED_POLICY not in {"RULE_FIRST", "HYBRID", "LLM_FIRST"}:
    STRUCTURED_POLICY = "RULE_FIRST"

STRUCTURED_CONFLICT_LOG_PATH = os.environ.get(
    "RENT_STRUCTURED_CONFLICT_LOG_PATH",
    os.path.join(ROOT_DIR, "artifacts", "debug", "structured_conflicts.jsonl"),
)
STRUCTURED_TRAINING_LOG_PATH = os.environ.get(
    "RENT_STRUCTURED_TRAINING_LOG_PATH",
    os.path.join(ROOT_DIR, "artifacts", "debug", "structured_training_samples.jsonl"),
)
ENABLE_STRUCTURED_CONFLICT_LOG = os.environ.get("RENT_STRUCTURED_CONFLICT_LOG", "1") != "0"
ENABLE_STRUCTURED_TRAINING_LOG = os.environ.get("RENT_STRUCTURED_TRAINING_LOG", "1") != "0"
ENABLE_STAGE_D_EXPLAIN = os.environ.get("RENT_ENABLE_STAGE_D_EXPLAIN", "1") != "0"

SEMANTIC_TOP_K = int(os.environ.get("RENT_SEMANTIC_TOPK", "4"))
SEMANTIC_FIELD_WEIGHTS = {
    "schools": 1.00,
    "stations": 1.00,
    "features": 0.80,
    "description": 0.60,
}
INTENT_HIT_THRESHOLD = 0.45
INTENT_EVIDENCE_TOP_N = 2

# Stage C P0: deposit/freshness soft-scoring controls.
W_DEPOSIT = float(os.environ.get("RENT_W_DEPOSIT", "0.05"))
W_FRESHNESS = float(os.environ.get("RENT_W_FRESHNESS", "0.06"))
DEPOSIT_MISSING_POLICY = os.environ.get("RENT_DEPOSIT_MISSING_POLICY", "light_penalty").strip().lower()
if DEPOSIT_MISSING_POLICY not in {"neutral", "light_penalty"}:
    DEPOSIT_MISSING_POLICY = "light_penalty"
FRESHNESS_MISSING_POLICY = os.environ.get("RENT_FRESHNESS_MISSING_POLICY", "light_penalty").strip().lower()
if FRESHNESS_MISSING_POLICY not in {"neutral", "light_penalty"}:
    FRESHNESS_MISSING_POLICY = "light_penalty"
DEPOSIT_SCORE_CAP = float(os.environ.get("RENT_DEPOSIT_SCORE_CAP", "3000"))
DEPOSIT_SCORE_TAU = float(os.environ.get("RENT_DEPOSIT_SCORE_TAU", "3000"))
FRESHNESS_HALF_LIFE_DAYS = float(os.environ.get("RENT_FRESHNESS_HALF_LIFE_DAYS", "14"))

# Stage C P1 prep: sidecar vectors for preference rerank.
PREF_VECTOR_ENABLED = os.environ.get("RENT_PREF_VECTOR_ENABLED", "1") != "0"
PREF_VECTOR_PATH = os.environ.get(
    "RENT_PREF_VECTOR_PATH",
    os.path.join(ROOT_DIR, "artifacts", "qdrant_local", "collection", "rent_listings", "pref_vectors.parquet"),
)
PREF_VECTOR_FEATURE_WEIGHT = float(os.environ.get("RENT_PREF_VECTOR_FEATURE_WEIGHT", "0.80"))
PREF_VECTOR_DESCRIPTION_WEIGHT = float(os.environ.get("RENT_PREF_VECTOR_DESCRIPTION_WEIGHT", "0.60"))
