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
DEFAULT_RECALL = int(os.environ.get("RENT_RECALL", "200"))

UNKNOWN_PENALTY_WEIGHTS = {
    "price": 0.35,
    "bedrooms": 0.30,
    "bathrooms": 0.20,
    "available_from": 0.15,
}
UNKNOWN_PENALTY_CAP = 0.60
FURNISH_ASK_AGENT_PENALTY = 0.06

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

SEMANTIC_TOP_K = int(os.environ.get("RENT_SEMANTIC_TOPK", "4"))
SEMANTIC_FIELD_WEIGHTS = {
    "schools": 1.00,
    "stations": 1.00,
    "features": 0.80,
    "description": 0.60,
}
INTENT_HIT_THRESHOLD = 0.45
INTENT_EVIDENCE_TOP_N = 2
