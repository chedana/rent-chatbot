# RunPod New Session Setup

This is the bootstrap checklist for a fresh RunPod session.

## 1) SSH + GitHub access

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub   # paste to GitHub SSH keys
ssh -T git@github.com
```

## 2) Clone repo

```bash
git clone git@github.com:chedana/rent-chatbot.git
cd /workspace/rent-chatbot
```

## 3) Create and activate Python venv

```bash
python3 -m venv chat_bot
source /workspace/chat_bot/bin/activate
```

## 4) Bootstrap base environment

```bash
bash /workspace/bootstrap.sh
```

## 5) Install Python dependencies

```bash
pip install -U pip setuptools wheel
pip install faiss-cpu playwright
playwright install chromium

pip install beautifulsoup4
pip install lxml

pip install sentence_transformers
pip install -U pyarrow
pip install -U qdrant-client
pip install vllm huggingface_hub
pip install pandas
```

## 6) Install system package(s)

```bash
apt update
apt install -y vim
```

## 7) Download model

```bash
huggingface-cli download Qwen/Qwen3-14B \
  --local-dir Qwen3-14B
```

## 8) Start vLLM server

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model ./Qwen3-14B \
  --dtype float16 \
  --max-model-len 8192 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

## 9) Data Artifacts (What to rebuild, when)

### A. Qdrant storage: `storage.sqlite`

- Path:
  - `/workspace/rent-chatbot/artifacts/qdrant_local/collection/rent_listings/storage.sqlite`
- What it is:
  - Local Qdrant collection payload/vector storage file.
  - Contains the indexed listing payload used by Stage A retrieval (including location prefilter fields such as `location_postcode_tokens`, `location_station_tokens`, `location_region_tokens`).
- When to rebuild:
  - Any change to index payload schema or tokenization logic.
  - Any change to source dataset used for retrieval.
- Rebuild command:

```bash
cd /workspace/rent-chatbot
export RENT_QDRANT_PATH=/workspace/rent-chatbot/artifacts/qdrant_local
export RENT_QDRANT_COLLECTION=rent_listings
export RENT_QDRANT_SOURCE_PATH=/workspace/rent-chatbot/data/web_data/properties_clean.jsonl
export RENT_QDRANT_RESET=1
python3 /workspace/rent-chatbot/data/qdrant/build_qdrant_from_source.py
```

### B. Stage C sidecar: `pref_vectors.parquet`

- Path (default):
  - `/workspace/rent-chatbot/artifacts/features/pref_vectors.parquet`
- What it is:
  - Stage C preference sidecar vectors (features/description segment embeddings).
  - Used for soft preference matching in rerank stage.
- Runtime binding:
  - `RENT_PREF_VECTOR_PATH` must point to this file.
- When to rebuild:
  - Source listing text changed.
  - Embedding model changed.
  - Segment split strategy changed.
- Build command:

```bash
cd /workspace/rent-chatbot
export RENT_PREF_VECTOR_SOURCE_PATH=/workspace/rent-chatbot/data/web_data/properties_clean.jsonl
export RENT_PREF_VECTOR_PATH=/workspace/rent-chatbot/artifacts/features/pref_vectors.parquet
python3 /workspace/rent-chatbot/data/qdrant/build_preference_sidecar_vectors.py
```

### C. Dependency note

- `storage.sqlite` and `pref_vectors.parquet` are independent artifacts:
  - Stage A uses Qdrant (`storage.sqlite`).
  - Stage C preference scoring uses sidecar vectors (`pref_vectors.parquet`).
- If you changed both retrieval payload logic and preference vector logic, rebuild both.
