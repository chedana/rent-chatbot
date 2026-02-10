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
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir Qwen2.5-7B-Instruct
```

## 8) Start vLLM server

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model ./Qwen2.5-7B-Instruct \
  --dtype float16 \
  --max-model-len 8192 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

