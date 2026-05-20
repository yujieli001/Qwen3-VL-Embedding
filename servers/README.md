# Qwen3-VL API Services

## Services Overview

Both models are served by one FastAPI app on port configured by `API_PORT`. Routes distinguish the model service.

| Service | Route Prefix | Description |
|---------|--------------|-------------|
| Embedding | `/embedding` | Embedding API |
| Reranker | `/reranker` | Reranker API |

Compatibility aliases are also available on the same port: `/embeddings`, `/encode`, and `/rerank`.

## Quick Start

```bash
cd /data/Qwen3-VL-Embedding
.venv/bin/python servers/start_combined.py
```

This starts one service:
- Unified API: http://0.0.0.0:${API_PORT}
- Embedding routes: `/embedding/health`, `/embedding/embeddings`, `/embedding/encode`
- Reranker routes: `/reranker/health`, `/reranker/rerank`

## Configuration

Model and device settings are read from `embedding_reranker.env`.

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_PORT` | set in `embedding_reranker.env` | Unified API port |
| `EMBEDDING_MODEL_PATH` | `/model/Qwen3-VL-Embedding-2B` | Embedding model path |
| `RERANKER_MODEL_PATH` | `/model/Qwen3-VL-Reranker-2B` | Reranker model path |
| `USE_EMBEDDING_GPU` / `USE_RERANKER_GPU` | `true` | Enable GPU per model |
| `UVICORN_WORKERS` | `1` | Must stay `1`; each worker loads both models |
| `EMBEDDING_CUDA_DEVICE` / `RERANKER_CUDA_DEVICE` | `0` / `1` | Physical GPU IDs |

## Systemd Service Setup

```bash
sudo cp /data/Qwen3-VL-Embedding/embedding_reranker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart embedding_reranker.service
sudo systemctl status embedding_reranker.service
```

View logs:

```bash
sudo journalctl -u embedding_reranker.service -f
```

## API Endpoints

### Health Checks

Load the port before running shell examples:

```bash
source embedding_reranker.env
```

```bash
curl http://localhost:${API_PORT}/health
curl http://localhost:${API_PORT}/embedding/health
curl http://localhost:${API_PORT}/reranker/health
```

### Embedding

```bash
curl -X POST "http://localhost:${API_PORT}/embedding/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"text": "A woman playing with her dog on a beach at sunset."},
      {"text": "A woman shares a joyful moment with her golden retriever."},
      {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
    ],
    "normalize": true
  }'
```

Alternative encode route:

```bash
curl -X POST "http://localhost:${API_PORT}/embedding/encode?normalize=true" \
  -H "Content-Type: application/json" \
  -d '[
    {"text": "Query text"},
    {"text": "Document text"}
  ]'
```

### Reranker

```bash
curl -X POST "http://localhost:${API_PORT}/reranker/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Retrieve images or text relevant to the user's query.",
    "query": {"text": "A woman playing with her dog on a beach at sunset."},
    "documents": [
      {"text": "A woman shares a joyful moment with her golden retriever."},
      {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
    ],
    "batch_size": 1
  }'
```

## Dify Knowledge Base

Use the same base URL for both embedding and reranker providers:

```text
http://<host>:${API_PORT}
```

Dify should call these paths on that base URL:

```text
POST /v1/embeddings
POST /v1/rerank
GET  /v1/models
```

The `/v1/embeddings` response follows OpenAI embedding format and accepts `input` or multimodal `messages`. The `/v1/rerank` response follows a Cohere-style rerank shape with `results`, `scores`, and `relevance_score`; it accepts `top_n`, `return_documents`, and `instruction`, and document objects are sanitized to model fields such as `text`, `image`, and `video`.

## Stop or Disable Service

```bash
sudo systemctl stop embedding_reranker.service
sudo systemctl disable embedding_reranker.service
```
