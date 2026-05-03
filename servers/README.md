# Qwen3-VL API Services

## Services Overview

| Service | Port | Description |
|---------|------|-------------|
| Combined API | 10011/10012 | Raw Embedding (10011) + Reranker (10012) |
| OpenAI Proxy | 10013 | OpenAI-compatible API (for Dify) |

## Quick Start

### Start All Services
```bash
# Start combined service (ports 10011, 10012)
.venv/bin/python servers/start_combined.py

# OR start OpenAI-compatible proxy (port 10013) - Recommended for Dify
.venv/bin/python servers/openai_proxy.py
```

A unified API server that provides both Embedding and Reranker functionality.

## Services

| Service | Port | Model Path |
|---------|------|------------|
| Embedding | 10011 | /model/Qwen3-VL-Embedding-2B |
| Reranker | 10012 | /model/Qwen3-VL-Reranker-2B |

## Quick Start

### Manual Start
```bash
cd /data/Qwen3-VL-Embedding
.venv/bin/python servers/start_combined.py
```

This starts both services:
- Embedding: http://0.0.0.0:10011
- Reranker: http://0.0.0.0:10012

## Systemd Service Setup

### 1. Copy service file to systemd directory
```bash
sudo cp /data/Qwen3-VL-Embedding/embedding_reranker.service /etc/systemd/system/
```

### 2. Reload systemd daemon
```bash
sudo systemctl daemon-reload
```

### 3. Enable and start service
```bash
# Start the combined service (both ports)
sudo systemctl start embedding_reranker.service
sudo systemctl enable embedding_reranker.service
```

### 4. Check service status
```bash
sudo systemctl status embedding_reranker.service
```

### 5. View logs
```bash
sudo journalctl -u embedding_reranker.service -f
```

## API Endpoints

### Health & Info

**Health Check (both ports)**
```bash
curl http://localhost:10011/health
curl http://localhost:10012/health
```

**Service Info**
```bash
curl http://localhost:10011/info
```

### Embedding Endpoints (Port 10011)

**Generate Embeddings**
```bash
curl -X POST "http://localhost:10011/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {"text": "A woman playing with her dog on a beach at sunset."},
      {"text": "A woman shares a joyful moment with her golden retriever."},
      {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
      {"text": "Beach sunset with dog", "image": "https://example.com/image.jpg"}
    ],
    "normalize": true
  }'
```

**Alternative Embed Endpoint**
```bash
curl -X POST "http://localhost:10011/embed?normalize=true" \
  -H "Content-Type: application/json" \
  -d '[
    {"text": "Query text"},
    {"text": "Document text"}
  ]'
```

### Reranker Endpoints (Port 10012)

**Rerank Documents (Text + Image)**
```bash
curl -X POST "http://localhost:10012/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Retrieve images or text relevant to the user'"'"'s query.",
    "query": {"text": "A woman playing with her dog on a beach at sunset."},
    "documents": [
      {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
      {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
      {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", 
       "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
    ],
    "fps": 1.0, 
    "max_frames": 64
  }'
```

### Combined Endpoints (Both ports)

**Process Embedding and Rerank**
```bash
curl -X POST "http://localhost:10011/process?use_reranker=true" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {"text": "Beach sunset"},
    "documents": [
      {"text": "A beautiful sunset on the beach"},
      {"text": "Rainy day in the city"}
    ]
  }'
```

## Stop Service
```bash
sudo systemctl stop embedding_reranker.service
```

## Disable Service (do not start on boot)
```bash
sudo systemctl disable embedding_reranker.service
```

## Input Format Examples

### Text Only
```json
{"text": "This is a text input"}
```

### Image Only
```json
{"image": "https://example.com/image.jpg"}
```

### Text + Image
```json
{"text": "Describe this image", "image": "https://example.com/image.jpg"}
```

### With Instruction (Embedding)
```json
{
  "text": "Product review analysis",
  "instruction": "Encode reviews for sentiment analysis"
}
```

## Python Client Example

```python
import requests

# Embedding
response = requests.post("http://localhost:10011/embeddings", json={
    "inputs": [
        {"text": "Query"},
        {"text": "Document"}
    ],
    "normalize": True
})
embeddings = response.json()["embeddings"]

# Reranker (Text + Image)
response = requests.post("http://localhost:10012/rerank", json={
    "instruction": "Retrieve images or text relevant to the user's query.",
    "query": {"text": "A woman playing with her dog on a beach at sunset."},
    "documents": [
        {"text": "A woman shares a joyful moment with her golden retriever..."},
        {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
        {"text": "...", "image": "..."}
    ],
    "fps": 1.0,
    "max_frames": 64
})
scores = response.json()["scores"]
```
