# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Qwen3-VL-Embedding 和 Qwen3-VL-Reranker 是基于 Qwen3-VL 的多模态嵌入和重排序模型，支持文本、图像、截图、视频和混合模态输入。

## Development Commands

```bash
# Setup environment (installs uv, dependencies, flash-attn)
bash scripts/setup_environment.sh

# Activate environment
source .venv/bin/activate

# Sync dependencies from pyproject.toml and uv.lock
uv sync

# Start unified embedding + reranker service (port configured by `API_PORT`)
.venv/bin/python servers/start_combined.py

# Evaluate on MMEB v2 benchmark
bash scripts/evaluation/mmeb_v2/eval_embedding.sh   # Embedding model
bash scripts/evaluation/mmeb_v2/eval_reranker.sh   # Reranker model
```

## Architecture

**Core Models** (`src/models/`):
- [qwen3_vl_embedding.py](src/models/qwen3_vl_embedding.py): `Qwen3VLEmbedder` 类实现双塔架构，使用 `[EOS]` token 的隐藏状态作为语义表示
- [qwen3_vl_reranker.py](src/models/qwen3_vl_reranker.py): `Qwen3VLReranker` 类实现单塔架构，使用 Cross-Attention 和 yes/no token 预测相关性分数

**API Services** (`servers/`):
- [embedding_server.py](servers/embedding_server.py): FastAPI embedding 服务模块（单独运行默认端口由 `API_PORT` 配置）
- [reranker_server.py](servers/reranker_server.py): FastAPI reranker 服务模块（单独运行默认端口由 `API_PORT` 配置）
- [unified_server.py](servers/unified_server.py): 单端口聚合服务，按路由区分 embedding 与 reranker
- [start_combined.py](servers/start_combined.py): 启动 `unified_server.py`，默认监听 `API_PORT`

**Evaluation** (`src/evaluation/mmeb_v2/`):
- [eval_embedding.py](src/evaluation/mmeb_v2/eval_embedding.py) / [eval_reranker.py](src/evaluation/mmeb_v2/eval_reranker.py): MMEB v2 基准评估
- [datasets/](src/evaluation/mmeb_v2/data/datasets/): 各类评估数据集实现 (图像分类、视频问答、检索等)
- [gather_results.py](src/evaluation/mmeb_v2/gather_results.py): 收集评估结果

**Input Format**:
- Embedding: `List[Dict]`，每个 Dict 包含 `text`/`image`/`video` 和可选的 `instruction`
- Reranker: `Dict` 包含 `query` 和 `documents` (均为多模态对象)，以及可选的 `instruction`

## GPU/CPU Configuration

通过环境变量控制 GPU/CPU 使用:
- `USE_EMBEDDING_GPU=true/false`: 控制 embedding 模型
- `USE_RERANKER_GPU=true/false`: 控制 reranker 模型

服务器代码在启动前设置 `CUDA_VISIBLE_DEVICES`，然后在模型加载时映射到 `cuda:0`。

## Key Constants

Embedding 模型:
- `MAX_LENGTH=8192`, `MIN_PIXELS=4096`, `MAX_PIXELS=1843200`, `MAX_TOTAL_PIXELS=7864320`

Reranker 模型:
- `MAX_LENGTH=10240`, `MIN_PIXELS=4096`, `MAX_PIXELS=1800*IMAGE_FACTOR*IMAGE_FACTOR`

## Testing

- Python 模块检查：`python -m py_compile <module>`
- API 健康检查：`curl http://localhost:${API_PORT}/embedding/health` 或 `curl http://localhost:${API_PORT}/reranker/health`
- 完整测试：启动服务并发送代表性请求

## Commit Style

使用简短的祈使句，可带中文或英文前缀如 `Fix:`、`优化:`、`Add:`。例如:
- `Fix: preserve rerank index dtype`
- `优化 GPU 配置读取`
- `Add: video frame sampling`
