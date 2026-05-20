# Repository Guidelines

## Project Structure & Module Organization
This repository implements Qwen3-VL multimodal embedding and reranking utilities. Core model wrappers live in `src/models/`, with evaluation code under `src/evaluation/mmeb_v2/`. Runtime API services are in `servers/`: `unified_server.py` for the route-based API, `start_combined.py` for launching it, plus standalone embedding and reranker modules. Example notebooks and generated sample outputs are in `examples/`; small demo inputs are in `data/examples/`; benchmark configs and launch scripts are in `scripts/evaluation/mmeb_v2/`. The technical report is stored in `assets/`.

## Build, Test, and Development Commands
- `bash scripts/setup_environment.sh`: installs `uv`, syncs dependencies, installs `flash-attn`, and verifies the Python environment.
- `source .venv/bin/activate`: activates the local environment created by setup.
- `uv sync`: refreshes dependencies from `pyproject.toml` and `uv.lock`.
- `.venv/bin/python servers/start_combined.py`: starts the unified embedding and reranker API on port `API_PORT`.
- `bash scripts/evaluation/mmeb_v2/eval_embedding.sh` or `bash scripts/evaluation/mmeb_v2/eval_reranker.sh`: run MMEB v2 evaluation jobs after datasets and model paths are configured.

## Coding Style & Naming Conventions
Use Python 3.11+ and follow the existing PEP 8 style: 4-space indentation, snake_case functions and variables, PascalCase classes, and uppercase constants such as `MAX_LENGTH`. Keep API request/response schemas as Pydantic models near their FastAPI endpoints. Prefer explicit environment-variable names, for example `EMBEDDING_MODEL_PATH` and `USE_EMBEDDING_GPU`, and document new service settings in `embedding_reranker.env` or `servers/README.md`.

## Testing Guidelines
No dedicated unit-test suite is currently present. For Python changes, at minimum run `python -m py_compile` on edited modules. For service changes, start the relevant server and check `/health` plus one representative embedding or rerank request. For evaluation changes, run the smallest applicable MMEB config before launching full benchmark jobs.

## Commit & Pull Request Guidelines
Recent history uses short, imperative summaries in Chinese or English, sometimes with a prefix such as `Fix:`. Keep commits focused, for example `Fix: preserve rerank index dtype` or `优化GPU配置读取`. Pull requests should include a concise description, affected entry points, model or dataset assumptions, verification commands, and API examples or screenshots when endpoints or notebooks change. Link related issues when available and avoid committing downloaded model weights or large generated artifacts.

## Security & Configuration Tips
Do not hard-code private model paths, API keys, or host-specific GPU assignments. Use `embedding_reranker.env` and environment variables for deployment configuration. Keep large model downloads under ignored local directories such as `models/`.
