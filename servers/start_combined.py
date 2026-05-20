#!/usr/bin/env python3
"""
Start the unified Embedding and Reranker API service on one port.
Port: configured by API_PORT
"""

import os
import sys

import uvicorn


def load_env_file(env_path):
    """Load environment variables from .env file."""
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    env_file = os.path.join(project_root, "embedding_reranker.env")
    load_env_file(env_file)
    print(f"Loaded environment from: {env_file}")
    print(f"Working directory: {project_root}")

    api_port = os.environ.get("API_PORT")
    if not api_port:
        raise SystemExit("API_PORT is required in embedding_reranker.env")
    try:
        port = int(api_port)
    except ValueError as exc:
        raise SystemExit(f"API_PORT must be an integer, got {api_port!r}") from exc

    workers_value = os.environ.get("UVICORN_WORKERS", "1")
    try:
        workers = int(workers_value)
    except ValueError as exc:
        raise SystemExit(f"UVICORN_WORKERS must be an integer, got {workers_value!r}") from exc
    if workers != 1:
        raise SystemExit("UVICORN_WORKERS must be 1 because each worker loads both GPU models")

    use_embedding_gpu = os.environ.get("USE_EMBEDDING_GPU", "true").lower() in ("true", "1", "yes")
    use_reranker_gpu = os.environ.get("USE_RERANKER_GPU", "true").lower() in ("true", "1", "yes")
    embedding_cuda_device = os.environ.get("EMBEDDING_CUDA_DEVICE", "0")
    reranker_cuda_device = os.environ.get("RERANKER_CUDA_DEVICE", "0")
    model_dtype = "bfloat16" if (use_embedding_gpu or use_reranker_gpu) else "float32"
    os.environ.setdefault("MODEL_DTYPE", model_dtype)
    os.environ.setdefault("PYTHONPATH", project_root)

    print(
        "Device configuration: "
        f"embedding={'GPU ' + embedding_cuda_device if use_embedding_gpu else 'CPU'}, "
        f"reranker={'GPU ' + reranker_cuda_device if use_reranker_gpu else 'CPU'}, "
        f"MODEL_DTYPE={os.environ['MODEL_DTYPE']}"
    )
    print(f"Starting unified service: http://0.0.0.0:{port}")
    print("  - Embedding routes: /embedding/health, /embedding/embeddings, /embedding/encode")
    print("  - Reranker routes: /reranker/health, /reranker/rerank")

    uvicorn.run(
        "servers.unified_server:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    main()
