#!/usr/bin/env python3
"""
Start Combined Embedding & Reranker Service on dual ports.
Runs two uvicorn servers on ports 10011 and 10012.
"""

import subprocess
import sys
import os
import time

# 加载.env 文件
def load_env_file(env_path):
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def main():
    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # 加载.env 配置文件
    env_file = os.path.join(project_root, "embedding_reranker.env")
    load_env_file(env_file)
    print(f"Loaded environment from: {env_file}")

    print(f"Working directory: {project_root}")

    # 从环境变量读取配置（各模型独立控制）
    use_embedding_gpu = os.environ.get("USE_EMBEDDING_GPU", "true").lower() in ("true", "1", "yes")
    use_reranker_gpu = os.environ.get("USE_RERANKER_GPU", "true").lower() in ("true", "1", "yes")
    embedding_cuda_device = os.environ.get("EMBEDDING_CUDA_DEVICE", "0")
    reranker_cuda_device = os.environ.get("RERANKER_CUDA_DEVICE", "0")

    # 设置 CUDA_VISIBLE_DEVICES
    if not use_embedding_gpu and not use_reranker_gpu:
        gpu_env = ""
        device_type = "CPU"
    elif use_embedding_gpu and use_reranker_gpu:
        gpu_env = f"{embedding_cuda_device},{reranker_cuda_device}"
        device_type = f"GPU (embedding:{embedding_cuda_device}, reranker:{reranker_cuda_device})"
    elif use_embedding_gpu:
        gpu_env = str(embedding_cuda_device)
        device_type = f"GPU (embedding:{embedding_cuda_device})"
    else:
        gpu_env = str(reranker_cuda_device)
        device_type = f"GPU (reranker:{reranker_cuda_device})"

    # 自动设置 MODEL_DTYPE：GPU 用 bfloat16，CPU 用 float32
    model_dtype = "bfloat16" if (use_embedding_gpu or use_reranker_gpu) else "float32"
    env = {**os.environ, "PYTHONPATH": project_root, "CUDA_VISIBLE_DEVICES": gpu_env, "MODEL_DTYPE": model_dtype}
    print(f"Device configuration: {device_type}, MODEL_DTYPE: {model_dtype}")

    # Start independent services so each port loads only the model it serves.
    # Read workers config from env (default to 1 to save GPU memory)
    workers = os.environ.get("UVICORN_WORKERS", "1")

    port_10011 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "servers.embedding_server:app", "--host", "0.0.0.0", "--port", "10011", "--workers", workers],
        env=env
    )

    time.sleep(2)  # Give first server time to start

    port_10012 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "servers.reranker_server:app", "--host", "0.0.0.0", "--port", "10012", "--workers", workers],
        env=env
    )

    print("Services started:")
    print("  - Embedding service: http://0.0.0.0:10011")
    print("  - Reranker service: http://0.0.0.0:10012")

    # Wait for both
    try:
        while True:
            code_10011 = port_10011.poll()
            code_10012 = port_10012.poll()
            if code_10011 is not None or code_10012 is not None:
                port_10011.terminate()
                port_10012.terminate()
                sys.exit(code_10011 if code_10011 is not None else code_10012)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        port_10011.terminate()
        port_10012.terminate()

if __name__ == "__main__":
    main()
