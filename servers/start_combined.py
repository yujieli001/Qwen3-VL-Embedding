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

    # 从环境变量读取配置
    use_gpu = os.environ.get("USE_GPU", "true").lower() in ("true", "1", "yes")
    gpu_config = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env = {**os.environ, "PYTHONPATH": project_root, "USE_GPU": str(use_gpu).lower(), "CUDA_VISIBLE_DEVICES": gpu_config if use_gpu else ""}

    device_type = "GPU" if use_gpu else "CPU"
    print(f"Device configuration: {device_type} (CUDA_VISIBLE_DEVICES={gpu_config if use_gpu else 'none'})")

    # Start independent services so each port loads only the model it serves.
    port_10011 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "servers.embedding_server:app", "--host", "0.0.0.0", "--port", "10011"],
        env=env
    )

    time.sleep(2)  # Give first server time to start

    port_10012 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "servers.reranker_server:app", "--host", "0.0.0.0", "--port", "10012"],
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
