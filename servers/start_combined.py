#!/usr/bin/env python3
"""
Start Combined Embedding & Reranker Service on dual ports.
Runs two uvicorn servers on ports 10011 and 10012.
"""

import subprocess
import sys
import os
import time

def main():
    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    print(f"Working directory: {project_root}")

    env = {**os.environ, "PYTHONPATH": project_root, "CUDA_VISIBLE_DEVICES": ""}

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
