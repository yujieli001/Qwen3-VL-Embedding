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

    # Start both servers
    port_10011 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "servers.combined_app:app", "--host", "0.0.0.0", "--port", "10011"],
        env={**os.environ, "PYTHONPATH": project_root}
    )

    time.sleep(2)  # Give first server time to start

    port_10012 = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "servers.combined_app:app", "--host", "0.0.0.0", "--port", "10012"],
        env={**os.environ, "PYTHONPATH": project_root}
    )

    print("Services started:")
    print("  - Embedding service: http://0.0.0.0:10011")
    print("  - Reranker service: http://0.0.0.0:10012")

    # Wait for both
    try:
        port_10011.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
        port_10011.terminate()
        port_10012.terminate()

if __name__ == "__main__":
    main()
