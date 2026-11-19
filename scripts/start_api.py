import os
import sys

# Ensure current working directory and sys.path are always the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import uvicorn
from api.app import app

def check_port(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) != 0

if __name__ == "__main__":
    target_port = 8000
    if not check_port(target_port):
        print(f"Warning: Port {target_port} is already in use. Trying port 8001...")
        target_port = 8001

    print("=" * 65)
    print("FASTAPI TELCO CHURN INFERENCE SERVER")
    print("=" * 65)
    print(f"Server URL:         http://127.0.0.1:{target_port}")
    print(f"Interactive Docs:   http://127.0.0.1:{target_port}/docs")
    print(f"Web Dashboard:      http://127.0.0.1:{target_port}/dashboard")
    print("=" * 65)
    print("Server is active and listening. Press CTRL+C to stop.")
    print("=" * 65)

    uvicorn.run(app, host="127.0.0.1", port=target_port, log_level="info")
