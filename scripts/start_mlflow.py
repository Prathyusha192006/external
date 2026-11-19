import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    print("=" * 65)
    print("MLFLOW TRACKING & MODEL REGISTRY UI")
    print("=" * 65)
    print("MLflow URL: http://127.0.0.1:5000")
    print("Press CTRL+C in this terminal to stop.")
    print("=" * 65)
    subprocess.run([
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--port", "5000",
        "--host", "127.0.0.1"
    ])
