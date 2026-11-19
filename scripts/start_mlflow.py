import subprocess
import sys

if __name__ == "__main__":
    print("Launching MLflow Tracking UI on http://127.0.0.1:5000 ...")
    subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--port", "5000", "--host", "127.0.0.1"])
