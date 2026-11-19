import subprocess
import sys

if __name__ == "__main__":
    print("Launching Streamlit Telco Churn MLOps Dashboard on http://localhost:8501 ...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.headless=true"])
