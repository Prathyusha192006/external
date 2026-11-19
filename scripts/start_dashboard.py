import os
import sys
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if __name__ == "__main__":
    print("=" * 65)
    print("STREAMLIT TELCO CHURN MLOPS DASHBOARD")
    print("=" * 65)
    print("Dashboard URL: http://localhost:8501")
    print("Press CTRL+C in this terminal to stop.")
    print("=" * 65)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(PROJECT_ROOT, "dashboard", "app.py"),
        "--server.port=8501",
        "--server.headless=true"
    ])
