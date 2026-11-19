import os
import shutil
import subprocess
import pandas as pd

def run_cmd(cmd_list):
    print(f"Executing: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.returncode != 0:
        print(f"Error ({res.returncode}): {res.stderr.strip()}")
        raise RuntimeError(f"Command failed: {' '.join(cmd_list)}")

def setup_dvc():
    print("=" * 70)
    print("CONFIGURING GIT & DVC ARTIFACT AND DATASET VERSIONING (v1 & v2)")
    print("=" * 70)

    os.makedirs("data", exist_ok=True)
    os.makedirs("dvc_remote_storage", exist_ok=True)

    # Make initial git commit for repo structure
    run_cmd(["git", "add", "src", "api", "dashboard", "main.py", "requirements.txt", ".gitignore", ".dvc/config"])
    run_cmd(["git", "commit", "-m", "Initial commit: Modular MLOps pipeline structure"])

    # --- VERSION 1 SETUP ---
    print("\n>>> Setting up Dataset Version 1 (v1: 6,000 records)...")
    v1_source = "data/versions/dataset_v1.csv"
    active_dataset = "data/active_dataset.csv"
    shutil.copy(v1_source, active_dataset)
    
    df1 = pd.read_csv(active_dataset)
    print(f"Version 1 dataset loaded: {len(df1)} rows.")

    run_cmd(["python", "-m", "dvc", "add", active_dataset])
    run_cmd(["python", "-m", "dvc", "push"])
    run_cmd(["git", "add", "data/active_dataset.csv.dvc", "data/.gitignore"])
    run_cmd(["git", "commit", "-m", "Track dataset v1.0 (6,000 rows) with DVC"])
    run_cmd(["git", "tag", "-a", "v1.0", "-m", "Dataset release v1.0"])
    print("--> Git tag 'v1.0' created and pushed to DVC remote!")

    # --- VERSION 2 SETUP ---
    print("\n>>> Setting up Dataset Version 2 (v2: 7,043 records)...")
    v2_source = "data/versions/dataset_v2.csv"
    shutil.copy(v2_source, active_dataset)
    
    df2 = pd.read_csv(active_dataset)
    print(f"Version 2 dataset loaded: {len(df2)} rows.")

    run_cmd(["python", "-m", "dvc", "add", active_dataset])
    run_cmd(["python", "-m", "dvc", "push"])
    run_cmd(["git", "add", "data/active_dataset.csv.dvc"])
    run_cmd(["git", "commit", "-m", "Track dataset v2.0 (7,043 rows) with DVC"])
    run_cmd(["git", "tag", "-a", "v2.0", "-m", "Dataset release v2.0"])
    print("--> Git tag 'v2.0' created and pushed to DVC remote!")

    print("\n" + "=" * 70)
    print("DVC & GIT VERSIONING SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    setup_dvc()
