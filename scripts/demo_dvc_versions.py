import os
import subprocess
import pandas as pd

def run_cmd(cmd_list):
    print(f"Executing: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.returncode != 0:
        print(f"Stderr: {res.stderr.strip()}")
        raise RuntimeError(f"Command failed: {' '.join(cmd_list)}")

def demonstrate_version_switching():
    print("=" * 75)
    print("DEMONSTRATING DVC & GIT DATASET VERSION SWITCHING & REPRODUCIBILITY")
    print("=" * 75)
    
    target_file = "data/active_dataset.csv"

    # Step 1: Switch to Version 1.0
    print("\n[STEP 1] Checking out Git tag 'v1.0' & running 'dvc checkout'...")
    run_cmd(["git", "checkout", "v1.0"])
    run_cmd(["python", "-m", "dvc", "checkout"])
    
    df_v1 = pd.read_csv(target_file)
    print(f"\n--> Successfully restored Version 1.0! Active dataset rows: {len(df_v1)}")
    assert len(df_v1) == 6000, f"Expected 6,000 rows for v1, got {len(df_v1)}"
    print(f"    Sample Customer IDs: {df_v1['customerID'].head(3).tolist()}")

    # Step 2: Switch to Version 2.0
    print("\n[STEP 2] Checking out Git tag 'v2.0' & running 'dvc checkout'...")
    run_cmd(["git", "checkout", "v2.0"])
    run_cmd(["python", "-m", "dvc", "checkout"])
    
    df_v2 = pd.read_csv(target_file)
    print(f"\n--> Successfully restored Version 2.0! Active dataset rows: {len(df_v2)}")
    assert len(df_v2) == 7043, f"Expected 7,043 rows for v2, got {len(df_v2)}"
    print(f"    Sample Customer IDs: {df_v2['customerID'].tail(3).tolist()}")

    # Switch back to master/main
    print("\n[STEP 3] Returning to master branch...")
    try:
        run_cmd(["git", "checkout", "master"])
    except:
        run_cmd(["git", "checkout", "main"])
    run_cmd(["python", "-m", "dvc", "checkout"])

    print("\n" + "=" * 75)
    print("VERIFICATION COMPLETE: Version switching and reproducibility 100% verified!")
    print("=" * 75)

if __name__ == "__main__":
    demonstrate_version_switching()
