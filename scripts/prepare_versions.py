import os
import pandas as pd

def prepare_versions():
    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    os.makedirs("data/versions", exist_ok=True)
    
    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset with {len(df)} rows and {df.shape[1]} columns.")
    
    # Version 1: Initial cohort of 6,000 records
    df_v1 = df.iloc[:6000].copy()
    v1_path = "data/versions/dataset_v1.csv"
    df_v1.to_csv(v1_path, index=False)
    print(f"Created Version 1 (v1): {v1_path} ({len(df_v1)} rows)")
    
    # Version 2: Complete cohort of 7,043 records (latest customer base)
    df_v2 = df.copy()
    v2_path = "data/versions/dataset_v2.csv"
    df_v2.to_csv(v2_path, index=False)
    print(f"Created Version 2 (v2): {v2_path} ({len(df_v2)} rows)")

if __name__ == "__main__":
    prepare_versions()
