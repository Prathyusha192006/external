import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataLoader")

def load_data(file_path: str = None, version: str = "v1") -> pd.DataFrame:
    """
    Loads Telco Customer Churn dataset by specified version or direct file path.
    
    Args:
        file_path (str, optional): Custom path to CSV file.
        version (str): Dataset version identifier ('v1', 'v2', etc.).
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if file_path is None:
        if version == "v1":
            file_path = "data/versions/dataset_v1.csv"
        elif version == "v2":
            file_path = "data/versions/dataset_v2.csv"
        else:
            file_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
            
    if not os.path.exists(file_path):
        # Fallback to raw if version file does not exist yet
        raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if os.path.exists(raw_path):
            logger.warning(f"File {file_path} not found. Falling back to {raw_path}")
            file_path = raw_path
        else:
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    logger.info(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Dataset successfully loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

if __name__ == "__main__":
    df = load_data(version="v1")
    print(df.head())
