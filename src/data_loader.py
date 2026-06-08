import os
import urllib.request
import pandas as pd

def fetch_data(url: str, output_path: str):
    """Fetches dataset from the given URL and saves it to output_path."""
    if not os.path.exists(output_path):
        print(f"Downloading dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            print("\nPlease download the dataset manually from Kaggle:")
            print("https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset")
            print(f"And place the CSV file at: {os.path.abspath(output_path)}\n")
            exit(1)
    else:
        print(f"Dataset already exists at {output_path}.")

def load_raw_data(file_path: str = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv") -> pd.DataFrame:
    """Loads the raw dataset into a pandas DataFrame."""
    return pd.read_csv(file_path)

if __name__ == "__main__":
    # URL for the IBM HR Analytics Employee Attrition dataset (Public mirror)
    DATA_URL = "https://raw.githubusercontent.com/pavopax/ibm-hr-analytics-attrition-dataset/master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    OUTPUT_FILE = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    
    # Ensure raw directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    fetch_data(DATA_URL, OUTPUT_FILE)
    
    # Verify data loading
    df = load_raw_data(OUTPUT_FILE)
    print(f"Dataset loaded successfully with shape: {df.shape}")
