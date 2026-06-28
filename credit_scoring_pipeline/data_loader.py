import os
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Primary and fallback dataset URLs from public repositories
DATA_URLS = [
    "https://raw.githubusercontent.com/praisan/hello-world/master/german_credit_data.csv"
]
LOCAL_DIR = "data"
LOCAL_FILE = "german_credit_data.csv"

def get_data_path():
    """Returns the local path of the dataset, creating the directory if needed."""
    os.makedirs(LOCAL_DIR, exist_ok=True)
    return os.path.join(LOCAL_DIR, LOCAL_FILE)

def load_dataset(urls=DATA_URLS, force_download=False):
    """
    Downloads the dataset from a public URL and caches it locally.
    If already downloaded, loads from the local cache.
    Tries fallback URLs if the primary URL fails.
    """
    local_path = get_data_path()
    
    if not force_download and os.path.exists(local_path):
        logger.info(f"Loading dataset from local cache: {local_path}")
        try:
            df = pd.read_csv(local_path)
            return df
        except Exception as e:
            logger.warning(f"Failed to read local cache: {e}. Re-downloading...")
            
    # Try URLs one by one
    last_error = None
    for url in urls:
        logger.info(f"Attempting to download dataset from: {url}")
        try:
            df = pd.read_csv(url)
            # Drop the unnamed index column if it is present
            if df.columns[0] == "Unnamed: 0":
                df = df.drop(df.columns[0], axis=1)
                
            df.to_csv(local_path, index=False)
            logger.info(f"Successfully cached dataset locally at: {local_path}")
            return df
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            last_error = e
            
    logger.error("All dataset URLs failed.")
    raise last_error

if __name__ == "__main__":
    # Test execution
    df = load_dataset()
    print("Dataset Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nSample Data:")
    print(df.head())
