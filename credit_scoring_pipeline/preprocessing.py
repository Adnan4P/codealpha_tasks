import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe, handles missing values, and maps target to binary.
    """
    df_clean = df.copy()
    
    # 1. Handle missing values in Saving accounts and Checking account
    # We fill with 'Unknown' representing that the user doesn't have an account or it was not reported
    for col in ['Saving accounts', 'Checking account']:
        if col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna('Unknown')
            logger.info(f"Filled {missing_count} missing values in '{col}' with 'Unknown'")
            
    # 2. Encode the target column 'Risk' into binary format (1 for Bad, 0 for Good)
    if 'Risk' in df_clean.columns:
        # Check values
        unique_risks = df_clean['Risk'].unique()
        logger.info(f"Unique values in Risk: {unique_risks}")
        df_clean['Risk'] = df_clean['Risk'].map({'good': 0, 'bad': 1})
        logger.info("Mapped 'Risk': 'good' -> 0, 'bad' -> 1")
    else:
        logger.warning("Target column 'Risk' not found in dataframe.")
        
    return df_clean

def split_data(df: pd.DataFrame, target_col='Risk', test_size=0.25, random_state=42):
    """
    Splits the data into features (X) and target (y), followed by train/test division.
    Uses stratification to handle class imbalance.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Split data into train (size={len(X_train)}) and test (size={len(X_test)}) sets (stratified).")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from src.data_loader import load_dataset
    df = load_dataset()
    df_clean = preprocess_raw_data(df)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    print("\nTrain target class distribution:")
    print(y_train.value_counts(normalize=True))
    print("\nTest target class distribution:")
    print(y_test.value_counts(normalize=True))
