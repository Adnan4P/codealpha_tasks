import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies custom domain engineering calculations to the dataframe.
    These are row-wise transformations and do not introduce leakage.
    """
    df_engineered = df.copy()
    
    # 1. Installment Burden (Monthly payment proxy)
    # Ratio of credit amount to duration in months
    df_engineered['Installment_Burden'] = df_engineered['Credit amount'] / df_engineered['Duration']
    
    # 2. Log Credit Amount (handles highly skewed values)
    df_engineered['Log_Credit_Amount'] = np.log1p(df_engineered['Credit amount'])
    
    # 3. Ordinal mapping of Saving accounts
    saving_map = {'Unknown': 0, 'little': 1, 'moderate': 2, 'quite rich': 3, 'rich': 4}
    df_engineered['Saving_Ordinal'] = df_engineered['Saving accounts'].map(saving_map).fillna(0)
    
    # 4. Ordinal mapping of Checking account
    checking_map = {'Unknown': 0, 'little': 1, 'moderate': 2, 'rich': 3}
    df_engineered['Checking_Ordinal'] = df_engineered['Checking account'].map(checking_map).fillna(0)
    
    # 5. Age Binning (Young, Adult, Senior)
    bins = [0, 25, 55, 120]
    labels = ['Young', 'Adult', 'Senior']
    df_engineered['Age_Group'] = pd.cut(df_engineered['Age'], bins=bins, labels=labels)
    
    logger.info("Successfully added engineered features: Installment_Burden, Log_Credit_Amount, Saving_Ordinal, Checking_Ordinal, Age_Group.")
    return df_engineered

def build_preprocessor(categorical_cols, numerical_cols):
    """
    Builds a scikit-learn ColumnTransformer that scales numerical variables
    and one-hot encodes categorical variables.
    """
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='drop')
    
    return preprocessor

def get_feature_lists():
    """
    Defines which columns are categorical and which are numerical
    after adding custom features.
    """
    # Categorical columns to be one-hot encoded
    categorical_cols = ['Sex', 'Housing', 'Purpose', 'Age_Group']
    
    # Numerical/ordinal columns to be scaled
    numerical_cols = [
        'Age', 'Job', 'Credit amount', 'Duration',
        'Installment_Burden', 'Log_Credit_Amount', 
        'Saving_Ordinal', 'Checking_Ordinal'
    ]
    
    return categorical_cols, numerical_cols

if __name__ == "__main__":
    from src.data_loader import load_dataset
    from src.preprocessing import preprocess_raw_data, split_data
    
    df = load_dataset()
    df_clean = preprocess_raw_data(df)
    df_engineered = add_custom_features(df_clean)
    
    X_train, X_test, y_train, y_test = split_data(df_engineered)
    
    cat_cols, num_cols = get_feature_lists()
    preprocessor = build_preprocessor(cat_cols, num_cols)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print("Preprocessed train shape:", X_train_processed.shape)
    print("Preprocessed test shape:", X_test_processed.shape)
    # Get column names after transformation
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols)
    all_feature_names = num_cols + list(encoded_cat_names)
    print("\nEngineered Feature Names:")
    print(all_feature_names)
