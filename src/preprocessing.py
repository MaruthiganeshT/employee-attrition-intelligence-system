import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(file_path: str = "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv") -> pd.DataFrame:
    """Load dataset from path."""
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, duplicates, and drop zero-variance columns."""
    df_cleaned = df.copy()
    
    # 1. Handle Duplicates
    if df_cleaned.duplicated().sum() > 0:
        print(f"Removing {df_cleaned.duplicated().sum()} duplicates...")
        df_cleaned.drop_duplicates(inplace=True)
    
    # 2. Drop zero-variance and identifier columns
    cols_to_drop = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    cols_to_drop = [c for c in cols_to_drop if c in df_cleaned.columns]
    df_cleaned.drop(columns=cols_to_drop, inplace=True)
    
    # 3. Handle Missing Values (IBM dataset usually has none, but for robustness)
    df_cleaned.fillna(df_cleaned.median(numeric_only=True), inplace=True)
    for col in df_cleaned.select_dtypes(include=['object']).columns:
        df_cleaned[col] = df_cleaned[col].fillna(
            df_cleaned[col].mode()[0]
            )

    return df_cleaned

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new meaningful features and encode target."""
    df_engineered = df.copy()
    
    # Encode target variable
    if 'Attrition' in df_engineered.columns:
        df_engineered['Attrition'] = df_engineered['Attrition'].map({'Yes': 1, 'No': 0})
        
    # Example Feature Engineering: Total satisfaction score
    satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction']
    if all(col in df_engineered.columns for col in satisfaction_cols):
        df_engineered['TotalSatisfaction'] = df_engineered[satisfaction_cols].sum(axis=1)
        
    # Career stability
    if {'YearsAtCompany', 'TotalWorkingYears'}.issubset(df_engineered.columns):
        df_engineered['CareerStability'] = (
            df_engineered['YearsAtCompany'] /
            (df_engineered['TotalWorkingYears'] + 1)
            )

    if {'MonthlyIncome', 'YearsAtCompany'}.issubset(df_engineered.columns):
        df_engineered['IncomePerYearAtCompany'] = (
            df_engineered['MonthlyIncome'] /
            (df_engineered['YearsAtCompany'] + 1)
            )
    if {'OverTime', 'JobSatisfaction'}.issubset(df_engineered.columns):
        overtime = df_engineered['OverTime'].map({
            'Yes': 1,
            'No': 0
            })
        df_engineered['OvertimeSatisfactionRisk'] = (
        overtime * (5 - df_engineered['JobSatisfaction'])
        )
    return df_engineered
    

def preprocess_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split data and apply transformations (scaling, encoding)."""
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Identify column types
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # We will use StandardScaler to mitigate the effect of outliers on Logistic Regression
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])
    
    # Fit on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names for SHAP explainability later
    num_feature_names = num_cols
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols).tolist()
    feature_names = num_feature_names + cat_feature_names
    
    # Convert back to DataFrame for easier handling with SHAP
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names

if __name__ == "__main__":
    
    df_raw = load_data()
    print(f"Raw data shape: {df_raw.shape}")
    
    df_clean = clean_data(df_raw)
    df_eng = engineer_features(df_clean)
    
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_and_split(df_eng)
    print(f"Processed training features shape: {X_train.shape}")
    print(f"Processed test features shape: {X_test.shape}")
    
    # Save the processed data and preprocessor
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of features after encoding: {len(feature_names)}")
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(
        feature_names,
        "models/feature_names.joblib"
        )
    print("Preprocessing complete. Data and preprocessor saved.")
