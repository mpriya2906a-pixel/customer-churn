import os
import joblib
import pandas as pd
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


NUMERIC_FEATURES = [
    'tenure',
    'monthly_charges',
    'total_charges',
    'support_tickets',
    'last_login_days_ago',
    'usage_frequency'
]

CATEGORICAL_FEATURES = [
    'contract_type',
    'payment_method',
    'internet_service'
]

TARGET_COLUMN = 'churn_label'
ID_COLUMN = 'customer_id'


def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer preprocessing pipeline.
    
    - Handles missing values via median imputation for numeric and most_frequent for categorical.
    - Scales numeric features using StandardScaler.
    - Encodes categorical variables using OneHotEncoder with handle_unknown='ignore'.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    return preprocessor


def create_full_pipeline(classifier, preprocessor: ColumnTransformer = None) -> Pipeline:
    """
    Creates an end-to-end scikit-learn Pipeline linking preprocessing and the model.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()
        
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])


def get_transformed_feature_names(fitted_preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts human-readable feature names from the fitted ColumnTransformer.
    """
    try:
        return list(fitted_preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        num_names = NUMERIC_FEATURES
        cat_encoder = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        return num_names + cat_names


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates feature matrix X and target y from the raw DataFrame.
    """
    # Clean total_charges if needed
    df_clean = df.copy()
    if 'total_charges' in df_clean.columns:
        df_clean['total_charges'] = pd.to_numeric(df_clean['total_charges'], errors='coerce')
    
    feature_cols = [c for c in df_clean.columns if c not in [ID_COLUMN, TARGET_COLUMN]]
    X = df_clean[feature_cols]
    y = df_clean[TARGET_COLUMN].astype(int)
    return X, y


def save_pipeline(obj, filepath: str):
    """
    Saves an object (pipeline, model, or preprocessor) using joblib.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"[Pipeline] Saved artifact to {filepath}")


def load_pipeline(filepath: str):
    """
    Loads a joblib artifact from disk.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found at {filepath}")
    return joblib.load(filepath)
