
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

KNOWN_TARGET_COLUMNS = {"label", "attack_cat"}

@dataclass
class FeatureGroups:
    numeric: List[str]
    categorical: List[str]

def infer_feature_groups(X: pd.DataFrame) -> FeatureGroups:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return FeatureGroups(numeric=numeric, categorical=categorical)

def split_xy(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' does not exist.")

    drop_cols = [c for c in KNOWN_TARGET_COLUMNS if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    y = df[target].copy()
    return X, y

def align_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    common = [c for c in X_train.columns if c in X_test.columns]
    return X_train[common].copy(), X_test[common].copy()

def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True):
    groups = infer_feature_groups(X)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, groups.numeric),
        ("categorical", categorical_pipeline, groups.categorical),
    ])
