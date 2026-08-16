
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .preprocessing import build_preprocessor

def logistic_pipeline(X, random_state=42, class_weight="balanced"):
    return Pipeline([
        ("preprocess", build_preprocessor(X, scale_numeric=True)),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state,
        )),
    ])

def random_forest_pipeline(
    X,
    random_state=42,
    class_weight="balanced_subsample",
):
    return Pipeline([
        ("preprocess", build_preprocessor(X, scale_numeric=False)),
        ("model", RandomForestClassifier(
            n_estimators=250,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

def xgboost_pipeline(X, random_state=42, scale_pos_weight=1.0):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("Install xgboost with: pip install xgboost") from exc

    return Pipeline([
        ("preprocess", build_preprocessor(X, scale_numeric=False)),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])
