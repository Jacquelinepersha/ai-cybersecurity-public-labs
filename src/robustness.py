
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def feature_bounds(X_train, columns):
    """Return numeric min/max bounds learned from training data."""
    bounds = {}
    for column in columns:
        if column not in X_train.columns:
            continue
        values = pd.to_numeric(X_train[column], errors="coerce")
        bounds[column] = (float(values.min()), float(values.max()))
    return bounds

def stress_single_feature(X, feature, multiplier, bounds=None):
    """
    Return a copied DataFrame with one numeric feature multiplied by a factor.
    This is an offline sensitivity test, not an optimization routine.
    """
    if feature not in X.columns:
        raise KeyError(f"Missing feature: {feature}")

    out = X.copy()
    values = pd.to_numeric(out[feature], errors="coerce").astype(float)
    stressed = values * float(multiplier)

    if bounds and feature in bounds:
        low, high = bounds[feature]
        stressed = stressed.clip(lower=low, upper=high)

    out[feature] = stressed
    return out

def feature_stress_curve(model, X, y, feature, multipliers, bounds=None):
    """Measure classification performance across directional feature stress."""
    rows = []

    for multiplier in multipliers:
        stressed = stress_single_feature(
            X,
            feature=feature,
            multiplier=multiplier,
            bounds=bounds,
        )

        pred = model.predict(stressed)
        score = model.predict_proba(stressed)[:, 1]

        rows.append({
            "feature": feature,
            "multiplier": float(multiplier),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "mean_attack_score": float(np.mean(score)),
        })

    return pd.DataFrame(rows)

def clean_and_stressed_metrics(
    model,
    X_clean,
    y_clean,
    X_stressed,
    y_stressed=None,
):
    """Return comparable clean/stressed precision, recall, and F1."""
    if y_stressed is None:
        y_stressed = y_clean

    rows = []
    for label, X_eval, y_eval in [
        ("clean", X_clean, y_clean),
        ("stressed", X_stressed, y_stressed),
    ]:
        pred = model.predict(X_eval)
        rows.append({
            "condition": label,
            "precision": float(precision_score(y_eval, pred, zero_division=0)),
            "recall": float(recall_score(y_eval, pred, zero_division=0)),
            "f1": float(f1_score(y_eval, pred, zero_division=0)),
        })

    return pd.DataFrame(rows).set_index("condition")
