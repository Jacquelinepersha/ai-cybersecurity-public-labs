
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor

def normalized_median_shift(reference, current, columns):
    """
    Compare medians after normalizing by reference IQR.
    Larger absolute values indicate stronger marginal shift.
    """
    rows = []

    for column in columns:
        if column not in reference.columns or column not in current.columns:
            continue

        ref = pd.to_numeric(reference[column], errors="coerce")
        cur = pd.to_numeric(current[column], errors="coerce")

        q1 = ref.quantile(0.25)
        q3 = ref.quantile(0.75)
        iqr = q3 - q1

        if pd.isna(iqr) or iqr == 0:
            normalized = np.nan
        else:
            normalized = (cur.median() - ref.median()) / iqr

        rows.append({
            "feature": column,
            "reference_median": float(ref.median()),
            "current_median": float(cur.median()),
            "reference_iqr": float(iqr) if pd.notna(iqr) else np.nan,
            "normalized_median_shift": float(normalized)
            if pd.notna(normalized)
            else np.nan,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    return frame.assign(
        absolute_shift=frame["normalized_median_shift"].abs()
    ).sort_values(
        "absolute_shift",
        ascending=False,
    ).drop(columns="absolute_shift")

def domain_classifier_auc(
    reference_X,
    current_X,
    sample_per_domain=20000,
    random_state=42,
):
    """
    Train a classifier to distinguish reference rows from current rows.
    AUC near 0.5 means the sampled domains are difficult to distinguish.
    """
    ref = reference_X.copy()
    cur = current_X.copy()
    rng = np.random.default_rng(random_state)

    if len(ref) > sample_per_domain:
        ref = ref.iloc[
            rng.choice(len(ref), sample_per_domain, replace=False)
        ].copy()

    if len(cur) > sample_per_domain:
        cur = cur.iloc[
            rng.choice(len(cur), sample_per_domain, replace=False)
        ].copy()

    combined = pd.concat([ref, cur], ignore_index=True)
    domain_y = np.concatenate([
        np.zeros(len(ref), dtype=int),
        np.ones(len(cur), dtype=int),
    ])

    X_train, X_valid, y_train, y_valid = train_test_split(
        combined,
        domain_y,
        test_size=0.30,
        stratify=domain_y,
        random_state=random_state,
    )

    model = Pipeline([
        ("preprocess", build_preprocessor(X_train, scale_numeric=True)),
        ("model", LogisticRegression(max_iter=1500, random_state=random_state)),
    ])

    model.fit(X_train, y_train)
    score = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, score)

    return float(auc), model

def simulate_region_concept_shift(
    X,
    y,
    feature,
    quantile=0.75,
    flip_probability=0.50,
    random_state=42,
):
    """
    Simulate concept drift by changing P(y|x) inside a defined feature region.
    This modifies copied labels only and is explicitly a teaching simulation.
    """
    if feature not in X.columns:
        raise KeyError(f"Missing feature: {feature}")
    if not 0 <= flip_probability <= 1:
        raise ValueError("flip_probability must be between 0 and 1.")

    numeric = pd.to_numeric(X[feature], errors="coerce")
    threshold = numeric.quantile(quantile)
    region = numeric >= threshold

    shifted_y = pd.Series(y).reset_index(drop=True).copy()
    region_indices = np.where(region.fillna(False).to_numpy())[0]

    rng = np.random.default_rng(random_state)
    flip_mask = rng.random(len(region_indices)) < flip_probability
    chosen = region_indices[flip_mask]

    shifted_y.iloc[chosen] = 1 - shifted_y.iloc[chosen].astype(int)

    metadata = {
        "feature": feature,
        "quantile": float(quantile),
        "threshold": float(threshold),
        "region_rows": int(len(region_indices)),
        "flipped_rows": int(len(chosen)),
    }

    return shifted_y, metadata
