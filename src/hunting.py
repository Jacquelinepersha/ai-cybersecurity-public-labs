"""
Unsupervised threat hunting helpers (Lab 30).

Labels are never used to fit anything here. They are only revealed at the
end, to grade the hunt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def realistic_mix(frame, attack_share, n_rows, random_state=0):
    """Sample a traffic window with a chosen attack share (0.0 = clean baseline)."""
    normal = frame[frame["label"] == 0]
    attacks = frame[frame["label"] == 1]
    n_attack = int(round(n_rows * attack_share))
    parts = [normal.sample(n_rows - n_attack, random_state=random_state)]
    if n_attack:
        parts.append(attacks.sample(n_attack, random_state=random_state))
    return pd.concat(parts).sample(frac=1, random_state=random_state).reset_index(drop=True)


def _log1p_nonnegative(values):
    return np.log1p(np.clip(values, 0, None))


def hunting_preprocessor(X):
    """Log-scale skewed counts, standardize, and one-hot the categorical fields."""
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([
            ("log", FunctionTransformer(_log1p_nonnegative)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical),
    ])


def fit_anomaly_model(X_baseline, n_estimators=200, random_state=0):
    """Learn what 'normal' looks like from a baseline window. No labels."""
    return Pipeline([
        ("preprocess", hunting_preprocessor(X_baseline)),
        ("model", IsolationForest(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)),
    ]).fit(X_baseline)


def anomaly_scores(model, X):
    """Higher = more unusual relative to the baseline."""
    return -model.decision_function(X)


def precision_at_k(labels, scores, ks=(50, 100, 300, 1000)):
    """If an analyst reviews the top k flows, how many are real attacks?"""
    labels = np.asarray(labels)
    order = np.argsort(-np.asarray(scores))
    rows = []
    for k in ks:
        top = labels[order[:k]]
        rows.append({
            "analyst_budget_k": k,
            "attacks_found": int(top.sum()),
            "precision_at_k": float(top.mean()),
            "lift_over_random": float(top.mean() / labels.mean()) if labels.mean() else np.nan,
        })
    return pd.DataFrame(rows)


def cluster_window(X_baseline, X_window, n_clusters=12, random_state=0):
    """Group a traffic window into behavior clusters using the baseline's scaling."""
    preprocess = hunting_preprocessor(X_baseline).fit(X_baseline)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5)
    return model.fit_predict(preprocess.transform(X_window))


def cluster_profile(window, clusters):
    """What each cluster looks like, before anyone looks at labels."""
    frame = window.assign(cluster=clusters)
    profile = frame.groupby("cluster").agg(
        flows=("proto", "size"),
        top_proto=("proto", lambda s: s.value_counts().index[0]),
        top_service=("service", lambda s: s.value_counts().index[0]),
        top_state=("state", lambda s: s.value_counts().index[0]),
        median_spkts=("spkts", "median"),
        median_dpkts=("dpkts", "median"),
        median_sbytes=("sbytes", "median"),
        median_ct_srv_src=("ct_srv_src", "median"),
    )
    return profile.sort_values("flows")
