
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

def class_balance_frame(y):
    series = pd.Series(y)
    counts = series.value_counts().sort_index()
    fractions = series.value_counts(normalize=True).sort_index()
    return pd.DataFrame({"count": counts, "fraction": fractions})

def balanced_class_weight_dict(y):
    y_array = np.asarray(y)
    classes = np.unique(y_array)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_array,
    )
    return dict(zip(classes, weights))

def make_controlled_binary_imbalance(
    X,
    y,
    positive_label=1,
    positive_to_negative_ratio=0.10,
    random_state=42,
):
    if not 0 < positive_to_negative_ratio <= 1:
        raise ValueError("positive_to_negative_ratio must be in (0, 1].")

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    pos_idx = y[y == positive_label].index.to_numpy()
    neg_idx = y[y != positive_label].index.to_numpy()

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Both positive and negative classes are required.")

    target_positive = int(round(
        len(neg_idx) * positive_to_negative_ratio
    ))
    target_positive = max(
        1,
        min(target_positive, len(pos_idx)),
    )

    rng = np.random.default_rng(random_state)
    chosen_pos = rng.choice(
        pos_idx,
        size=target_positive,
        replace=False,
    )
    keep = np.concatenate([neg_idx, chosen_pos])
    rng.shuffle(keep)

    return (
        X.iloc[keep].reset_index(drop=True),
        y.iloc[keep].reset_index(drop=True),
    )

def random_undersample_majority(
    X,
    y,
    target_ratio=1.0,
    random_state=42,
):
    if not 0 < target_ratio <= 1:
        raise ValueError("target_ratio must be in (0, 1].")

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    counts = y.value_counts()
    if len(counts) != 2:
        raise ValueError("This helper requires a binary target.")

    minority_label = counts.idxmin()
    majority_label = counts.idxmax()

    minority_idx = y[y == minority_label].index.to_numpy()
    majority_idx = y[y == majority_label].index.to_numpy()

    desired_majority = int(round(
        len(minority_idx) / target_ratio
    ))
    desired_majority = min(
        desired_majority,
        len(majority_idx),
    )

    rng = np.random.default_rng(random_state)
    chosen_majority = rng.choice(
        majority_idx,
        size=desired_majority,
        replace=False,
    )

    keep = np.concatenate([
        minority_idx,
        chosen_majority,
    ])
    rng.shuffle(keep)

    return (
        X.iloc[keep].reset_index(drop=True),
        y.iloc[keep].reset_index(drop=True),
    )
