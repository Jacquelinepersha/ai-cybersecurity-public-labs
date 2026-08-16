
import numpy as np
import pandas as pd

def random_label_flip(y, fraction, random_state=42):
    """
    Flip a fraction of binary labels in a copied training target.
    Intended only for offline robustness research.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")

    out = pd.Series(y).reset_index(drop=True).copy()

    if fraction == 0 or len(out) == 0:
        return out

    n_flip = max(1, int(round(len(out) * fraction)))
    n_flip = min(n_flip, len(out))

    rng = np.random.default_rng(random_state)
    indices = rng.choice(
        np.arange(len(out)),
        size=n_flip,
        replace=False,
    )

    unique = set(out.dropna().unique())
    if not unique.issubset({0, 1}):
        raise ValueError("random_label_flip expects a binary 0/1 target.")

    out.iloc[indices] = 1 - out.iloc[indices].astype(int)
    return out

def corrupt_numeric_training_rows(
    X,
    fraction,
    columns,
    noise_fraction=0.25,
    bounds=None,
    random_state=42,
):
    """
    Corrupt selected numeric features in a fraction of copied training rows.
    Noise is bounded and multiplicative. No external system is involved.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")
    if noise_fraction < 0:
        raise ValueError("noise_fraction must be non-negative.")

    out = X.reset_index(drop=True).copy()
    if fraction == 0 or len(out) == 0:
        return out

    n_rows = max(1, int(round(len(out) * fraction)))
    n_rows = min(n_rows, len(out))

    rng = np.random.default_rng(random_state)
    indices = rng.choice(
        np.arange(len(out)),
        size=n_rows,
        replace=False,
    )

    for column in columns:
        if column not in out.columns:
            continue

        values = pd.to_numeric(
            out.loc[indices, column],
            errors="coerce",
        ).astype(float)

        noise = rng.uniform(
            -noise_fraction,
            noise_fraction,
            size=len(indices),
        )
        corrupted = values * (1.0 + noise)

        if bounds and column in bounds:
            low, high = bounds[column]
            corrupted = corrupted.clip(lower=low, upper=high)

        if out[column].dtype.kind in "iu":  # int/unsigned columns can't hold noised floats
            out[column] = out[column].astype(float)

        out.loc[indices, column] = corrupted.to_numpy()

    return out

def poisoning_summary(y_clean, y_poisoned):
    """Summarize how many binary labels changed."""
    clean = pd.Series(y_clean).reset_index(drop=True)
    poisoned = pd.Series(y_poisoned).reset_index(drop=True)

    if len(clean) != len(poisoned):
        raise ValueError("Targets must have equal length.")

    changed = clean != poisoned

    return {
        "rows": int(len(clean)),
        "changed_labels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
    }
