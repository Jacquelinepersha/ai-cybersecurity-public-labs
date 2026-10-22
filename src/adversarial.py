
import numpy as np
import pandas as pd

DEFAULT_NETWORK_NUMERIC_CANDIDATES = [
    "dur",
    "sbytes",
    "dbytes",
    "spkts",
    "dpkts",
    "rate",
    "sload",
    "dload",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
]

def available_numeric_attack_features(
    X,
    candidates=None,
):
    candidates = (
        candidates
        or DEFAULT_NETWORK_NUMERIC_CANDIDATES
    )
    numeric = set(
        X.select_dtypes(
            include=["number", "bool"]
        ).columns
    )

    return [
        c for c in candidates
        if c in X.columns and c in numeric
    ]

def training_bounds(X_train, columns):
    bounds = {}
    for column in columns:
        values = pd.to_numeric(
            X_train[column],
            errors="coerce",
        )
        bounds[column] = (
            float(values.min()),
            float(values.max()),
        )
    return bounds

def bounded_random_perturbation(
    X,
    columns,
    epsilon,
    bounds=None,
    random_state=42,
):
    if epsilon < 0:
        raise ValueError(
            "epsilon must be non-negative."
        )

    out = X.copy()
    rng = np.random.default_rng(random_state)

    for column in columns:
        if column not in out.columns:
            continue

        values = pd.to_numeric(
            out[column],
            errors="coerce",
        ).astype(float)

        noise = rng.uniform(
            -epsilon,
            epsilon,
            size=len(out),
        )
        perturbed = values * (1.0 + noise)

        if bounds and column in bounds:
            low, high = bounds[column]
            perturbed = perturbed.clip(
                lower=low,
                upper=high,
            )

        out[column] = perturbed

    return out

def augment_attacks_with_noise(
    X_train,
    y_train,
    columns,
    epsilon=0.05,
    copies=1,
    positive_label=1,
    bounds=None,
    random_state=42,
):
    if copies < 1:
        raise ValueError("copies must be >= 1.")

    X_train = X_train.reset_index(drop=True)
    y_train = pd.Series(
        y_train
    ).reset_index(drop=True)

    attack_rows = X_train[
        y_train == positive_label
    ].copy()

    if attack_rows.empty:
        raise ValueError(
            "No positive-class rows found."
        )

    all_X = [X_train]
    all_y = [y_train]

    for i in range(copies):
        perturbed = bounded_random_perturbation(
            attack_rows,
            columns=columns,
            epsilon=epsilon,
            bounds=bounds,
            random_state=random_state + i,
        )
        all_X.append(perturbed)
        all_y.append(
            pd.Series(
                [positive_label] * len(perturbed)
            )
        )

    return (
        pd.concat(all_X, ignore_index=True),
        pd.concat(all_y, ignore_index=True),
    )
