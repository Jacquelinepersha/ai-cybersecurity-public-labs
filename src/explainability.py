
import pandas as pd
from sklearn.inspection import permutation_importance

def permutation_importance_frame(
    estimator,
    X,
    y,
    scoring="f1",
    n_repeats=5,
    random_state=42,
    max_samples=0.25,
):
    result = permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        max_samples=max_samples,
    )

    frame = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    return frame.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def reference_row(X_train):
    """Build a neutral reference row: median for numeric, mode for categorical."""
    reference = {}
    for column in X_train.columns:
        series = X_train[column]
        if pd.api.types.is_numeric_dtype(series):
            reference[column] = series.median()
        else:
            reference[column] = series.mode(dropna=True).iloc[0]
    return pd.Series(reference)


def local_occlusion_explanation(model, row, reference, positive_class=1):
    """
    Explain ONE alert: replace each feature with its reference value and
    measure how much the attack score drops.

    A large positive `score_drop` means the model's score for this event
    depended on that feature's value. It shows model reliance for this
    event, not the cause of the attack.
    """
    row = row.to_frame().T if isinstance(row, pd.Series) else row.iloc[[0]]
    class_index = list(model.classes_).index(positive_class)
    base_score = float(model.predict_proba(row)[0, class_index])

    variants = []
    for column in row.columns:
        variant = row.copy()
        variant[column] = reference[column]
        variants.append(variant)
    batch = pd.concat(variants, ignore_index=True)
    for column in row.columns:
        batch[column] = batch[column].astype(row[column].dtype, errors="ignore")
    scores = model.predict_proba(batch)[:, class_index]

    frame = pd.DataFrame({
        "feature": row.columns,
        "event_value": row.iloc[0].values,
        "reference_value": [reference[c] for c in row.columns],
        "score_without_feature": scores,
    })
    frame["score_drop"] = base_score - frame["score_without_feature"]
    frame = frame.sort_values("score_drop", ascending=False).reset_index(drop=True)
    return base_score, frame


def alert_explanation_sentence(
    base_score,
    explanation,
    threshold=0.5,
    top_n=3,
    context_hint=None,
):
    """Turn a local explanation into one sentence an analyst can read quickly."""
    verdict = "attack" if base_score >= threshold else "normal"
    drivers = explanation[explanation["score_drop"] > 0].head(top_n)
    if drivers.empty:
        signal_text = "no single feature moved the score on its own"
    else:
        def fmt(value):
            return f"{value:.3g}" if isinstance(value, float) else str(value)

        signal_text = ", ".join(
            f"{r.feature}={fmt(r.event_value)}" for r in drivers.itertuples()
        )
    sentence = (
        f"Predicted {verdict} (score {base_score:.2f}). "
        f"Signals the model relied on: {signal_text}."
    )
    if context_hint:
        sentence += f" Consistent with {context_hint}, not proof of it."
    sentence += " Validate with related telemetry before acting."
    return sentence
