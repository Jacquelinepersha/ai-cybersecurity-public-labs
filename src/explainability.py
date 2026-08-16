
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
