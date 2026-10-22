
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)

def apply_threshold(scores, threshold):
    return (
        np.asarray(scores) >= float(threshold)
    ).astype(int)

def threshold_sweep(
    y_true,
    scores,
    thresholds=None,
    beta=2.0,
    false_negative_cost=10.0,
    false_positive_cost=1.0,
):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    rows = []
    n = len(y_true)

    for threshold in thresholds:
        pred = apply_threshold(scores, threshold)
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            pred,
            labels=[0, 1],
        ).ravel()

        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(
                y_true, pred, zero_division=0
            )),
            "recall": float(recall_score(
                y_true, pred, zero_division=0
            )),
            "f_beta": float(fbeta_score(
                y_true, pred, beta=beta, zero_division=0
            )),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "normalized_cost": float(
                (
                    false_negative_cost * fn
                    + false_positive_cost * fp
                ) / n
            ),
        })

    return pd.DataFrame(rows)

def choose_threshold(table, objective="f_beta"):
    if objective == "f_beta":
        row = table.loc[table["f_beta"].idxmax()]
    elif objective == "cost":
        row = table.loc[
            table["normalized_cost"].idxmin()
        ]
    else:
        raise ValueError(
            "objective must be 'f_beta' or 'cost'."
        )

    return float(row["threshold"]), row
