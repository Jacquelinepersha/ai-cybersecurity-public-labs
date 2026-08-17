
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: float | None

def binary_metrics(y_true, y_pred, y_score=None):
    """Calculate security-relevant binary-classification metrics."""
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    auc = None
    if y_score is not None:
        auc = float(roc_auc_score(y_true, y_score))

    return BinaryMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        false_positive_rate=float(fpr),
        false_negative_rate=float(fnr),
        roc_auc=auc,
    )

def binary_metrics_frame(model_name, metrics):
    """Convert BinaryMetrics into a one-row DataFrame."""
    row = asdict(metrics)
    row["model"] = model_name
    return pd.DataFrame([row]).set_index("model")

def multiclass_report(y_true, y_pred):
    """Return a per-class classification report."""
    return pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
    ).T

def error_rows(X_original, y_true, y_pred):
    """Attach actual/predicted labels and identify binary error types."""
    result = X_original.copy().reset_index(drop=True)
    result["actual"] = np.asarray(y_true)
    result["predicted"] = np.asarray(y_pred)
    result["error_type"] = "correct"

    fp_mask = (result["actual"] == 0) & (result["predicted"] == 1)
    fn_mask = (result["actual"] == 1) & (result["predicted"] == 0)

    result.loc[fp_mask, "error_type"] = "false_positive"
    result.loc[fn_mask, "error_type"] = "false_negative"
    return result
