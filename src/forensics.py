
import numpy as np
import pandas as pd

def attach_prediction_context(
    raw_test,
    y_true,
    y_pred,
    y_score=None,
):
    out = raw_test.reset_index(drop=True).copy()
    out["actual"] = np.asarray(y_true)
    out["predicted"] = np.asarray(y_pred)

    if y_score is not None:
        out["attack_score"] = np.asarray(y_score)

    out["error_type"] = "correct"
    out.loc[
        (out["actual"] == 0)
        & (out["predicted"] == 1),
        "error_type",
    ] = "false_positive"
    out.loc[
        (out["actual"] == 1)
        & (out["predicted"] == 0),
        "error_type",
    ] = "false_negative"
    out.loc[
        (out["actual"] == 1)
        & (out["predicted"] == 1),
        "error_type",
    ] = "true_positive"
    out.loc[
        (out["actual"] == 0)
        & (out["predicted"] == 0),
        "error_type",
    ] = "true_negative"

    return out

def error_slice_by_category(
    forensic_table,
    category_column,
):
    if category_column not in forensic_table.columns:
        raise KeyError(
            f"Missing category column: {category_column}"
        )

    attacks = forensic_table[
        forensic_table["actual"] == 1
    ].copy()

    aggregations = {
        "attack_count": ("actual", "size"),
        "detected_count": ("predicted", "sum"),
    }

    if "attack_score" in attacks.columns:
        aggregations["mean_attack_score"] = (
            "attack_score",
            "mean",
        )

    grouped = attacks.groupby(
        category_column,
        dropna=False,
    ).agg(**aggregations)

    grouped["missed_count"] = (
        grouped["attack_count"]
        - grouped["detected_count"]
    )
    grouped["recall"] = (
        grouped["detected_count"]
        / grouped["attack_count"]
    )

    return grouped.sort_values(
        ["recall", "attack_count"],
        ascending=[True, False],
    )

def numeric_false_negative_contrast(
    forensic_table,
    numeric_columns,
):
    fn = forensic_table[
        forensic_table["error_type"]
        == "false_negative"
    ]
    tp = forensic_table[
        forensic_table["error_type"]
        == "true_positive"
    ]

    rows = []
    for column in numeric_columns:
        if column not in forensic_table.columns:
            continue

        fn_median = pd.to_numeric(
            fn[column],
            errors="coerce",
        ).median()
        tp_median = pd.to_numeric(
            tp[column],
            errors="coerce",
        ).median()

        rows.append({
            "feature": column,
            "false_negative_median": fn_median,
            "true_positive_median": tp_median,
            "difference": fn_median - tp_median,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    return frame.assign(
        absolute_difference=frame["difference"].abs()
    ).sort_values(
        "absolute_difference",
        ascending=False,
    ).drop(
        columns="absolute_difference"
    )
