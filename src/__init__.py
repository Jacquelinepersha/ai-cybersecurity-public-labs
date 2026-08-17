"""
Shared code for Labs 01-05 of the AI Cybersecurity Masterclass.

Five modules, in the order the labs use them:

    data_loader    load_unsw, load_feature_dictionary
    preprocessing  split_xy, align_columns, build_preprocessor
    features       drop_identifier_like_columns, add_safe_derived_features
    models         logistic_pipeline, random_forest_pipeline, xgboost_pipeline
    evaluation     binary_metrics, binary_metrics_frame, multiclass_report
"""

from .data_loader import load_unsw, load_feature_dictionary
from .preprocessing import split_xy, align_columns, build_preprocessor
from .features import drop_identifier_like_columns, add_safe_derived_features
from .models import (
    logistic_pipeline,
    random_forest_pipeline,
    xgboost_pipeline,
)
from .evaluation import (
    binary_metrics,
    binary_metrics_frame,
    multiclass_report,
    error_rows,
)
