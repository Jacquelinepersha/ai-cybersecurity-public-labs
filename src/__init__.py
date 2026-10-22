"""
Shared code for the AI Cybersecurity Masterclass labs.

Core modules (Labs 01-07). Later labs import their own helper modules
(imbalance, thresholds, robustness, poisoning, drift, mitre_data, ...)
directly, for example `from src.drift import domain_classifier_auc`:

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
