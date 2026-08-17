# src/ — what Labs 01–05 need

Six files. Copy the whole `src/` folder into your project root, next to `notebooks/`.

| File | Provides | Used by |
|---|---|---|
| `data_loader.py` | `load_unsw`, `load_feature_dictionary` | 00, 01, 02, 03, 04, 05 |
| `preprocessing.py` | `split_xy`, `align_columns`, `build_preprocessor` | 02, 03, 04, 05 |
| `features.py` | `drop_identifier_like_columns`, `add_safe_derived_features` | 02, 03, 04, 05 |
| `models.py` | `logistic_pipeline`, `random_forest_pipeline`, `xgboost_pipeline` | 03, 04, 05 |
| `evaluation.py` | `binary_metrics`, `binary_metrics_frame`, `multiclass_report` | 02, 03, 04, 05 |
| `__init__.py` | makes `src` a package; re-exports the above | all |

## Install

```
your-project/
├── notebooks/
├── src/          <- this folder
└── data/raw/     <- the UNSW-NB15 CSVs
```

## IMPORTANT — merge, do not replace

If your repo `src/` already holds the Lessons 06–40 modules (`adversarial.py`,
`poisoning.py`, `drift.py`, `imbalance.py`, `thresholds.py`, `robustness.py`,
`cti.py`, `mitre_data.py`, `rag_security.py`, `forensics.py`,
`explainability.py`), copy these six files *alongside* them. Do not overwrite
the folder — `drift.py` imports `preprocessing.py`, so a wholesale replace
breaks it.

If you add those modules later, extend `__init__.py` to export them too.

## Requirements

```
pandas>=2.1  numpy>=1.26  scikit-learn>=1.4  xgboost>=2.0  matplotlib>=3.8
```

`xgboost` is only needed for Lab 05. Without it, `xgboost_pipeline()` raises a
clear install message rather than failing obscurely.

## Note on models.py

This copy takes `class_weight` and `scale_pos_weight` as arguments, so students
can switch class weighting off and see what it was doing:

```python
logistic_pipeline(X_train, class_weight=None)
xgboost_pipeline(X_train, scale_pos_weight=2.5)
```

Verified: all import paths and code paths used by the five notebooks execute
cleanly against scikit-learn 1.7 / pandas 2.3 / xgboost 3.2.
