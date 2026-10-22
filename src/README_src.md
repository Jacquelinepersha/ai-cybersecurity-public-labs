# src/: shared code for all labs

| File | Provides |
|---|---|
| `data_loader.py` | `load_unsw`, `load_feature_dictionary` (auto-loads the bundled dataset) |
| `preprocessing.py` | `split_xy`, `align_columns`, `build_preprocessor` |
| `features.py` | `drop_identifier_like_columns`, `add_safe_derived_features` |
| `models.py` | `logistic_pipeline`, `random_forest_pipeline`, `xgboost_pipeline` |
| `evaluation.py` | `binary_metrics`, `binary_metrics_frame`, `multiclass_report`, `error_rows` |
| `imbalance.py` | class balance and resampling helpers (Lab 08) |
| `thresholds.py` | threshold sweeps and selection (Labs 09, 17, 22) |
| `explainability.py` | permutation importance (Lab 10) |
| `forensics.py` | false-negative error slicing (Lab 11) |
| `adversarial.py` | evasion-style perturbations (Labs 12, 17, 22) |
| `robustness.py` | feature-stress testing and bounds (Labs 13, 14) |
| `poisoning.py` | label-flip and corruption experiments (Labs 14, 22) |
| `drift.py` | concept drift and distribution shift (Labs 15, 16, 22) |
| `mitre_data.py` | MITRE ATT&CK / ATLAS STIX loading (Labs 18–21) |
| `cti.py` | CTI analytics (Lab 20) |
| `rag_security.py` | RAG retrieval and poisoning checks (Lab 21) |

`xgboost` is only needed where a lab uses `xgboost_pipeline()`. Without it, that function raises a clear install message.
