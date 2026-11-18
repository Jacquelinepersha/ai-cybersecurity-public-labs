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
| `explainability.py` | permutation importance and per-alert local explanations (Lab 10) |
| `forensics.py` | false-negative error slicing (Lab 11) |
| `adversarial.py` | evasion-style perturbations (Labs 12, 17, 22) |
| `robustness.py` | feature-stress testing and bounds (Labs 13, 14) |
| `poisoning.py` | label-flip and corruption experiments (Labs 14, 22) |
| `drift.py` | concept drift and distribution shift (Labs 15, 16, 22) |
| `mitre_data.py` | MITRE ATT&CK / ATLAS STIX loading (Labs 18–21) |
| `cti.py` | CTI analytics (Lab 20) |
| `rag_security.py` | RAG retrieval, chunk metadata, retrieval/generation evaluation, provenance, quarantine, restore (Lab 21) |
| `llm_app.py` | simulated SOC assistant, prompt-injection payloads, layered controls, audit log (Labs 23, 26, 27, 29) |
| `agent_security.py` | simulated triage agent, tool risk tiers, task profiles, policy and approval gates (Labs 24, 27) |
| `llm_eval.py` | claim extraction and verification against ATT&CK, failure taxonomy, rubric (Lab 25) |
| `privacy_attacks.py` | model extraction, membership inference, canary scanning of outputs and logs (Labs 26, 29) |
| `redteam.py` | red-team cases with expected safe behavior, campaigns, regression across versions (Labs 27, 29) |
| `ai_monitoring.py` | deployment replay, PSI and baseline signals, model registry and rollback (Lab 28) |
| `ai_metrics.py` | metrics with baseline, threshold, owner, and decision; scorecards (Lab 29) |

`xgboost` is only needed where a lab uses `xgboost_pipeline()`. Without it, that function raises a clear install message.
