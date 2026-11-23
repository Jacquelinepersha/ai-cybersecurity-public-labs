# AI Cybersecurity Masterclass: Public Labs

[![Red-team regression](https://github.com/Jacquelinepersha/ai-cybersecurity-public-labs/actions/workflows/redteam-regression.yml/badge.svg)](https://github.com/Jacquelinepersha/ai-cybersecurity-public-labs/actions/workflows/redteam-regression.yml)

33 hands-on labs that apply machine learning to network threat detection, and then secure the AI systems themselves. Built to accompany the [AI Cybersecurity MasterClass playlist](https://www.youtube.com/playlist?list=PLG4l6Ssi9A00) from [AI60secLab](https://ai60seclab.com). Every lab runs end to end, and the notebooks are committed **with their outputs**, so you can read the results on GitHub without running anything.

This is an independent educational project, not an accredited certification program.

## What this course is, and what it is not

**It is** a practical course in two things:
1. building and honestly evaluating ML detectors on security telemetry (Units 1 to 4);
2. securing AI systems: prompt injection, agents, hallucinations, data leakage, red teaming, incident response, and measurement (Unit 5);
3. hunting without labels and automating red teaming in CI, including against real models (Unit 6).

**It is not** a complete cybersecurity curriculum. It does not teach networking, identity and access management, cryptography, vulnerability management, or SOC tooling on their own. Lab 00b gives the minimum security background the labs need; for the wider field, pair this course with a fundamentals track such as CompTIA Security+ study material or a hands-on SOC analyst path.

## Prerequisites

- **Python, pandas, and scikit-learn:** comfortable. These labs are advanced on the ML side.
- **Security basics:** helpful, not required. Start with Lab 00b if TCP, ports, or ATT&CK tactics are new to you.

## Run a lab in one click

Click any **Open in Colab** badge, then choose **Runtime → Run all**. The first cell clones this repo, which already includes the dataset, so there is nothing to download or upload. Labs 18 to 21 and 25 download the official MITRE ATT&CK or ATLAS data at runtime and need an internet connection.

**Levels:** *Apprentice* labs need only the prerequisites above. *Practitioner* labs assume the earlier units. *Expert* labs assume you are comfortable designing experiments and reading security research.

### Unit 1. Foundations

| # | Lab | Level | Companion video | Open |
|---|---|---|---|---|
| 00 | [How the labs fit together](notebooks/00_how_these_labs_fit_together.ipynb) | Apprentice | [▶ What Is AI Cybersecurity?](https://www.youtube.com/watch?v=ro8SVq50R5o&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/00_how_these_labs_fit_together.ipynb) |
| 00b | [Security primer: flows, attack categories, dataset artifacts](notebooks/00b_security_primer.ipynb) | Apprentice | none yet | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/00b_security_primer.ipynb) |
| 01 | [Security data exploration](notebooks/01_security_data_exploration.ipynb) | Apprentice | none yet | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/01_security_data_exploration.ipynb) |
| 02 | [Binary threat detection](notebooks/02_binary_threat_detection.ipynb) | Apprentice | [▶ The ML Pipeline Behind Cyber Defense](https://www.youtube.com/watch?v=FJRtcoYgBvg&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/02_binary_threat_detection.ipynb) |
| 03 | [Multiclass attack classification](notebooks/03_multiclass_attack_classification.ipynb) | Apprentice | [▶ How AI Tells Cyber Attacks Apart](https://www.youtube.com/watch?v=KxNnMcXg2Tw&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/03_multiclass_attack_classification.ipynb) |
| 04 | [Feature engineering for cybersecurity](notebooks/04_feature_engineering.ipynb) | Apprentice | [▶ Feature Engineering for Catching Hackers](https://www.youtube.com/watch?v=_32ty41DpSY&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/04_feature_engineering.ipynb) |
| 05 | [Model comparison for threat detection](notebooks/05_model_comparison.ipynb) | Practitioner | [▶ Which AI Model Catches Hackers Best?](https://www.youtube.com/watch?v=EiNTtaDPIco&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/05_model_comparison.ipynb) |

### Unit 2. Evaluation that holds up

| # | Lab | Level | Companion video | Open |
|---|---|---|---|---|
| 06 | [Confusion matrices and error analysis](notebooks/06_confusion_matrices_error_analysis.ipynb) | Apprentice | [▶ Why 99% AI Accuracy Can Still Miss Cyber Attacks](https://www.youtube.com/watch?v=YDzz-1FOm8Q&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/06_confusion_matrices_error_analysis.ipynb) |
| 07 | [Data leakage and validation design](notebooks/07_data_leakage_validation_design.ipynb) | Practitioner | [▶ Is Your Cybersecurity AI Model Cheating?](https://www.youtube.com/watch?v=I-5g9lnQ9t4&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/07_data_leakage_validation_design.ipynb) |
| 08 | [Class imbalance and cost-sensitive learning](notebooks/08_class_imbalance_cost_sensitive_learning.ipynb) | Practitioner | [▶ One Number Decides: Attack or Not?](https://www.youtube.com/watch?v=7yn55E7UxZ4&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/08_class_imbalance_cost_sensitive_learning.ipynb) |
| 09 | [Threshold optimization without test leakage](notebooks/09_threshold_optimization_without_test_leakage.ipynb) | Practitioner | [▶ One Number Decides: Attack or Not?](https://www.youtube.com/watch?v=7yn55E7UxZ4&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/09_threshold_optimization_without_test_leakage.ipynb) |
| 10 | [Explainable cyber ML: global and local explanations](notebooks/10_explainable_cyber_ml_permutation_importance.ipynb) | Practitioner | [▶ Why the AI Flagged This Attack](https://www.youtube.com/watch?v=dmeavNGQ-JQ&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/10_explainable_cyber_ml_permutation_importance.ipynb) |
| 11 | [False-negative forensics and error slicing](notebooks/11_false_negative_forensics_error_slicing.ipynb) | Practitioner | none yet | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/11_false_negative_forensics_error_slicing.ipynb) |

### Unit 3. Robustness and adversarial ML

| # | Lab | Level | Companion video | Open |
|---|---|---|---|---|
| 12 | [Controlled evasion-style robustness testing](notebooks/12_controlled_evasion_style_robustness.ipynb) | Expert | [▶ How Hackers Fool AI Models](https://www.youtube.com/watch?v=deKvWTAQ0To&list=PLG4l6Ssi9A00)<br>[▶ Can a Small Change Make Cybersecurity AI Miss an Attack?](https://www.youtube.com/watch?v=FPYX_cPciUw&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/12_controlled_evasion_style_robustness.ipynb) |
| 13 | [Directional feature stress and model sensitivity](notebooks/13_directional_feature_stress_model_sensitivity.ipynb) | Expert | none yet | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/13_directional_feature_stress_model_sensitivity.ipynb) |
| 14 | [Data poisoning robustness](notebooks/14_data_poisoning_robustness.ipynb) | Expert | [▶ How Attackers Poison AI Training Data](https://www.youtube.com/watch?v=IOadMjp3aCU&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/14_data_poisoning_robustness.ipynb) |
| 15 | [Controlled concept drift and adaptation](notebooks/15_controlled_concept_drift_adaptation.ipynb) | Expert | [▶ Why Security AI Gets Dumber Over Time](https://www.youtube.com/watch?v=7BqRTjWFUBw&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/15_controlled_concept_drift_adaptation.ipynb) |
| 16 | [Distribution shift and generalization](notebooks/16_distribution_shift_generalization.ipynb) | Expert | [▶ Why Security AI Gets Dumber Over Time](https://www.youtube.com/watch?v=7BqRTjWFUBw&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/16_distribution_shift_generalization.ipynb) |
| 17 | [Robustness mitigation comparison](notebooks/17_robustness_mitigation_comparison.ipynb) | Expert | [▶ Making Security AI Harder to Fool](https://www.youtube.com/watch?v=ug9UqkDUgEo&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/17_robustness_mitigation_comparison.ipynb) |

### Unit 4. Threat intelligence and capstone

| # | Lab | Level | Companion video | Open |
|---|---|---|---|---|
| 18 | [MITRE ATT&CK threat mapping with Python](notebooks/18_mitre_attack_threat_mapping_python.ipynb) | Practitioner | [▶ MITRE ATT&CK: The Hacker Playbook, Explained](https://www.youtube.com/watch?v=aLlwzBgRYJg&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/18_mitre_attack_threat_mapping_python.ipynb) |
| 19 | [MITRE ATLAS and AI threat modeling](notebooks/19_mitre_atlas_ai_threat_modeling.ipynb) | Expert | [▶ MITRE ATLAS: The AI-Attack Playbook](https://www.youtube.com/watch?v=QAUlOZvrlQY&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/19_mitre_atlas_ai_threat_modeling.ipynb) |
| 20 | [Cyber threat intelligence analytics](notebooks/20_cyber_threat_intelligence_analytics.ipynb) | Practitioner | [▶ Cyber Threat Intelligence in 60 Seconds](https://www.youtube.com/watch?v=JNR7B3pcemo&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/20_cyber_threat_intelligence_analytics.ipynb) |
| 21 | [RAG for CTI: auditable retrieval and poisoning recovery](notebooks/21_rag_cti_retrieval_poisoning.ipynb) | Expert | [▶ How RAG Powers Cyber Threat Intelligence in 60 Seconds](https://www.youtube.com/watch?v=EfpYLez4Fu8&list=PLG4l6Ssi9A00)<br>[▶ How Data and RAG Poisoning Corrupt Cybersecurity AI in 60 Seconds](https://www.youtube.com/watch?v=HfcSu6s-pQA&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/21_rag_cti_retrieval_poisoning.ipynb) |
| 22 | [Final research capstone](notebooks/22_final_ai_cybersecurity_research_capstone.ipynb) | Expert | [▶ What Is Experimental Design in Cybersecurity ML?](https://www.youtube.com/watch?v=cqDkcR_TJp8&list=PLG4l6Ssi9A00)<br>[▶ From Security Model to Published Research](https://www.youtube.com/watch?v=cUUY_uEApcE&list=PLG4l6Ssi9A00)<br>[▶ Turn Security AI Into Decisions, Not Just Alerts](https://www.youtube.com/watch?v=O1Rr-mCNrWY&list=PLG4l6Ssi9A00)<br>[▶ How to Finalize a Cybersecurity Research Paper in 60 Seconds](https://www.youtube.com/watch?v=4Uby8PYnCGY&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/22_final_ai_cybersecurity_research_capstone.ipynb) |

### Unit 5. LLM and agent security

| # | Lab | Level | Companion video | Open |
|---|---|---|---|---|
| 23 | [Prompt injection and layered controls](notebooks/23_prompt_injection_layered_controls.ipynb) | Practitioner | [▶ How Prompt Injection Works in Cybersecurity in 60 Seconds](https://www.youtube.com/watch?v=ZEZMfywIPp0&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/23_prompt_injection_layered_controls.ipynb) |
| 24 | [Securing an AI security agent](notebooks/24_secure_ai_agents_least_privilege.ipynb) | Expert | [▶ How an AI Security Agent Works in 60 Seconds](https://www.youtube.com/watch?v=MT3GFAZcNQk&list=PLG4l6Ssi9A00)<br>[▶ How to Secure AI Agents in 60 Seconds](https://www.youtube.com/watch?v=U9Hujhc8768&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/24_secure_ai_agents_least_privilege.ipynb) |
| 25 | [Evaluating LLM hallucinations in security answers](notebooks/25_llm_hallucination_evaluation.ipynb) | Practitioner | [▶ How to Test LLM Hallucinations in Cybersecurity in 60 Seconds](https://www.youtube.com/watch?v=iCBlRHD7bxw&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/25_llm_hallucination_evaluation.ipynb) |
| 26 | [How AI leaks data: extraction, membership, canaries](notebooks/26_ai_data_leakage_extraction_membership.ipynb) | Expert | [▶ How AI Leaks Sensitive Data in 60 Seconds](https://www.youtube.com/watch?v=lZMdVRU6lZM&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/26_ai_data_leakage_extraction_membership.ipynb) |
| 27 | [AI red teaming as a regression loop](notebooks/27_ai_red_teaming_regression.ipynb) | Expert | [▶ How AI Red Teaming Tests AI Safety in 60 Seconds](https://www.youtube.com/watch?v=_KCq0iEtnzk&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/27_ai_red_teaming_regression.ipynb) |
| 28 | [Incident response for a deployed AI detector](notebooks/28_ai_security_incident_response.ipynb) | Expert | [▶ How AI Security Incident Response Works in 60 Seconds](https://www.youtube.com/watch?v=c19cfYykn6w&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/28_ai_security_incident_response.ipynb) |
| 29 | [Measuring AI security: outcomes, not motion](notebooks/29_measuring_ai_security.ipynb) | Expert | [▶ How to Measure AI Security in 60 Seconds](https://www.youtube.com/watch?v=DcbG2ol3XCQ&list=PLG4l6Ssi9A00) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/29_measuring_ai_security.ipynb) |

### Unit 6. Hunting and automation

| # | Lab | Level | Companion video | Open |
|---|---|---|---|---|
| 30 | [Unsupervised threat hunting](notebooks/30_unsupervised_threat_hunting.ipynb) | Practitioner | none yet | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/30_unsupervised_threat_hunting.ipynb) |
| 31 | [Automated red teaming, CI gates, and real models](notebooks/31_automated_red_teaming_ci_live_models.ipynb) | Expert | none yet | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jacquelinepersha/ai-cybersecurity-public-labs/blob/main/notebooks/31_automated_red_teaming_ci_live_models.ipynb) |

## How Unit 5 works without an API key

Unit 5 labs use a deterministic, simulated assistant (`src/llm_app.py`) and agent (`src/agent_security.py`). They reproduce a documented failure of real LLMs: they understand instructions written in many phrasings, including ones hidden in retrieved data, and act on them. That makes every result repeatable and free to run. The **controls** you build and measure (filters, trust separation, schema validation, deterministic policy, output filters, logging, approval gates) are real code that could sit in front of a real model. What the simulation cannot tell you is how often a specific production model falls for a specific payload; that needs testing against that model.

Labs 24, 26, 28, and 29 use real UNSW-NB15 traffic and real models. Lab 25 verifies answers against the live ATT&CK release.

**Want to test a real LLM?** Lab 31 connects the same controls to any OpenAI-compatible endpoint, including a free local model through Ollama, by setting `LLM_BASE_URL` and `LLM_MODEL`. It also shows how to run the open-source scanners garak and PyRIT, and how this repository's red-team regression suite (`tests/`) runs automatically on every push through GitHub Actions.

## Things the data will try to fool you with

Lab 00b measures these directly. Keep them in mind in every lab:
- **The class balance is upside down.** About 68% of training rows are attacks. Real networks are overwhelmingly normal, so precision here looks better than it would in production.
- **One field does a lot of the work.** Source TTL (`sttl`) largely encodes which testbed machine generated the traffic. A one-line TTL rule beats the majority baseline with no learning at all. That is a dataset artifact, not attack behavior.
- **Some categories are indistinguishable.** Large parts of Analysis, Backdoor, and DoS share the exact same flow shape.
- **42% of training rows are exact duplicates.** This matters for leakage (Lab 07) and for membership inference (Lab 26).
- **No IPs, ports, or timestamps.** The prepared partition cannot support entity grouping or train-on-past, test-on-future validation.

## Run locally

```bash
git clone https://github.com/Jacquelinepersha/ai-cybersecurity-public-labs.git
cd ai-cybersecurity-public-labs
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/
```

The dataset loads automatically from `data/raw/` (see `data/README.md`).

`requirements.txt` pins the exact library versions every lab was tested with end to end (Python 3.13), so local results reproduce. Colab uses its own preinstalled versions, which can shift results slightly.

**On a Mac**, XGBoost (used only in Lab 05) also needs the OpenMP runtime, which macOS does not include. Install it once with [Homebrew](https://brew.sh): `brew install libomp`, then restart the Jupyter kernel. If you skip this, Lab 05 compares the other two models and continues. Colab needs nothing extra.

## Repository structure

```
ai-cybersecurity-public-labs/
├── notebooks/     # labs 00-31, committed with outputs
├── src/           # shared data, modeling, evaluation, and security helpers
├── tests/         # red-team regression suite (runs in GitHub Actions)
├── data/raw/      # compressed UNSW-NB15 prepared train/test partition
├── requirements.txt
└── LICENSE
```

## Method

Each lab follows the same structure: a **Security foundation** section with the cybersecurity concepts the lab needs, then the research question, method, results, and a short written interpretation, and finally **From lab to SOC**, which connects the result to real security operations and ends with check questions. The core rules throughout:
- evaluate precision, recall, and F1 alongside accuracy, since accuracy alone hides the failures that matter most in security data;
- change one factor at a time against a named baseline;
- repeat with several seeds, and only count an effect larger than the spread between runs;
- say what the result does not prove.

## Build your portfolio from this

[▶ How to Build an AI Cybersecurity Portfolio in 60 Seconds](https://www.youtube.com/watch?v=Fb8eycYHQhE&list=PLG4l6Ssi9A00). The capstone (Lab 22, which drafts a decision report mapped to MITRE ATT&CK, MITRE ATLAS, OWASP, and the NIST AI RMF) and the Unit 5 scorecard (Lab 29) are designed to be shown: they state a question, a method, evidence, and limits.

## Core technology

Python · pandas · scikit-learn · XGBoost · matplotlib · Jupyter / Google Colab · UNSW-NB15 · MITRE ATT&CK · MITRE ATLAS

## Dataset credit

UNSW-NB15 was created by Nour Moustafa and Jill Slay at the Cyber Range Lab, UNSW Canberra. It is free to use for academic research; commercial use must be agreed with the authors. See `data/README.md` for the required citations.

The code in this repository is MIT licensed. The dataset is not covered by that license.
