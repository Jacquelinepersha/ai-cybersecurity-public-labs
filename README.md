# AI Cybersecurity — Public Labs

Four hands-on labs applying machine learning to network threat detection, using the UNSW-NB15 dataset. Built to accompany the AI Cybersecurity Masterclass, and shared here as public, runnable evidence of the work — not just slides.

This is an independent educational project, not an accredited certification program.

## Labs

1. **Security Data Exploration** — inspect the dataset, understand `label` vs `attack_cat`, check class balance and possible leakage.
2. **Binary Threat Detection** — train Logistic Regression and Random Forest baselines to separate normal from attack traffic, and compare their errors, not just their scores.
3. **Multiclass Attack Classification** — predict attack families and compare per-class precision, recall, F1, macro averages, and weighted averages.
4. **Feature Engineering for Cybersecurity** — design defensible derived features and test whether they actually improve held-out performance.

Each notebook runs in Google Colab or locally in Jupyter — the same file works either way.

## Setup

```bash
pip install -r requirements.txt
```

Place the official UNSW-NB15 prepared CSV files in `data/raw/`:

```
data/raw/
├── UNSW_NB15_training-set.csv
├── UNSW_NB15_testing-set.csv
└── UNSW_NB15_features.csv   (optional)
```

The raw dataset is intentionally excluded from this repository — see `data/README.md`.

Then open any notebook in `notebooks/` and run it top to bottom.

## Repository structure

```
ai-cybersecurity-public-labs/
├── notebooks/     # the 4 labs
├── src/           # reusable data loading, preprocessing, modeling, and evaluation code
├── data/          # dataset instructions (raw data not included)
├── requirements.txt
└── LICENSE
```

## Core technology

Python · pandas · scikit-learn · XGBoost · matplotlib · Jupyter / Google Colab · UNSW-NB15

## Method

Each lab follows the same structure: research question, method, results, and a short written interpretation — not just a final score. The core rule throughout: evaluate precision, recall, and F1 alongside accuracy, since accuracy alone can hide the failures that matter most in security data.
