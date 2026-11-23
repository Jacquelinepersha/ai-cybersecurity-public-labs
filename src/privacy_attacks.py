"""
How an AI system leaks without being "hacked": through its own interface.

Three offline experiments on a model you train yourself:
- model extraction: copy a model's behavior from its API answers;
- membership inference: tell whether a record was in the training data;
- canary testing: plant a fake secret and check whether it shows up in
  outputs or logs.
Everything runs on local copies of public data. No external system is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .preprocessing import build_preprocessor
from sklearn.pipeline import Pipeline


class PredictionAPI:
    """
    Wraps a fitted model the way a service would expose it.

    detail: "label" returns 0/1; "rounded" returns a score rounded to 1
    decimal; "full" returns the raw probability.
    query_budget: maximum rows answered; None means unlimited.
    """

    def __init__(self, model, detail="full", query_budget=None):
        self.model = model
        self.detail = detail
        self.query_budget = query_budget
        self.queries_answered = 0

    def query(self, X):
        if self.query_budget is not None:
            remaining = self.query_budget - self.queries_answered
            if remaining <= 0:
                return np.array([])
            X = X.iloc[:remaining]
        self.queries_answered += len(X)
        score = self.model.predict_proba(X)[:, 1]
        if self.detail == "label":
            return (score >= 0.5).astype(float)
        if self.detail == "rounded":
            return np.round(score, 1)
        return score


def extraction_experiment(api, attacker_pool, holdout, budget, random_state=0):
    """
    The attacker samples `budget` rows from public-looking traffic, queries
    the API, and trains a surrogate on the answers. Fidelity is how often the
    surrogate agrees with the target on traffic neither has seen.

    With labels only, the surrogate is a classifier. When the API returns
    scores, the attacker distills them: a regressor learns the score itself,
    which carries more information than the label (how close each query was to
    the boundary), and its prediction is thresholded at 0.5.
    """
    queries = attacker_pool.sample(n=min(budget, len(attacker_pool)), random_state=random_state)
    answers = api.query(queries)
    if len(answers) == 0:
        return {"budget": budget, "answered": 0, "fidelity": np.nan}
    queries = queries.iloc[:len(answers)]
    labels = (answers >= 0.5).astype(int)
    if len(np.unique(labels)) < 2:
        return {"budget": budget, "answered": len(answers), "fidelity": np.nan}

    preprocess = build_preprocessor(queries, scale_numeric=False)
    if api.detail == "label":
        surrogate = Pipeline([("preprocess", preprocess),
                              ("model", DecisionTreeClassifier(max_depth=12, random_state=random_state))])
        surrogate.fit(queries, labels)
        copied = surrogate.predict(holdout)
    else:
        surrogate = Pipeline([("preprocess", preprocess),
                              ("model", DecisionTreeRegressor(max_depth=12, random_state=random_state))])
        surrogate.fit(queries, answers)
        copied = (surrogate.predict(holdout) >= 0.5).astype(int)

    target_labels = (api.model.predict_proba(holdout)[:, 1] >= 0.5).astype(int)
    fidelity = float((copied == target_labels).mean())
    return {"budget": budget, "answered": len(answers), "fidelity": fidelity}


def membership_scores(model, X, y):
    """Confidence the model assigns to each record's TRUE label."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    idx = np.array([classes.index(v) for v in np.asarray(y)])
    return proba[np.arange(len(idx)), idx]


def _membership_observations(model, X_members, y_members, X_outsiders, y_outsiders, detail):
    def observe(X, y):
        s = membership_scores(model, X, y)
        if detail == "label":
            return (s >= 0.5).astype(float)
        if detail == "rounded":
            return np.round(s, 1)
        return s

    scores = np.concatenate([observe(X_members, y_members), observe(X_outsiders, y_outsiders)])
    truth = np.concatenate([np.ones(len(X_members)), np.zeros(len(X_outsiders))])
    return truth, scores


def membership_inference_auc(model, X_members, y_members, X_outsiders, y_outsiders, detail="full"):
    """
    A simple attack: records the model is unusually sure about are guessed
    to be training members. AUC 0.5 means the attacker learns nothing.
    """
    truth, scores = _membership_observations(model, X_members, y_members, X_outsiders, y_outsiders, detail)
    return float(roc_auc_score(truth, scores))


def membership_tpr_at_fpr(model, X_members, y_members, X_outsiders, y_outsiders, detail="full", fpr=0.01):
    """
    Share of members the attacker identifies while wrongly flagging at most
    `fpr` of outsiders. This is the standard way to report membership
    inference (Carlini et al., IEEE S&P 2022): an attack matters when it can
    name some members with confidence, which an average-case AUC can hide.
    """
    truth, scores = _membership_observations(model, X_members, y_members, X_outsiders, y_outsiders, detail)
    fprs, tprs, _ = roc_curve(truth, scores)
    return float(tprs[np.searchsorted(fprs, fpr, side="right") - 1])


# ---------------------------------------------------------------------------
# Canary testing on an assistant's outputs and logs
# ---------------------------------------------------------------------------
def scan_for_canaries(texts, canaries):
    """Count how many texts contain each canary string."""
    rows = []
    for canary in canaries:
        hits = [t for t in texts if canary in str(t)]
        rows.append({"canary": canary, "hits": len(hits)})
    return pd.DataFrame(rows)


def simulate_assistant_logging(
    cases,
    controls,
    log_system_prompt=True,
    redact=False,
    retention_days=None,
):
    """
    Run cases through the Lab 23 assistant and keep what a typical logging
    setup keeps. A common mistake is logging the full prompt, including the
    system prompt that holds a secret. Returns (outputs, log_records).

    Each case is one day older than the next (newest last); records older
    than `retention_days` are dropped.
    """
    from .llm_app import SYSTEM_PROMPT, redact_secrets, run_pipeline

    outputs, logs = [], []
    n = len(cases)
    for day, case in enumerate(cases):
        result = run_pipeline(case["user_request"], case["retrieved"], controls)
        record = {
            "age_days": n - 1 - day,
            "user_request": case["user_request"],
            "retrieved_text": " ".join(t for _, t, _ in case["retrieved"]),
            "answer": result["answer"],
            "tool_args": " ".join(str(c.args) for c in result["executed"]),
        }
        if log_system_prompt:
            record["system_prompt"] = SYSTEM_PROMPT
        if redact:
            record = {k: redact_secrets(v) if isinstance(v, str) else v for k, v in record.items()}
        outputs.append(result["answer"])
        logs.append(record)
    if retention_days is not None:
        logs = [r for r in logs if r["age_days"] <= retention_days]
    return outputs, logs
