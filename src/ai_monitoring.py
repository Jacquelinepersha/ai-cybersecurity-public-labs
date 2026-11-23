"""
Operating a deployed detector: monitoring, incident signals, rollback.

A simulated 30-day deployment replays the UNSW-NB15 test set in daily
batches. Two incidents are injected on purpose:
- a poisoned model update that creates a blind spot for one attack category;
- a telemetry pipeline change that rescales byte counts.
The lab detects them from monitoring signals, contains them, and measures
recovery. Everything is offline and deterministic.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def daily_batches(X, y, categories, days=30, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    splits = np.array_split(order, days)
    return [
        (X.iloc[idx].reset_index(drop=True),
         pd.Series(np.asarray(y)[idx]),
         pd.Series(np.asarray(categories)[idx]))
        for idx in splits
    ]


def psi(reference, current, bins=10):
    """Population stability index between two score samples."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def fingerprint(obj):
    return hashlib.sha256(repr(obj).encode()).hexdigest()[:12]


@dataclass
class ModelRegistry:
    """Known-good versions with a fingerprint, so rollback is one decision."""
    versions: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def register(self, name, model, training_fingerprint, note=""):
        self.versions[name] = {"model": model, "training_data": training_fingerprint, "note": note}

    def deploy(self, name, day, reason):
        self.history.append({"day": day, "deployed": name, "reason": reason})
        return self.versions[name]["model"]


def scale_bytes(X, factor=1000.0):
    """Simulate a telemetry change: byte counters suddenly reported in a different unit."""
    out = X.copy()
    for column in ["sbytes", "dbytes", "sload", "dload"]:
        if column in out.columns:
            out[column] = out[column] * factor
    return out


def monitor_day(model, X, y, categories, reference_scores, watch_category="Fuzzers", feedback_share=0.05, seed=0):
    """
    What operators can see each day. Labels are only known for a small
    analyst-reviewed sample (delayed feedback), not for every flow.
    """
    rng = np.random.default_rng(seed)
    scores = model.predict_proba(X)[:, 1]
    alerts = scores >= 0.5
    y = np.asarray(y)
    categories = np.asarray(categories)
    reviewed = rng.random(len(X)) < feedback_share
    reviewed_attacks = reviewed & (y == 1)
    reviewed_alerts = reviewed & alerts
    watched = categories == watch_category
    return {
        "alert_rate": float(alerts.mean()),
        "score_psi": psi(reference_scores, scores),
        "feedback_recall": float(alerts[reviewed_attacks].mean()) if reviewed_attacks.any() else np.nan,
        "feedback_precision": float((y[reviewed_alerts] == 1).mean()) if reviewed_alerts.any() else np.nan,
        # ground truth, hidden from operators, used only to grade the response
        "true_recall": float(alerts[y == 1].mean()),
        "true_precision": float((y[alerts] == 1).mean()) if alerts.any() else np.nan,
        f"true_{watch_category.lower()}_recall": float(alerts[watched].mean()) if watched.any() else np.nan,
    }


def baseline_limits(frame, columns, k=3.0):
    """Alert limits from the baseline window: mean +/- k standard deviations."""
    limits = {}
    for column in columns:
        mean, std = frame[column].mean(), frame[column].std(ddof=1)
        limits[column] = (mean - k * std, mean + k * std)
    return limits


def signals(frame, limits):
    """Mark which monitored values left their baseline band."""
    out = frame.copy()
    for column, (low, high) in limits.items():
        out[f"{column}_signal"] = (out[column] < low) | (out[column] > high)
    out["any_signal"] = out[[c for c in out.columns if c.endswith("_signal")]].any(axis=1)
    return out
