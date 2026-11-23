"""
Measure outcomes, not motion.

A useful metric has a definition, a baseline, a threshold, an owner, and a
decision attached to it. This module keeps that structure explicit so a
scorecard can say what to do next, not only what happened.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd


@dataclass
class Metric:
    name: str
    layer: str               # detection, robustness, LLM application, agent, privacy, operations
    definition: str
    value: float
    baseline: float
    threshold: float
    better: str              # "higher" or "lower"
    owner: str
    decision_if_breached: str

    def status(self):
        breached = self.value < self.threshold if self.better == "higher" else self.value > self.threshold
        if breached:
            return "BREACHED"
        improved = self.value > self.baseline if self.better == "higher" else self.value < self.baseline
        same = abs(self.value - self.baseline) < 1e-9
        return "on target" if same else ("improved" if improved else "regressed, within threshold")


def scorecard(metrics):
    rows = []
    for m in metrics:
        row = asdict(m)
        row["status"] = m.status()
        rows.append(row)
    frame = pd.DataFrame(rows)
    order = {"BREACHED": 0, "regressed, within threshold": 1, "on target": 2, "improved": 3}
    return frame.sort_values("status", key=lambda s: s.map(order)).reset_index(drop=True)


def motion_metrics_example():
    """Numbers that describe activity but cannot tell you if risk went down."""
    return pd.DataFrame([
        {"motion metric": "alerts generated this week", "why it misleads": "more alerts can mean more attacks, more noise, or a broken sensor"},
        {"motion metric": "red-team tests executed", "why it misleads": "running tests is not the same as fixed failures staying fixed"},
        {"motion metric": "prompts blocked by the filter", "why it misleads": "rises with false positives too; says nothing about what got through"},
        {"motion metric": "models retrained", "why it misleads": "retraining on bad data makes things worse"},
        {"motion metric": "dashboards published", "why it misleads": "no decision is attached to any number"},
    ])
