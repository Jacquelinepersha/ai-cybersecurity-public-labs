"""
Evaluate security answers by their claims, not their tone.

The ground truth is the official MITRE ATT&CK STIX bundle. Two simulated
answer generators stand in for "an LLM without retrieval" and "an LLM with
retrieval": they write fluent, confident text with controlled error rates.
The evaluator (claim extraction, verification, failure taxonomy, rubric) is
the real deliverable: it works on any model's output text.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .mitre_data import active_objects, external_id

TECHNIQUE_ID = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")


def build_truth(bundle):
    """Index the ATT&CK facts the evaluator checks against."""
    techniques, groups, retired = {}, {}, {}
    for obj in bundle.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        ext = external_id(obj)
        if not ext or not ext.startswith("T"):
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            retired[ext] = obj.get("name")
            continue
        techniques[ext] = {
            "stix_id": obj["id"],
            "name": obj.get("name"),
            "tactics": sorted({p["phase_name"] for p in obj.get("kill_chain_phases", [])}),
        }
    for obj in active_objects(bundle, "intrusion-set"):
        groups[obj["id"]] = obj.get("name")

    by_stix = {v["stix_id"]: k for k, v in techniques.items()}
    uses = {}
    for rel in active_objects(bundle, "relationship"):
        if rel.get("relationship_type") != "uses":
            continue
        if rel.get("source_ref") in groups and rel.get("target_ref") in by_stix:
            uses.setdefault(by_stix[rel["target_ref"]], set()).add(groups[rel["source_ref"]])

    tactic_names = sorted({t for v in techniques.values() for t in v["tactics"]})
    return {
        "techniques": techniques,
        "retired": retired,
        "group_names": sorted(set(groups.values())),
        "uses": uses,
        "tactics": tactic_names,
    }


# ---------------------------------------------------------------------------
# Simulated generators
# ---------------------------------------------------------------------------
CLOSED_BOOK_RATES = {
    "wrong_id": 0.12,       # a real ID, but for a different technique
    "fabricated_id": 0.08,  # an ID that does not exist
    "retired_id": 0.05,     # a revoked or deprecated ID
    "wrong_tactic": 0.15,
    "wrong_group": 0.20,    # a real group that is not linked to the technique
    "no_citation": 0.50,
    "abstain": 0.00,
}
GROUNDED_RATES = {
    "wrong_id": 0.02,
    "fabricated_id": 0.00,
    "retired_id": 0.00,
    "wrong_tactic": 0.03,
    "wrong_group": 0.03,
    "no_citation": 0.02,
    "abstain": 0.00,  # abstains automatically when the evidence is missing
}


def make_questions(truth, n=120, seed=0):
    rng = np.random.default_rng(seed)
    with_groups = [t for t in truth["techniques"] if truth["uses"].get(t)]
    without_groups = [t for t in truth["techniques"] if not truth["uses"].get(t)]
    picks = list(rng.choice(with_groups, size=int(n * 0.8), replace=False))
    picks += list(rng.choice(without_groups, size=n - len(picks), replace=False))
    return pd.DataFrame({
        "question_id": range(len(picks)),
        "technique_id": picks,
        "question": [
            f"Which ATT&CK technique is '{truth['techniques'][t]['name']}'? Give its ID, tactic, and one group known to use it."
            for t in picks
        ],
    })


def generate_answer(truth, technique_id, rates, grounded, rng):
    """Write a fluent, confident answer with controlled errors."""
    tech = truth["techniques"][technique_id]
    all_ids = sorted(truth["techniques"])
    groups_using = sorted(truth["uses"].get(technique_id, set()))

    answer_id = technique_id
    if rng.random() < rates["fabricated_id"]:
        answer_id = f"T{rng.integers(1800, 1999)}.{rng.integers(1, 9):03d}"
    elif truth["retired"] and rng.random() < rates["retired_id"]:
        answer_id = str(rng.choice(sorted(truth["retired"])))
    elif rng.random() < rates["wrong_id"]:
        answer_id = str(rng.choice(all_ids))

    tactic = tech["tactics"][0] if tech["tactics"] else "unknown"
    if rng.random() < rates["wrong_tactic"]:
        tactic = str(rng.choice([t for t in truth["tactics"] if t not in tech["tactics"]]))

    if grounded and not groups_using:
        group_sentence = "Insufficient data: no group is linked to this technique in the retrieved ATT&CK data."
    else:
        if groups_using and rng.random() >= rates["wrong_group"]:
            group = str(rng.choice(groups_using))
        else:
            group = str(rng.choice([g for g in truth["group_names"] if g not in groups_using]))
        group_sentence = f"It is used by {group}."

    citation = "" if rng.random() < rates["no_citation"] else f" Source: MITRE ATT&CK {technique_id if grounded else answer_id}."
    return (
        f"{tech['name']} is ATT&CK technique {answer_id}, under the {tactic} tactic. "
        f"{group_sentence}{citation}"
    )


def run_generator(truth, questions, system, seed=0):
    rates = GROUNDED_RATES if system == "retrieval-grounded" else CLOSED_BOOK_RATES
    rng = np.random.default_rng(seed)
    return questions.assign(
        system=system,
        answer=[
            generate_answer(truth, t, rates, system == "retrieval-grounded", rng)
            for t in questions["technique_id"]
        ],
    )


# ---------------------------------------------------------------------------
# Evaluator: decompose into claims, verify each, classify failures
# ---------------------------------------------------------------------------
def extract_claims(answer, truth):
    """Break free text into checkable claims using ATT&CK vocabularies."""
    claims = []
    ids = TECHNIQUE_ID.findall(answer.split("Source:")[0])
    if ids:
        claims.append({"type": "technique_id", "value": ids[0]})
    for tactic in truth["tactics"]:
        if re.search(rf"\b{re.escape(tactic)}\b", answer):
            claims.append({"type": "tactic", "value": tactic})
    match = re.search(r"used by (.+?)\.(?:\s|$)", answer)
    if match:
        claims.append({"type": "group", "value": match.group(1).strip()})
    if "Insufficient data" in answer:
        claims.append({"type": "abstention", "value": "group"})
    source = re.search(r"Source: MITRE ATT&CK (T\d{4}(?:\.\d{3})?)", answer)
    claims.append({"type": "citation", "value": source.group(1) if source else None})
    return claims


def verify_claim(claim, technique_id, truth):
    """Return (supported, failure_category)."""
    kind, value = claim["type"], claim["value"]
    tech = truth["techniques"][technique_id]
    if kind == "technique_id":
        if value == technique_id:
            return True, None
        if value in truth["retired"]:
            return False, "outdated (retired ID)"
        if value not in truth["techniques"]:
            return False, "fabricated entity"
        return False, "wrong attribute (real ID, wrong technique)"
    if kind == "tactic":
        return (True, None) if value in tech["tactics"] else (False, "wrong attribute (tactic)")
    if kind == "group":
        # ATT&CK records documented links only. A group that is not linked may still
        # use the technique, so this is "unsupported", not proven false. Either way it
        # is unverified and must not drive an attribution decision.
        if value not in truth["group_names"]:
            return False, "fabricated entity"
        return (True, None) if value in truth["uses"].get(technique_id, set()) else (False, "group not supported by ATT&CK")
    if kind == "abstention":
        has_groups = bool(truth["uses"].get(technique_id))
        return (False, "unnecessary abstention") if has_groups else (True, None)
    if kind == "citation":
        if value is None:
            return False, "unsupported (no citation)"
        return (True, None) if value == technique_id else (False, "citation does not support claim")
    return False, "unknown claim type"


def evaluate_answers(answers, truth):
    rows = []
    for item in answers.itertuples(index=False):
        for claim in extract_claims(item.answer, truth):
            supported, failure = verify_claim(claim, item.technique_id, truth)
            rows.append({
                "system": item.system,
                "question_id": item.question_id,
                "claim_type": claim["type"],
                "claim_value": claim["value"],
                "supported": supported,
                "failure": failure,
            })
    return pd.DataFrame(rows)


CRITICAL_FAILURES = {"fabricated entity", "group not supported by ATT&CK", "outdated (retired ID)"}


def rubric(claims):
    """
    Multi-dimensional score per system. One number hides which failure you
    have; the rubric keeps them apart.
    """
    def per_system(g):
        factual = g[g["claim_type"].isin(["technique_id", "tactic", "group"])]
        answers = g.groupby("question_id")
        critical = answers.apply(lambda a: a["failure"].isin(CRITICAL_FAILURES).any(), include_groups=False)
        return pd.Series({
            "claim_precision": factual["supported"].mean(),
            "id_accuracy": g.loc[g["claim_type"] == "technique_id", "supported"].mean(),
            "tactic_accuracy": g.loc[g["claim_type"] == "tactic", "supported"].mean(),
            "group_accuracy": g.loc[g["claim_type"] == "group", "supported"].mean(),
            "citation_validity": g.loc[g["claim_type"] == "citation", "supported"].mean(),
            "answers_with_critical_error": critical.mean(),
            "correct_abstentions": int((g["claim_type"] == "abstention").sum()),
        })

    return claims.groupby("system").apply(per_system, include_groups=False).T
