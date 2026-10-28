#!/usr/bin/env python3
"""Audit case-export provenance signals without rewriting published observations.

Exit 0 for an informational report; --strict returns 2 if source-level
reproduction cannot be established from this archive. These warnings do not
imply that a dataset was fabricated.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "CPLF LLM protocols io complete package" / "CPLF LLM protocols io FORMAL RELEASE"


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def audit(root: Path = RELEASE) -> dict:
    public = root / "per_case_outputs"
    prediction = public / "prediction_outputs.csv"
    feedback = public / "feedback_outputs.csv"
    ablation = public / "ablation_outputs.csv"

    rubric_fields = ("diagnostic_consistency", "actionability", "personalization", "answer_leakage")
    off_grid = Counter()
    acceptance_disagreements = 0
    feedback_rows = 0
    feedback_case_ids = set()
    feedback_fields = set()
    for row in rows(feedback):
        feedback_rows += 1
        feedback_case_ids.add(row["case_id"])
        feedback_fields.update(row)
        for field in rubric_fields:
            scaled = float(row[field]) * 4
            if abs(scaled - round(scaled)) > 0.000004:
                off_grid[field] += 1
        expected = float(row["fqs"]) >= .70 and float(row["answer_leakage"]) <= .20
        if expected != bool(int(row["accepted_after_verification"])):
            acceptance_disagreements += 1

    prediction_rows = 0
    prediction_learners = set()
    prediction_datasets = Counter()
    for row in rows(prediction):
        prediction_rows += 1
        prediction_learners.add((row["dataset"], row["learner_id"]))
        prediction_datasets[row["dataset"]] += 1

    proxy_fields = set()
    ablation_rows = 0
    for row in rows(ablation):
        ablation_rows += 1
        proxy_fields.update(name for name in row if name.endswith("_proxy"))

    split_ids = set()
    for path in (root / "fixed_splits").glob("*_splits.csv"):
        for row in rows(path):
            split_ids.add((row["dataset"], row["learner_or_session_id"]))
    joined = len(prediction_learners & split_ids)

    findings = []
    if any(off_grid.values()):
        findings.append("Feedback rubric values are not direct one-shot 1-5 integer-evaluator outputs; transformation provenance required")
    if "feedback_final" not in feedback_fields:
        findings.append("Published feedback export omits actual generated feedback text and raw evaluator records")
    if joined == 0:
        findings.append("Published prediction IDs have no exact join to the anonymized fixed split IDs")
    if proxy_fields:
        findings.append("Ablation export uses proxy fields, not raw prediction observations for AUC/ECE reproduction")
    findings.append("Original training logs, checkpoints and third-party raw dataset transformations are not bundled")

    return {
        "assessment": "PUBLISHED_FILE_AUDIT_NOT_INDEPENDENT_RETRAINING",
        "prediction_rows": prediction_rows,
        "prediction_datasets": dict(sorted(prediction_datasets.items())),
        "unique_prediction_learners": len(prediction_learners),
        "prediction_to_split_exact_id_matches": joined,
        "feedback_rows": feedback_rows,
        "unique_feedback_case_ids": len(feedback_case_ids),
        "off_quarter_grid_feedback_fields": {key: off_grid[key] for key in rubric_fields},
        "acceptance_flag_threshold_disagreements": acceptance_disagreements,
        "ablation_rows": ablation_rows,
        "ablation_proxy_fields": sorted(proxy_fields),
        "provenance_findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit published case-level export limits")
    parser.add_argument("--strict", action="store_true", help="Exit 2 if provenance findings remain")
    args = parser.parse_args()
    report = audit()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 2 if args.strict and report["provenance_findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
