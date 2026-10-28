#!/usr/bin/env python3
"""Check deposited canonical evidence; never turn absent evidence into a pass."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = {2024, 2025, 2026, 2027, 2028}
RAW_COLUMNS = ("diagnostic_raw", "actionability_raw", "personalization_raw", "leakage_raw")
NORMAL_COLUMNS = ("diagnostic_consistency", "actionability", "personalization", "answer_leakage")
JSON_KEYS = ("diagnostic", "actionability", "personalization", "leakage")


def check_predictions(path: Path, method_col: str, failures: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"dataset", "case_id", method_col, "seed", "y_true", "y_prob"}
    if method_col == "method":
        required.add("learner_id")
    missing = required - set(frame.columns)
    if missing:
        failures.append(f"{path.name}: missing {sorted(missing)}")
        return frame
    if frame.empty or frame[list(required)].isna().any().any():
        failures.append(f"{path.name}: empty or null required observations")
    if not set(frame["seed"].dropna().unique()).issubset(SEEDS):
        failures.append(f"{path.name}: unexpected seed")
    if not frame["y_true"].isin([0, 1]).all():
        failures.append(f"{path.name}: y_true not binary")
    if not frame["y_prob"].between(0, 1).all():
        failures.append(f"{path.name}: y_prob not in [0,1]")
    return frame


def check_feedback(path: Path, failures: list[str]) -> int:
    frame = pd.read_csv(path)
    required = {"dataset", "case_id", "method", "seed", "relevance", "fqs",
                "feedback_final", "evaluator_raw_response", *RAW_COLUMNS, *NORMAL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        failures.append(f"canonical feedback missing {sorted(missing)}")
        return len(frame)
    if frame.empty or frame[list(required)].isna().any().any():
        failures.append("canonical feedback has empty or null required fields")
        return len(frame)
    if not frame["feedback_final"].astype(str).str.strip().all():
        failures.append("canonical feedback text is empty")
    for raw_col, norm_col in zip(RAW_COLUMNS, NORMAL_COLUMNS):
        if not frame[raw_col].isin([1, 2, 3, 4, 5]).all():
            failures.append(f"{raw_col} is not integer 1..5")
        expected = (frame[raw_col] - 1) / 4.0
        if not np.allclose(frame[norm_col], expected, atol=1e-9):
            failures.append(f"{norm_col} disagrees with raw integers")
    expected_fqs = np.maximum(0.0, .25 * (frame["relevance"] + frame["diagnostic_consistency"]
        + frame["actionability"] + frame["personalization"]) - .4 * frame["answer_leakage"])
    if not np.allclose(frame["fqs"], expected_fqs, atol=1e-6):
        failures.append("canonical feedback FQS mismatch")
    for idx, (raw_text, raw_values) in enumerate(zip(frame["evaluator_raw_response"],
                                                     frame[list(RAW_COLUMNS)].itertuples(index=False, name=None)), 1):
        try:
            try:
                payload = json.loads(raw_text)
            except ValueError:
                embedded = re.search(r"\{.*\}", raw_text, flags=re.S)
                if not embedded:
                    raise
                payload = json.loads(embedded.group(0))
            if any(int(payload[key]) != int(raw_values[i]) for i, key in enumerate(JSON_KEYS)):
                raise ValueError("raw evaluator JSON differs from recorded scores")
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(f"canonical feedback row {idx}: {exc}")
            break
    return len(frame)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    canonical = root / "artifacts" / "canonical"
    findings: list[str] = []
    failures: list[str] = []
    missing: list[str] = []

    feedback = canonical / "feedback_outputs.csv"
    if feedback.is_file():
        count = check_feedback(feedback, failures)
        findings.append(f"canonical feedback: {count:,} rows")
    else:
        missing.append("feedback_outputs.csv")

    for filename, label, method_col in (
        ("prediction_outputs.csv", "prediction", "method"),
        ("ablation_prediction_outputs.csv", "ablation predictions", "variant"),
    ):
        path = canonical / filename
        if path.is_file():
            frame = pd.read_csv(path)
            if any("proxy" in column.lower() for column in frame.columns):
                failures.append(f"{filename}: legacy proxy column not allowed")
            check_predictions(path, method_col, failures)
            findings.append(f"canonical {label}: {len(frame):,} rows")
        else:
            missing.append(filename)

    for name in missing:
        print(f"[MISSING] canonical/{name}: not archived")
    prediction = canonical / "prediction_outputs.csv"
    splits = canonical / "split_assignments.csv"
    if prediction.is_file():
        if not splits.is_file():
            failures.append("canonical predictions lack linked split_assignments.csv")
        else:
            pred = pd.read_csv(prediction)
            split_frame = pd.read_csv(splits)
            needed_pred = {"dataset", "learner_id"}
            needed_split = {"dataset", "learner_or_session_id", "split"}
            if needed_pred <= set(pred) and needed_split <= set(split_frame):
                if not pred.learner_id.astype(str).str.startswith("CID_").all():
                    failures.append("canonical predictions use noncanonical learner IDs")
                if split_frame.duplicated(["dataset", "learner_or_session_id"]).any():
                    failures.append("canonical split assignments have duplicate IDs")
                test_ids = set(zip(split_frame.loc[split_frame.split == "test", "dataset"],
                                   split_frame.loc[split_frame.split == "test", "learner_or_session_id"]))
                observed_ids = set(zip(pred.dataset, pred.learner_id))
                if not observed_ids.issubset(test_ids):
                    failures.append(f"canonical prediction IDs not in test split: {len(observed_ids - test_ids)}")
            else:
                failures.append("canonical predictions/split assignments lack ID columns")
    for item in findings:
        print(f"[INFO] {item}")
    for issue in failures:
        print(f"[FAIL] {issue}")
    if failures:
        return 1
    if args.strict and missing:
        print("[FAIL] strict evidence check requires the missing original canonical exports")
        return 1
    if missing:
        print("[PASS] available-file structure only; primary evidence is incomplete")
    else:
        print("[PASS] deposited canonical file fields checked; not independent retraining")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
