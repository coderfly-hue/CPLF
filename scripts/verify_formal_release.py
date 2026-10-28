"""Verify release files with no heavy dependencies.

This script does not falsely equate structural consistency with independent
scientific replication or access to unbundled training checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "CPLF LLM protocols io complete package" / "CPLF LLM protocols io FORMAL RELEASE"


SPLITS = {"OULAD": (20124, 2875, 5750), "ASSIST2009": (2738, 391, 783),
          "EdNet": (35000, 5000, 10000), "MathDial": (2003, 286, 572),
          "CodeWorkout": (907, 130, 259), "CodeStream": (1472, 210, 422)}
CASE_FILES = {"prediction_outputs.csv": ("case_id", "CASE_"),
              "feedback_outputs.csv": ("case_id", "FEEDBACK_"),
              "ablation_outputs.csv": ("case_id", "ABLATION_")}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def verify(root: Path = REPO) -> tuple[list[str], list[str], dict]:
    errors, notices = [], []
    source = root / "CPLF LLM protocols io complete package" / "CPLF LLM protocols io FORMAL RELEASE"
    counts = {}
    released_manifest = list(read_csv(source / "MANIFEST_SHA256.csv"))
    if len(released_manifest) != len({r["relative_path"] for r in released_manifest}):
        errors.append("Duplicate paths in published file manifest")
    for row in released_manifest:
        rel = Path(row["relative_path"])
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"Unsafe manifest path: {rel}")
            continue
        if not (source / rel).is_file():
            errors.append(f"Missing published file: {rel}")
            continue
        payload = (source / rel).read_bytes()
        if len(payload) != int(row["bytes"]) or hashlib.sha256(payload).hexdigest() != row["sha256"]:
            errors.append(f"Published file SHA or size mismatch: {rel}")
    counts["manifest_files"] = len(released_manifest)

    for dataset, expected in SPLITS.items():
        path = source / "fixed_splits" / f"{dataset}_splits.csv"
        label_counts = Counter()
        seen = set()
        for row in read_csv(path):
            ident = row["learner_or_session_id"]
            if not ident.startswith("ANON_"):
                errors.append(f"Non-public identifier in {path.name}")
                break
            if ident in seen:
                errors.append(f"Duplicate split ID in {path.name}")
            seen.add(ident)
            label_counts[row["split"]] += 1
            if row["dataset"] != dataset:
                errors.append(f"Wrong dataset value in {path.name}")
        observed = tuple(label_counts.get(x, 0) for x in ("train", "validation", "test"))
        if observed != expected or len(seen) != sum(expected):
            errors.append(f"Split differs from manuscript Table 1: {dataset}: {observed}")
    counts["split_ids"] = sum(sum(values) for values in SPLITS.values())

    for filename, (column, prefix) in CASE_FILES.items():
        count = 0
        for row in read_csv(source / "per_case_outputs" / filename):
            count += 1
            if not row[column].startswith(prefix):
                errors.append(f"Unexpected case ID in {filename} at row {count}")
                break
            if filename == "prediction_outputs.csv":
                if not row["learner_id"].startswith("ANONLEARNER_"):
                    errors.append(f"Unexpected learner ID in {filename} at row {count}")
                    break
            if filename == "feedback_outputs.csv":
                try:
                    expected_fqs = max(0.0, .25 * sum(float(row[key]) for key in (
                        "relevance", "diagnostic_consistency", "actionability", "personalization"))
                        - .4 * float(row["answer_leakage"]))
                    if abs(expected_fqs - float(row["fqs"])) > 2e-6:
                        errors.append(f"FQS arithmetic mismatch in {filename} row {count}")
                        break
                except (KeyError, ValueError):
                    errors.append(f"Invalid FQS fields in {filename} row {count}")
                    break
        counts[filename] = count
    if [counts.get(x) for x in CASE_FILES] != [90000, 45000, 15000]:
        errors.append("Unexpected per-case record counts")

    result_groups = defaultdict(list)
    for row in read_csv(source / "release_data/seed_level_results.csv"):
        key = (row["analysis"], row["method"], row["dataset"], row["metric"])
        result_groups[key].append((row["seed"], float(row["value"])))
    summaries = {}
    for row in read_csv(source / "release_data/seed_level_summary.csv"):
        key = (row["analysis"], row["method"], row["dataset"], row["metric"])
        summaries[key] = row
    for key, values in result_groups.items():
        seeds = {seed for seed, _ in values}
        if seeds != {"2024", "2025", "2026", "2027", "2028"} or len(values) != 5:
            errors.append(f"Invalid five-seed run group: {key}")
            continue
        summary = summaries.get(key)
        if summary is None:
            errors.append(f"Missing five-seed summary: {key}")
            continue
        xs = [val for _, val in values]
        if (abs(statistics.mean(xs) - float(summary["mean"])) > .0006 or
                abs(statistics.stdev(xs) - float(summary["sd"])) > .0006 or
                int(summary["n_seeds"]) != 5):
            errors.append(f"Five-seed aggregate mismatch: {key}")
    if set(summaries) != set(result_groups):
        errors.append("Seed summary group coverage differs from seed-level results")
    counts["seed_groups"] = len(result_groups)

    expected_auc = {"OULAD": .891, "ASSIST2009": .866, "EdNet": .823}
    for dataset, expected in expected_auc.items():
        key = ("overall_prediction", "CPLF-LLM", dataset, "AUC")
        if key not in summaries or abs(float(summaries[key]["mean"]) - expected) > 1e-9:
            errors.append(f"Manuscript headline AUC mismatch: {dataset}")
    feedback_summary = {r["method"]: r for r in read_csv(source / "reported_results/feedback_generation_summary.csv")}
    for method, expected in (("Direct LLM", .700), ("CPLF-LLM", .840)):
        if method not in feedback_summary or abs(float(feedback_summary[method]["fqs_mean"]) - expected) > 1e-9:
            errors.append(f"Manuscript headline FQS mismatch: {method}")

    notices.append("Offline file checks do not repeat model training")
    notices.append("Published prediction learner ID namespace does not directly join to fixed split IDs")
    notices.append("Selected per-case prediction AUCs may differ from five-seed summary AUCs; see docs/REPRODUCIBILITY_SCOPE.md")
    return errors, notices, counts


def main() -> int:
    p = argparse.ArgumentParser(description="Validate CPLF-LLM computational GitHub release")
    p.add_argument("--mode", choices=("author", "independent"), default="author")
    a = p.parse_args()
    errors, notes, counts = verify()
    for error in errors:
        print("ERROR:", error)
    for note in notes:
        print("NOTE:", note)
    print("Checked:", counts)
    if a.mode == "independent":
        print("INDEPENDENT REPRODUCTION: NOT ASSESSED (unbundled raw training evidence)")
        return 2
    if errors:
        print("FORMAL RELEASE FILES: FAIL")
        return 1
    print("FORMAL RELEASE FILES: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
