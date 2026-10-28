#!/usr/bin/env python3
"""Rebuild manuscript-aligned numerical tables from released evidence files.

This is a checkable published-artifact reconstruction, not a substitution for
the missing original training and human-evaluation logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "CPLF LLM protocols io complete package" / "CPLF LLM protocols io FORMAL RELEASE"
RESULTS = FORMAL / "reported_results"
SEED_ROWS = FORMAL / "release_data" / "seed_level_results.csv"
SEED_SUMMARY = FORMAL / "release_data" / "seed_level_summary.csv"
PLOT_VALUES = ROOT / "paper" / "plot_values"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regenerate_seed_summary() -> pd.DataFrame:
    records = pd.read_csv(SEED_ROWS)
    required = ["analysis", "method", "dataset", "metric", "seed", "value"]
    if records[required].isna().any().any():
        raise ValueError("Seed-level records contain missing source fields")
    groups = ["analysis", "method", "dataset", "metric"]
    counts = records.groupby(groups, dropna=False)["seed"].nunique()
    if not counts.eq(5).all() or not set(records.seed.unique()) == {2024, 2025, 2026, 2027, 2028}:
        raise ValueError("Expected exactly five distinct reported seeds in every group")
    calculated = records.groupby(groups, as_index=False)["value"].agg(
        mean="mean", sd="std", n_seeds="count")
    published = pd.read_csv(SEED_SUMMARY)
    check = calculated.merge(published, on=groups, validate="one_to_one", suffixes=("_calc", "_pub"), how="outer", indicator=True)
    if not check._merge.eq("both").all():
        raise ValueError("Stored summary and seed-level group coverage differ")
    for column in ("mean", "sd"):
        if (check[f"{column}_calc"] - check[f"{column}_pub"]).abs().max() > .0006:
            raise ValueError(f"Five-seed {column} differs from published values")
    return calculated


def splits_table() -> pd.DataFrame:
    outputs = []
    for file in sorted((FORMAL / "fixed_splits").glob("*_splits.csv")):
        frame = pd.read_csv(file)
        required = {"dataset", "learner_or_session_id", "split"}
        if not required <= set(frame):
            raise ValueError(f"Missing split columns in {file.name}")
        if frame.learner_or_session_id.duplicated().any():
            raise ValueError(f"Duplicate split IDs in {file.name}")
        counts = frame.split.value_counts()
        outputs.append({"dataset": file.stem.removesuffix("_splits"),
                        "train": int(counts.get("train", 0)),
                        "validation": int(counts.get("validation", 0)),
                        "test": int(counts.get("test", 0)),
                        "total": len(frame)})
    if len(outputs) != 6:
        raise ValueError("Expected six published fixed split tables")
    return pd.DataFrame(outputs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/reproduced_tables")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    generated: dict[str, tuple[pd.DataFrame, list[Path], str]] = {}
    generated["table1_split_counts.csv"] = (splits_table(), sorted((FORMAL / "fixed_splits").glob("*_splits.csv")),
                                             "recount of archived anonymized split files")
    recalculated = regenerate_seed_summary()
    generated["five_seed_summary_recomputed.csv"] = (recalculated, [SEED_ROWS, SEED_SUMMARY],
                                                     "arithmetic reaggregation of published five-seed rows")
    generated["table2_overall_prediction.csv"] = (
        recalculated[recalculated.analysis == "overall_prediction"], [SEED_ROWS, SEED_SUMMARY],
        "published-seed aggregation, not primary training replay")
    generated["table4_dynamic_prediction.csv"] = (
        recalculated[recalculated.analysis == "dynamic_prediction"], [SEED_ROWS, SEED_SUMMARY],
        "published-seed aggregation, not primary training replay")
    published_tables = {
        "table6_feedback_generation.csv": "feedback_generation_summary.csv",
        "table7_feedback_dataset_level.csv": "feedback_dataset_level_summary.csv",
        "table8_human_validation.csv": "human_validation_summary.csv",
        "table12_principal_comparisons.csv": "principal_statistical_comparisons.csv",
        "table13_ablation.csv": "ablation_summary.csv",
    }
    for name, source_name in published_tables.items():
        source = RESULTS / source_name
        generated[name] = (pd.read_csv(source), [source],
                           "archived paper-reported table values; original paired/evaluator records not replayed")
    for name, source_name in (
        ("sequence_length_table.csv", "sequence_length_table_transcription.csv"),
        ("feedback_error_distribution.csv", "feedback_error_table_transcription.csv"),
    ):
        source = PLOT_VALUES / source_name
        generated[name] = (pd.read_csv(source), [source],
                           "transcribed manuscript table means; not original seed/case records")

    report = {"scope": "published numerical table reproduction; not independent retraining", "tables": {}}
    for name, (frame, sources, evidence) in generated.items():
        target = output / name
        frame.to_csv(target, index=False)
        report["tables"][name] = {
            "evidence_class": evidence,
            "rows": len(frame),
            "sources": [{"path": str(src.relative_to(ROOT)), "sha256": sha(src)} for src in sources],
            "output_sha256": sha(target),
        }
        print(f"Generated {name}: {len(frame)} rows")
    (output / "table_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
