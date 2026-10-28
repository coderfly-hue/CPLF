#!/usr/bin/env python3
"""Strict, traceable adapters from six dataset source forms to CPLF input.

Source datasets are NOT bundled. Configuration decisions that the manuscript
does not pin (notably OULAD's score-to-binary threshold and the schemas of the
three feedback sources) must be supplied explicitly. This script never fills
missing outcome labels with artificial zeros.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

REQUIRED = ("student_id", "exercise_id", "concepts", "correct", "timestamp")
FEEDBACK = ("question_text", "response_text", "reference_feedback")
FEEDBACK_DATASETS = ("MathDial", "CodeWorkout", "CodeStream")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_source(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".jsonl":
        return pd.json_normalize(pd.read_json(path, lines=True).to_dict(orient="records"))
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            obj = obj.get("records", obj.get("data", obj))
        if not isinstance(obj, list):
            raise ValueError("JSON source must contain a record list or records/data list")
        return pd.json_normalize(obj)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported source format: {path}")


def require(frame: pd.DataFrame, fields, label: str) -> None:
    missing = sorted(set(fields) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def make_oulad(args) -> tuple[pd.DataFrame, list[Path], dict]:
    source = Path(args.student_assessment)
    meta = Path(args.assessments)
    events = read_source(source)
    assessments = read_source(meta)
    require(events, ["id_student", "id_assessment", "score", "date_submitted"], "OULAD studentAssessment")
    require(assessments, ["id_assessment", "assessment_type", "code_module", "code_presentation"], "OULAD assessments")
    if assessments.id_assessment.duplicated().any():
        raise ValueError("OULAD assessments contains duplicate id_assessment")
    joined = events.merge(assessments, on="id_assessment", how="left", validate="many_to_one")
    if joined[["assessment_type", "code_module", "code_presentation"]].isna().any().any():
        raise ValueError("OULAD assessment metadata join is incomplete")
    score = pd.to_numeric(joined.score, errors="coerce")
    if score.isna().any() or not score.between(0, 100).all():
        raise ValueError("OULAD assessment score missing or outside [0,100]")
    frame = pd.DataFrame({
        "student_id": joined.id_student.astype(str),
        "exercise_id": (joined.code_module.astype(str) + ":" +
                        joined.code_presentation.astype(str) + ":" + joined.id_assessment.astype(str)),
        "concepts": joined.assessment_type.astype(str),
        "correct": (score >= args.score_threshold).astype(int),
        "timestamp": pd.to_numeric(joined.date_submitted, errors="coerce"),
    })
    return frame, [source, meta], {"score_threshold": args.score_threshold,
                                    "timestamp_semantics": "relative day of submission, not absolute Unix time"}


def make_assist(args) -> tuple[pd.DataFrame, list[Path], dict]:
    path = Path(args.input)
    frame = read_source(path)
    require(frame, ["user_id", "problem_id", "skill_id", "correct", "order_id"], "ASSIST2009 Skill Builder")
    if frame[["user_id", "problem_id", "skill_id", "correct", "order_id"]].isna().any().any():
        raise ValueError("ASSIST2009 source contains missing identifiers, concept tags, responses or sequence order")
    out = pd.DataFrame({
        "student_id": frame.user_id.astype(str),
        "exercise_id": frame.problem_id.astype(str),
        "concepts": frame.skill_id.astype(str),
        "correct": frame.correct,
        "timestamp": frame[args.time_column] if args.time_column else frame.order_id,
    })
    return out, [path], {"timestamp_semantics": args.time_column or "within-source order_id, not wall-clock time"}


def make_ednet(args) -> tuple[pd.DataFrame, list[Path], dict]:
    question_path = Path(args.questions)
    questions = read_source(question_path)
    require(questions, ["question_id", "correct_answer", "tags"], "EdNet questions")
    if questions.question_id.duplicated().any():
        raise ValueError("EdNet question metadata contains duplicate question_id")
    folder = Path(args.events_dir)
    event_files = sorted(folder.glob("*.csv"))
    if not event_files:
        raise ValueError("EdNet events directory contains no per-user CSV files")
    pieces = []
    for path in event_files:
        events = read_source(path)
        require(events, ["question_id", "user_answer", "timestamp"], str(path))
        if events[["question_id", "user_answer", "timestamp"]].isna().any().any():
            raise ValueError(f"EdNet source {path} contains incomplete question/response/timestamp rows")
        if "student_id" not in events:
            events["student_id"] = path.stem
        pieces.append(events[["student_id", "question_id", "user_answer", "timestamp"]])
    source = pd.concat(pieces, ignore_index=True)
    joined = source.merge(questions[["question_id", "correct_answer", "tags"]],
                          on="question_id", how="left", validate="many_to_one")
    if joined[["correct_answer", "tags"]].isna().any().any():
        raise ValueError("EdNet question metadata join is incomplete")
    out = pd.DataFrame({
        "student_id": joined.student_id.astype(str),
        "exercise_id": joined.question_id.astype(str),
        "concepts": joined.tags.astype(str),
        "correct": (joined.user_answer.astype(str).str.strip() ==
                    joined.correct_answer.astype(str).str.strip()).astype(int),
        "timestamp": joined.timestamp,
    })
    return out, [question_path, *event_files], {"timestamp_semantics": "source EdNet timestamp",
                                                   "question_join": "questions.question_id"}


def make_feedback(args) -> tuple[pd.DataFrame, list[Path], dict]:
    source, config = Path(args.input), Path(args.field_map)
    frame = read_source(source)
    mapping = json.loads(config.read_text(encoding="utf-8"))
    required = [*REQUIRED, *FEEDBACK]
    if set(mapping) != set(required):
        raise ValueError(f"{args.dataset}: field-map must define exactly {required}; provide actual source field names")
    require(frame, mapping.values(), args.dataset)
    out = pd.DataFrame({target: frame[src] for target, src in mapping.items()})
    return out, [source, config], {"field_mapping": mapping,
                                   "mapping_source": "author-supplied actual dataset field names"}


def validate_and_export(frame: pd.DataFrame, dataset: str, output: Path,
                        sources: list[Path], decisions: dict, hash_ids_env: str | None) -> None:
    require(frame, REQUIRED, dataset)
    if frame.empty or frame[list(REQUIRED)].isna().any().any():
        raise ValueError(f"{dataset}: empty source or missing mandatory interaction values")
    out = frame.copy()
    outcome = pd.to_numeric(out.correct, errors="coerce")
    if outcome.isna().any() or not outcome.isin([0, 1]).all():
        raise ValueError(f"{dataset}: correct must be observed 0/1; no labels are inferred")
    time = pd.to_numeric(out.timestamp, errors="coerce")
    if time.isna().any():
        raise ValueError(f"{dataset}: timestamp/order cannot be missing or nonnumeric")
    if not (time == time.round()).all():
        raise ValueError(f"{dataset}: timestamps/order must use unambiguous integer units")
    if not out.student_id.astype(str).str.strip().all() or not out.exercise_id.astype(str).str.strip().all():
        raise ValueError(f"{dataset}: student/exercise identifiers cannot be blank")
    if not out.concepts.astype(str).str.strip().all():
        raise ValueError(f"{dataset}: concept tags cannot be blank")
    if dataset in FEEDBACK_DATASETS:
        for field in FEEDBACK:
            if out[field].isna().any() or not out[field].astype(str).str.strip().all():
                raise ValueError(f"{dataset}: missing {field}; raw feedback cases cannot be synthesized")
    if hash_ids_env:
        secret = os.environ.get(hash_ids_env)
        if not secret:
            raise ValueError(f"Missing configured ID pseudonymization secret in {hash_ids_env}")
        import hmac
        out.student_id = ["CID_" + hmac.new(secret.encode(), (dataset + "\0" + str(x)).encode(), "sha256").hexdigest()[:20].upper()
                          for x in out.student_id]
        decisions["id_handling"] = "keyed deterministic HMAC pseudonyms; secret not recorded"
    else:
        decisions["id_handling"] = "source identifiers retained locally; review privacy before sharing"
    out.correct = outcome.astype(int)
    out.timestamp = time.astype("int64")
    out = out.sort_values(["student_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    manifest = {
        "dataset": dataset,
        "status": "new preprocessing from explicitly supplied source files; not historical run provenance",
        "source_files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)} for path in sources],
        "decisions": decisions,
        "rows": len(out), "learners": int(out.student_id.nunique()),
        "output": {"path": str(output), "sha256": sha(output)},
    }
    manifest_path = output.with_suffix(output.suffix + ".provenance.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {dataset}: {len(out):,} rows, {out.student_id.nunique():,} learners; {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="dataset", required=True)
    for dataset in ("OULAD", "ASSIST2009", "EdNet", *FEEDBACK_DATASETS):
        cmd = sub.add_parser(dataset)
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--hash-ids-env", help="Environment variable holding a private HMAC key")
        if dataset == "OULAD":
            cmd.add_argument("--student-assessment", required=True)
            cmd.add_argument("--assessments", required=True)
            cmd.add_argument("--score-threshold", type=float, required=True,
                             help="Author-verified binarization threshold; not given in manuscript")
        elif dataset == "ASSIST2009":
            cmd.add_argument("--input", required=True)
            cmd.add_argument("--time-column", help="Use real source timestamp when documented; otherwise order_id")
        elif dataset == "EdNet":
            cmd.add_argument("--events-dir", required=True)
            cmd.add_argument("--questions", required=True)
        else:
            cmd.add_argument("--input", required=True)
            cmd.add_argument("--field-map", required=True,
                             help="JSON mapping of eight canonical fields to verified source fields")
    args = parser.parse_args()
    if args.dataset == "OULAD":
        if not 0 <= args.score_threshold <= 100:
            parser.error("OULAD threshold must be in [0,100]")
        result = make_oulad(args)
    elif args.dataset == "ASSIST2009":
        result = make_assist(args)
    elif args.dataset == "EdNet":
        result = make_ednet(args)
    else:
        result = make_feedback(args)
    validate_and_export(*result[:1], args.dataset, Path(args.output), result[1], result[2], args.hash_ids_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
