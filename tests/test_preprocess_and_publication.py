"""Synthetic unit fixtures exercise input rules; they are not experiment data."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.preprocess_datasets import (
    make_assist, make_ednet, make_feedback, make_oulad, validate_and_export,
)
from scripts.reproduce_tables import regenerate_seed_summary, splits_table

ROOT = Path(__file__).resolve().parents[1]


class SourceAdapterTests(unittest.TestCase):
    def test_oulad_requires_author_chosen_threshold(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            folder = Path(tmp)
            events = folder / "studentAssessment.csv"
            meta = folder / "assessments.csv"
            pd.DataFrame({"id_student": [7, 8], "id_assessment": [11, 11],
                          "score": [51, 35], "date_submitted": [20, 22]}).to_csv(events, index=False)
            pd.DataFrame({"id_assessment": [11], "assessment_type": ["TMA"],
                          "code_module": ["AAA"], "code_presentation": ["2013J"]}).to_csv(meta, index=False)
            args = SimpleNamespace(student_assessment=events, assessments=meta, score_threshold=50)
            frame, sources, choices = make_oulad(args)
            self.assertEqual(frame.correct.tolist(), [1, 0])
            self.assertEqual(choices["score_threshold"], 50)
            dest = folder / "normalized.csv"
            validate_and_export(frame, "OULAD", dest, sources, choices, None)
            self.assertEqual(len(pd.read_csv(dest)), 2)
            self.assertTrue(dest.with_suffix(".csv.provenance.json").is_file())

    def test_assist2009_source_order_and_binary_response(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            src = Path(tmp) / "assist.csv"
            pd.DataFrame({"user_id": [1], "problem_id": [2], "skill_id": [3],
                          "correct": [1], "order_id": [9]}).to_csv(src, index=False)
            frame, _, choices = make_assist(SimpleNamespace(input=src, time_column=None))
            self.assertEqual(frame.timestamp.tolist(), [9])
            self.assertIn("order_id", choices["timestamp_semantics"])

    def test_ednet_question_answer_join(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            folder = Path(tmp)
            events = folder / "KT1"
            events.mkdir()
            pd.DataFrame({"question_id": ["q1"], "user_answer": ["b"],
                          "timestamp": [123456]}).to_csv(events / "u12.csv", index=False)
            bank = folder / "questions.csv"
            pd.DataFrame({"question_id": ["q1"], "correct_answer": ["b"],
                          "tags": ["addition"]}).to_csv(bank, index=False)
            frame, _, _ = make_ednet(SimpleNamespace(events_dir=events, questions=bank))
            self.assertEqual(frame.correct.tolist(), [1])
            self.assertEqual(frame.student_id.tolist(), ["u12"])

    def test_feedback_adapters_require_explicit_verified_mappings(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            folder = Path(tmp)
            mapping = {field: field for field in ["student_id", "exercise_id", "concepts",
                                                   "correct", "timestamp", "question_text",
                                                   "response_text", "reference_feedback"]}
            cfg = folder / "mapping.json"
            cfg.write_text(json.dumps(mapping), encoding="utf-8")
            src = folder / "source.csv"
            pd.DataFrame([{**{field: "source" for field in mapping}, "correct": 1,
                           "timestamp": 100}]).to_csv(src, index=False)
            for dataset in ("MathDial", "CodeWorkout", "CodeStream"):
                frame, inputs, decisions = make_feedback(SimpleNamespace(
                    dataset=dataset, input=src, field_map=cfg))
                validate_and_export(frame, dataset, folder / f"{dataset}.csv",
                                    inputs, decisions, None)
            self.assertTrue((folder / "CodeStream.csv").is_file())

    def test_missing_outcomes_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as tmp:
            frame = pd.DataFrame([{"student_id": "s", "exercise_id": "q", "concepts": "c",
                                   "correct": None, "timestamp": 1}])
            with self.assertRaisesRegex(ValueError, "missing mandatory"):
                validate_and_export(frame, "ASSIST2009", Path(tmp) / "out.csv", [], {}, None)


class ReproductionHelpersTests(unittest.TestCase):
    def test_reported_seed_values_aggregate_consistently(self):
        self.assertEqual(len(regenerate_seed_summary()), 267)

    def test_split_counts_recomputed_from_archived_ids(self):
        data = splits_table()
        self.assertEqual(len(data), 6)
        self.assertEqual(data.total.sum(), 88922)


if __name__ == "__main__":
    unittest.main()
