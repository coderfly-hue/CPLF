"""Standard-library regression tests for the published formal release."""
from __future__ import annotations

import unittest
from pathlib import Path

from scripts.verify_formal_release import verify
from scripts.verify_formal_release import REPO, RELEASE
from scripts.audit_published_cases import audit
from scripts.verify_root_release_manifest import verify as verify_root_manifest


class FormalReleaseTests(unittest.TestCase):
    def test_repository_manifest(self):
        self.assertEqual(verify_root_manifest(), [])
    def test_release_integrity(self):
        errors, _, counts = verify()
        self.assertEqual(errors, [])
        self.assertGreaterEqual(counts["manifest_files"], 40)
        self.assertEqual(counts["split_ids"], 88922)

    def test_paper_linked_directory_is_single_source(self):
        parent = REPO / "CPLF LLM protocols io complete package"
        dirs = [p for p in parent.iterdir() if p.is_dir()]
        self.assertEqual(dirs, [RELEASE])
        for name in ("prediction_outputs.csv", "feedback_outputs.csv", "ablation_outputs.csv"):
            self.assertTrue((RELEASE / "per_case_outputs" / name).is_file())

    def test_research_results_are_present(self):
        self.assertTrue((RELEASE / "release_data/seed_level_results.csv").is_file())
        self.assertTrue((RELEASE / "reported_results/feedback_generation_summary.csv").is_file())

    def test_paper_mapping_and_provenance_are_explicit(self):
        for name in ("ALGORITHM1_MAPPING.md", "EXPERIMENT_REPRODUCTION.md",
                     "DATA_PROVENANCE_AND_LIMITATIONS.md"):
            self.assertTrue((REPO / "docs" / name).is_file())
        self.assertTrue((RELEASE / "FORMER_SUPPORTING_INFORMATION_MAP.md").is_file())
        self.assertTrue((RELEASE / "prompts/README.md").is_file())
        self.assertTrue((RELEASE / "per_case_outputs/README.md").is_file())

    def test_case_export_audit_exposes_known_limitations(self):
        result = audit()
        self.assertEqual(result["feedback_rows"], 45000)
        self.assertEqual(result["unique_feedback_case_ids"], 1800)
        self.assertEqual(result["prediction_rows"], 90000)
        self.assertTrue(result["provenance_findings"])


if __name__ == "__main__":
    unittest.main()
