#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

required = [
    "fixed_splits/OULAD_splits.csv",
    "fixed_splits/ASSIST2009_splits.csv",
    "fixed_splits/EdNet_splits.csv",
    "fixed_splits/MathDial_splits.csv",
    "fixed_splits/CodeWorkout_splits.csv",
    "fixed_splits/CodeStream_splits.csv",
    "per_case_outputs/prediction_outputs.csv",
    "per_case_outputs/feedback_outputs.csv",
    "per_case_outputs/ablation_outputs.csv",
    "release_data/seed_level_results.csv",
    "release_data/all_baseline_pairwise_tests.csv",
    "release_data/all_ablation_pairwise_tests.csv",
]

missing = [rel for rel in required if not os.path.isfile(os.path.join(ROOT, rel))]
if missing:
    print("PUBLIC RELEASE CHECK: FAIL")
    for rel in missing:
        print("missing:", rel)
    sys.exit(1)

print("PUBLIC RELEASE CHECK: PASS")
print("All required split, per-case, seed-level, and statistical result files are present.")
