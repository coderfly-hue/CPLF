# Manuscript supporting-information replacement map

This file resolves the supporting-information cross-reference in
`PROTOCOLS_IO_PROTOCOL.md`. The supplied manuscript describes public GitHub
files instead of a separate supporting-information attachment.

| Manuscript information | File in this formal-release directory | Verification scope |
| --- | --- | --- |
| Table 1 dataset and split sizes | `fixed_splits/*.csv` | Counts and uniqueness, not raw-data lineage |
| Tables 2-4 prediction/profile metrics | `release_data/seed_level_results.csv`, `release_data/seed_level_summary.csv`, `reported_results/profile_validity_summary.csv` | Published numeric rows, not retrained checkpoints |
| Qwen, Direct LLM and evaluator setup | `protocol_config.yaml`, `llm_baseline_implementation_config.md`, `requirements.txt` | Documented configuration |
| Complete archived prompt wording | `prompts/01*` through `prompts/08*` | Runtime fields documented in `prompts/README.md` |
| Tables 6-8 feedback and human validation | `reported_results/feedback_generation_summary.csv`, `reported_results/feedback_dataset_level_summary.csv`, `reported_results/human_validation_summary.csv` | Reported summaries |
| Tables 9-11 verification analysis | `reported_results/feedback_generation_summary.csv`, `reported_results/ablation_summary.csv`, manuscript Tables 9-11 | Not all before/after error-rate and revision-stage source records are archived |
| Tables 12-13 paired statistics and ablations | `release_data/all_baseline_pairwise_tests.csv`, `release_data/all_ablation_pairwise_tests.csv`, `reported_results/ablation_summary.csv` | Reported comparisons, not independently rerun bootstrap |
| Five-seed result schema | `release_data/seed_level_results_SCHEMA.csv` | Header-only schema, not observation data |
| Case-level prediction, feedback and ablation exports | `per_case_outputs/*.csv` | See `per_case_outputs/README.md` for limitations |
| Executable reference protocol | `cplf_llm_protocol_complete.py`, `scripts/*.py` | Code available; source datasets and archived models not included |

For the paper's Algorithm 1, see the repository-level
`docs/ALGORITHM1_MAPPING.md`. For independent reproducibility limits,
see `docs/EXPERIMENT_REPRODUCTION.md` and
`docs/DATA_PROVENANCE_AND_LIMITATIONS.md` at the repository root.
