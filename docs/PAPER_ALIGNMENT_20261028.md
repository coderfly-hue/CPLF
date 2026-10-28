# 2026-10-28 revised manuscript alignment checklist

The following locations refer to the paper-linked protocol directory:
`CPLF LLM protocols io complete package/CPLF LLM protocols io FORMAL RELEASE/`.
Numbers were checked against the 35-page revised manuscript supplied by the
author. Matching a reported table is a document-consistency check, not an
independent training rerun.

| Revised manuscript | Released evidence | Check |
| --- | --- | --- |
| Table 1: six dataset counts and 70/10/20 learner/session splits | `fixed_splits/*.csv` | All six train/validation/test group counts match |
| Tables 2 and 4: prediction AUC, F1, NLL and Brier | `release_data/seed_level_results.csv`, `seed_level_summary.csv` | Five seeds and principal CPLF-LLM AUC means match |
| Section LLM generation: Qwen2.5-7B-Instruct, fixed revision, LoRA r16/alpha32, 3 epochs | `protocol_config.yaml`, protocol source | Published settings present |
| Verification: BGE relevance, Llama four-dimension rubric, weighted FQS, thresholds | Protocol source, `prompts/05`–`08` | Functions and rubric prompts present |
| Table 6: five feedback methods, FQS and five dimensions | `reported_results/feedback_generation_summary.csv` | All principal CPLF-LLM and Direct LLM values match |
| Table 7: dataset-level feedback FQS | `reported_results/feedback_dataset_level_summary.csv` | Published dataset breakdown present |
| Table 8: human validation agreement | `reported_results/human_validation_summary.csv` | Published dimensions and agreement values present |
| Table 12: principal paired-bootstrap comparisons | `reported_results/principal_statistical_comparisons.csv` | All five principal comparisons present |
| Table 13: ablation AUC/ECE/FQS/leakage | `reported_results/ablation_summary.csv` | Six variants and main numeric rows match |
| Statistical methods: five seeds, 10,000 resamples, Holm adjustment | `release_data/`, `scripts/export_reproducibility_results.py` | Parameters, result files and export utility present |
| Data Availability: named GitHub subdirectory | Directory named exactly as referenced | Present |

The manuscript leaves the temporal `Encoder_theta` abstract; the executable
protocol supplies a one-layer GRU instantiation. Full input-to-output
verification of archived model training is distinct from checking the above
file/table correspondence. See `REPRODUCIBILITY_SCOPE.md`.

**Algorithm 1 code-path alignment:** The repository contains the
`src/cplf_llm/` modules named after Algorithm 1, backed by the same
implementation as the retained `cplf_llm_protocol_complete.py` script.
The revised LaTeX manuscript now refers to the actual
`docs/ALGORITHM1_MAPPING.md` path. Any previously circulated PDF with the
older path should be regenerated before submission.

**Case export exception:** The scores-only feedback CSV cannot be shown to
be the direct output of the specified single-integer evaluator because its
rubric columns are continuous. The ablation-case file labels its prediction
and calibration values as proxies. See `DATA_PROVENANCE_AND_LIMITATIONS.md`.

## 2026-10-28 canonical-evidence enhancement

The repository has subsequently been extended to make the implementation paths
and evidence requirements named in the manuscript executable rather than merely
descriptive:

- `src/cplf_llm/` now contains the modular paths referenced after Algorithm 1,
  backed by the same executable core as the formal protocol script.
- new feedback generation runs retain raw independent-evaluator integers 1--5
  and their normalized values; FQS is derived from those values.
- inference explicitly enforces the manuscript's 150-word feedback bound.
- the implementation-specific ECE choice is recorded as 10 equal-width bins.
- `scripts/canonicalize_ablation_outputs.py` refuses proxy ablation fields as
  raw evidence; prediction-side ablations must export `y_true/y_prob`.
- `scripts/canonical_ids.py` supplies a shared deterministic learner/session ID
  namespace for new standardized data, fixed splits and predictions.
- `scripts/capture_run_manifest.py` records hashes and environment metadata for
  genuine checkpoints, LoRA adapters, split files and outputs.
- the manuscript source and supplied figures are stored under `paper/` for
  repository-to-paper cross-checking.

Historical CSVs are not rewritten to simulate missing raw evidence. Their status
is recorded in `artifacts/LEGACY_ARTIFACT_STATUS.json`; genuine rerun artifacts
belong in `artifacts/canonical/`.
