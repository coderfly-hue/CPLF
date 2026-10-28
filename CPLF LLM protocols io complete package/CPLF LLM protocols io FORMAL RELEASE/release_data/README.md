# Released machine-readable results

This directory contains the machine-readable numerical results accompanying the CPLF-LLM manuscript and protocol. The released tables correspond to the final experimental results confirmed by the authors.

Files:

- `seed_level_results.csv` — five-seed run-level values for the reported analyses.
- `seed_level_summary.csv` — mean, sample standard deviation, and number of seeds recomputed from `seed_level_results.csv`.
- `all_baseline_pairwise_tests.csv` — complete paired-bootstrap comparisons between CPLF-LLM and applicable baselines, including effect estimates, 95% confidence intervals, raw empirical p-values, and Holm-adjusted p-values.
- `all_ablation_pairwise_tests.csv` — paired-bootstrap comparisons between the full framework and each ablated variant.
- `seed_level_results_SCHEMA.csv` — **header-only schema**, retained for export-format reference; it is not an observation table and is not a substitute for `seed_level_results.csv`.

The inferential tables use 10,000 paired bootstrap resamples. Learners are the resampling clusters for learner-level prediction/profile outcomes and held-out task cases are the resampling units for feedback outcomes, consistent with the manuscript Statistical Analysis section.

The script `../scripts/export_reproducibility_results.py` documents and reproduces the machine-readable export format from archived per-case/per-seed outputs.
The archived original prediction/feedback/evaluator and baseline-run logs
needed for an independent regeneration of every inferential table are not
all included here. The repo-level `docs/DATA_PROVENANCE_AND_LIMITATIONS.md`
distinguishes reported values from source-derived replay evidence.
