# Case-level published exports: data dictionary and provenance

These are numerical exports accompanying the paper, not a self-contained
archive of every original training or inference event. Do not modify the
numerical values to force alignment with the manuscript means.

## `prediction_outputs.csv` (90,000 rows)

Columns: `dataset`, `case_id`, `learner_id`, `y_true`, `method`, `seed`,
`y_prob`. Covers OULAD, ASSIST2009 and EdNet for five methods and the five
reported seeds. The published anonymized `ANONLEARNER_*` IDs are in a
different namespace from `fixed_splits/ANON_*` IDs. These tables cannot be
joined exactly without the original mapping/selection procedure. Some
case-level recalculated AUCs differ from stored five-seed summaries.

## `feedback_outputs.csv` (45,000 rows)

Columns: `dataset`, `case_id`, `method`, `seed`, `relevance`,
`diagnostic_consistency`, `actionability`, `personalization`,
`answer_leakage`, `fqs`, `accepted_after_verification`. The 1,800 unique
held-out cases (600 from each of MathDial, CodeWorkout and CodeStream)
are represented across five feedback methods and five seeds.

`fqs` follows the paper's Eq. (38) within six-decimal rounding. The file
contains **scores, not actual generated feedback text**. Its four
rubric-based fields have continuous values, whereas the packaged evaluator
assigns single 1-5 integer scores and normalizes in steps of .25. A raw
evaluator log and any score-aggregation/export code must be supplied to
explain that difference. `accepted_after_verification` in this export
indicates the numerical acceptance-threshold outcome; do not treat it as
a complete stored history of first-pass acceptance, revision and rejection.

## `ablation_outputs.csv` (15,000 rows)

Columns: `case_id`, `variant`, `seed`, `pred_score_proxy`,
`ece_case_proxy`, `fqs`, `answer_leakage`. The first two score columns are
**proxies**. AUC and ECE are aggregate metrics and cannot be independently
reconstructed from these proxy columns without the source prediction labels,
probabilities and aggregation procedure. The paper-level ablation numbers
are in `../reported_results/ablation_summary.csv`.

The archive does not establish the generation lineage of every case-level
field. See the repository's `docs/DATA_PROVENANCE_AND_LIMITATIONS.md` and
`scripts/audit_published_cases.py` before using these files for a claimed
independent experimental rerun.

## Legacy status in the 2026-10-28 paper-aligned repository

These three CSV files are retained as historical exports.  The repository's
canonical evidence contract is now defined in `../../../artifacts/`.

- `feedback_outputs.csv`: not canonical raw evaluator evidence because the
  historical export does not retain the evaluator's integer 1--5 scores.
- `ablation_outputs.csv`: not canonical AUC/ECE evidence because it contains
  proxy fields rather than held-out `y_true/y_prob`.
- `prediction_outputs.csv`: AUC can be recomputed from `y_true/y_prob`, but the
  historical learner-ID namespace is not asserted to be joinable to the
  published fixed-split namespace.

Do not round, relabel or reverse-engineer these fields to manufacture raw
observations. Genuine rerun outputs should be passed through the strict scripts
in the repository-level `scripts/` directory.
