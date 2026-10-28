# Canonical exports (currently not populated)

This directory intentionally contains **no claimed primary experimental
outputs** from the historical five-seed runs. Empty presence is not a passing
independent-reproduction result.

After retrieving genuine source records, generate the following CSVs using
`scripts/build_canonical_release.py`:

- `prediction_outputs.csv`: `dataset,case_id,learner_id,method,seed,y_true,y_prob`;
  the `learner_id` must join an independently recorded canonical split ID.
- `split_assignments.csv`: `dataset,learner_or_session_id,split` from the same
  genuinely standardized source, supplied with `--split-assignments` when
  building the canonical prediction export. Predictions must use `CID_` IDs.
- `feedback_outputs.csv`: dataset/case/method/seed, original feedback text,
  actual `evaluator_raw_response`, raw evaluator integers, normalized scores,
  relevance, FQS and verification state. The strict checker compares JSON
  ratings with retained integer scores.
- `ablation_prediction_outputs.csv`: `dataset,case_id,variant,seed,y_true,y_prob`.
- Derived metric tables recomputed from those source records.

Continuous legacy rubric scores and `*_proxy` columns are not substitutes.
See `docs/CANONICAL_EVIDENCE_WORKFLOW.md` and
`docs/RELEASE_READINESS_AND_SOURCE_EVIDENCE.md`.
