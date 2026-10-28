# Canonical experimental artifacts

This directory defines the evidence chain required to reproduce manuscript
tables from model outputs rather than from hand-entered summary values.

## Canonical outputs

A genuine rerun should populate:

- `canonical/prediction_outputs.csv`: one row per held-out prediction with
  dataset, case_id, learner_id, method, seed, y_true, y_prob.
- `canonical/feedback_outputs.csv`: generated feedback text, raw evaluator
  integers (1--5), normalized rubric scores, semantic relevance, FQS, and
  verification state.
- `canonical/ablation_prediction_outputs.csv`: true labels and probabilities for
  each prediction-side ablation.
- `canonical/ablation_feedback_outputs.csv`: raw/normalized feedback scores for
  each feedback-side ablation.
- `run_manifests/`: split hashes, model revisions, software versions, seeds, and
  artifact SHA-256 values.
- `checkpoints/` and `adapters/`: genuine archived model artifacts when they are
  redistributed.

The release does not fabricate missing historical raw evaluator integer outputs,
ablation probabilities, checkpoints, or adapters. Legacy published exports remain
under the formal-release directory and are explicitly labelled when they cannot be
reconstructed from archived raw observations.

Use `scripts/build_canonical_release.py` after a genuine rerun/export. It refuses
continuous rubric scores incompatible with the manuscript's 1--5 evaluator protocol
and refuses proxy ablation columns as canonical evidence.
