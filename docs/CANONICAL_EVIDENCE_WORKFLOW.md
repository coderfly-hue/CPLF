# Canonical evidence workflow for the four reproducibility issues

This document operationalizes the manuscript's evaluation protocol without
manufacturing missing historical artifacts.

## 1. Feedback evaluator scores

For every feedback case, method and seed, preserve the independent evaluator's
actual response JSON, raw integer scores `1..5`, the generated feedback text,
and model revision provenance. Normalize each score with `(raw-1)/4`, then
compute FQS from the normalized values and semantic relevance. The canonical
exporter requires original integers and feedback text: even on-grid normalized
values cannot be reverse-labelled as raw evaluator evidence. Legacy continuous
exports must not be rounded to create fake raw evidence.

Run after a genuine evaluator export:

```bash
python scripts/canonicalize_feedback_outputs.py \
  --input RUN_OUTPUT/feedback.csv \
  --output artifacts/canonical/feedback_outputs.csv
```

## 2. Ablation AUC/ECE

Each prediction-side ablation must export the held-out `y_true` and `y_prob` for
every dataset and seed.  `pred_score_proxy` and `ece_case_proxy` are legacy
summary aids and are rejected as canonical evidence.

```bash
python scripts/canonicalize_ablation_outputs.py \
  --input RUN_OUTPUT/ablation_predictions.csv \
  --output artifacts/canonical/ablation_prediction_outputs.csv

python scripts/recompute_prediction_metrics.py \
  --input artifacts/canonical/ablation_prediction_outputs.csv \
  --group-method-column variant \
  --ece-bins 10 \
  --seed-output artifacts/canonical/ablation_seed_metrics.csv \
  --summary-output artifacts/canonical/ablation_prediction_summary.csv
```

## 3. Split/prediction identifiers

Generate the same canonical learner/session ID from the same source identifier
before writing either fixed splits or per-case predictions.  Do not infer a
mapping between two already-divergent anonymous namespaces by row number.

```bash
python scripts/canonical_ids.py \
  --input standardized_dataset.csv \
  --source-id-column student_id \
  --output standardized_dataset_with_ids.csv
```

The resulting canonical ID must be propagated unchanged into both split files
and prediction exports. Store a true split assignment table with
`dataset,learner_or_session_id,split` and pass it alongside the held-out
predictions when exporting:

```bash
python scripts/build_canonical_release.py \
  --predictions RUN_OUTPUT/held_out_predictions.csv \
  --split-assignments RUN_OUTPUT/canonical_split_assignments.csv
```

Legacy `ANONLEARNER_` IDs are intentionally rejected as canonical evidence;
never guess their mapping by row order.

## 4. Checkpoints, LoRA adapters and logs

The executable training functions already save prediction checkpoints/test
predictions and LoRA adapter outputs.  For every completed run, add an immutable
manifest containing seed, configuration hash, split hashes, model revision,
software versions and SHA-256 hashes of the produced artifacts:

```bash
python scripts/capture_run_manifest.py \
  --run-type cplf_generator_lora \
  --seed 2024 \
  --config "CPLF LLM protocols io complete package/CPLF LLM protocols io FORMAL RELEASE/protocol_config.yaml" \
  --artifact RUN/adapter_model.safetensors \
  --artifact RUN/adapter_config.json \
  --output artifacts/run_manifests/cplf_generator_seed_2024.json
```

Do the same for all five seeds and for the prediction runs.  Genuine large
artifacts may be stored externally if the manifest records an immutable archive
reference and checksum.

## Release rule

`artifacts/canonical/` is the only location that may be described as raw
reproduction evidence.  Historical noncanonical files remain available for
traceability but are labelled in `artifacts/LEGACY_ARTIFACT_STATUS.json`.
