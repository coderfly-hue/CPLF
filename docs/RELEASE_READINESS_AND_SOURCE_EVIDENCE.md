# Release readiness and primary-evidence checklist

## Status as of 28 October 2026

The archive is suitable for publication as an **annotated reference
implementation and historical numerical-results release** after the ordinary
repository checks pass. It is **not** a complete independent experimental
reproduction archive. The manuscript reports completed five-seed experiments;
the repository cannot establish their complete execution history from its
present files alone. This distinction must be preserved in descriptions,
release tags and documentation.

The author's 35-page manuscript PDF was used to cross-check the model names,
seeds, feedback rubric, reported tables and documentation paths. The PDF does
not contain the primary run records listed below. It cannot serve as a source
from which to synthesize them.

## Required source evidence before claiming end-to-end reproduction

| Item | Required original evidence | Destination / check | Current state |
| --- | --- | --- | --- |
| Prediction linkage | Original standardized source IDs, selection/export rules and exact mapping to published split and case IDs | Consistent canonical IDs in split and prediction records; join/check per dataset | Not archived; 0 exact legacy ID joins |
| Prediction run provenance | Five-seed training configurations, logs, trained predictor checkpoints and held-out predictions | `artifacts/checkpoints/` or immutable external archive; `artifacts/run_manifests/` | Not archived |
| Feedback raw scoring | Generated feedback text, evaluator raw JSON, four integer scores 1--5, evaluator revision, and any historical aggregation/conversion script | `artifacts/canonical/feedback_outputs.csv` plus verified original records | Not archived; legacy scores are continuous |
| Ablation AUC/ECE | Held-out `y_true`/`y_prob` for every prediction-side variant, dataset and seed | `artifacts/canonical/ablation_prediction_outputs.csv`; recompute AUC/ECE | Not archived; legacy columns are proxies |
| Generator provenance | Five-seed genuine LoRA adapters, training logs, model revision and environment metadata | `artifacts/adapters/` or immutable external archive; run manifests | Not archived |
| Dataset preparation | Source dataset versions, original standardized transformations and split generation records | Reproduction documentation and hash-linked run manifests | Not archived |
| Inferential replay | Original paired observations and bootstrap RNG/analysis records for all published comparisons | Re-run the published bootstrap utilities on matched observations | Not fully archived |
| Evaluator revision | Immutable ID or commit from the experiment records for `meta-llama/Llama-3.1-8B-Instruct` | `artifacts/experiment_manifest.json` and linked run manifest | Not stated in manuscript; `revision` is `null` |
| Software rights | Written authorization to publish code, figures and original outputs, followed by an author-chosen license | Root `LICENSE` when approved | Not provided by the paper |

The provided `release_data/seed_level_results.csv` is a populated table of
reported seed values, **not** the missing primary GPU logs. Do not rebrand it
as raw run traces. `per_case_outputs/` is a disclosed legacy export: do not
silently rescale rubric columns, rename proxies, assign legacy anonymous IDs
by row order or infer a model revision from publication date.

## What the included checks prove

```bash
python scripts/verify_formal_release.py
python scripts/audit_published_cases.py
python scripts/verify_root_release_manifest.py
python scripts/verify_canonical_release.py --root .
python -m unittest discover -s tests -v
```

These checks establish structural, numerical and hash consistency of the
released material. The ordinary canonical checker reports absent canonical
data as information; it does not certify its presence.

```bash
python scripts/audit_published_cases.py --strict
python scripts/verify_canonical_release.py --root . --strict
```

The strict commands must return nonzero while the known primary-evidence gaps
remain. They are evidence-availability gates, not tests of whether the paper's
reported scientific conclusions are true or false.

## How to supply genuine records

1. Recover the unmodified original run/export files from the experiment
   environment, with original identifiers, timestamps and model revisions.
2. Add source-to-standardized-ID provenance and ensure the same canonical ID
   is emitted into splits and held-out predictions. Keep the original legacy
   export unchanged and document the mapping, not a guessed join.
3. Run `scripts/build_canonical_release.py` only on genuine raw feedback,
   predictions and ablation predictions. The helper rejects proxy ablation
   columns and incompatible feedback scores.
4. Archive checkpoint and adapter hashes with
   `scripts/capture_run_manifest.py` for every actual seed/run.
5. Recompute metrics and check them against the manuscript; investigate and
   document any differences, including export scope, rather than adjusting
   observed rows to match a target mean.
6. Run both strict checks and independently reproduce the analysis outputs
   before upgrading the publication's reproducibility claim.

## Publication wording

> This repository provides the CPLF-LLM executable reference protocol,
> machine-readable numerical tables, public fixed splits and disclosed legacy
> case-level exports associated with the manuscript. File-level and numerical
> consistency checks pass. Complete independent replication requires the
> original model/data transformations, run artifacts, evaluator source outputs
> and corresponding provenance records, which are not bundled here.

No mock, simulated or imputed training/evaluation records should be added in
place of missing original evidence.
