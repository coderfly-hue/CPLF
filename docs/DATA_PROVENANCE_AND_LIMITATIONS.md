# Public result provenance and interpretation

This note compares the public archive with the supplied 35-page 2026-10-28
manuscript and the actual packaged code. The paper **reports completed
experiments**; the following observations concern what can be independently
verified from the archive, not a determination that the reported research is
synthetic or fraudulent. Numerical CSV values are preserved verbatim.

## Evidence classes

| File family | What is present | What it establishes | Source evidence still needed |
| --- | --- | --- | --- |
| `reported_results/*.csv` | Tables corresponding to the manuscript | Report-to-file transcription | Original analysis logs and derived notebooks/scripts |
| `release_data/seed_level_results.csv` and summary | Five-seed numeric rows; summary arithmetic is checkable | Five reported seed values aggregate correctly | Checkpoint/log-to-row provenance |
| `release_data/*pairwise_tests.csv` | Published effect sizes, intervals and p-values | Reported inferential table | Matched original run outputs to rerun bootstrap |
| `fixed_splits/*.csv` | Six anonymized 70/10/20 group assignments | Correct counts and no duplicate public IDs | Mapping to original standardized learner/session identifiers |
| `per_case_outputs/prediction_outputs.csv` | 90,000 probability rows | Partial case-level calculations | Selection/export rules, exact IDs and alignment to seed summaries |
| `per_case_outputs/feedback_outputs.csv` | 45,000 rubric-score rows | FQS arithmetic within displayed precision | Feedback text, raw evaluator responses and scoring/export transformations |
| `per_case_outputs/ablation_outputs.csv` | 15,000 proxy/score rows | Descriptive proxy values | Raw predictions and the actual AUC/ECE aggregation |

## Identified non-equivalences

1. **Feedback rubric resolution.** The manuscript and
   `FeedbackEvaluator.rubric()` require one integer rating from 1 to 5 per
   evaluator dimension, normalized to one of `0,.25,.5,.75,1`. In the
   supplied 45,000-row feedback export, every value in each of the four
   rubric dimensions is off that grid. An additional aggregation or
   postprocessing step may explain this, but none is supplied. Thus this
   file cannot be described as the direct raw output of that evaluator.
2. **Export schema.** The runnable feedback function emits before/final
   feedback texts and score fields, while the released score-only CSV does
   not include these texts or the raw JSON. The current reference output
   schema and the published export represent different stages; the archival
   transformation is missing.
3. **Prediction identifiers and AUC.** Anonymous learner IDs in the public
   case export do not join directly to the anonymous split-list IDs. Some
   grouped case-level AUCs differ from the five-seed summary values; the
   previous release audit observed a maximum absolute discrepancy of
   approximately .0129. Different selection/export scopes could explain
   this, but that explanation is unverified without source provenance.
4. **Ablation proxy fields.** `pred_score_proxy` and `ece_case_proxy` are
   explicitly named proxies, not the raw labels and probability vectors
   needed to recompute AUC or ECE. These fields should never be relabeled
   as observed per-case AUC/ECE or used as independent proof of Table 13.
5. **Data schema only.** `seed_level_results_SCHEMA.csv` has a header and
   no observations. It is a documented export schema, not a missing five-seed
   dataset; the populated reported results are in `seed_level_results.csv`.

## What was actually checked here

- The formal-release manifest hashes and counts.
- Six anonymized split counts and distinct public identifiers.
- Five-seed summary arithmetic within CSV rounding precision.
- Feedback FQS arithmetic in 45,000 rows.
- Alignment of selected reported AUC/FQS headlines with the manuscript.
- Structural and rubric-resolution warnings emitted by
  `scripts/audit_published_cases.py`.

**Not checked:** original model-training execution on the reported A100,
checkpoint provenance, per-case feedback text, original evaluator JSON,
dataset conversion, or independent reruns of published paired bootstrap
tests. No such evidence was manufactured or inferred from the paper's
aggregate results.

## Appropriate research reporting

> The manuscript reports five-seed benchmark experiments and their
> aggregate results. This repository provides the associated published
> numerical tables and executable reference protocol. Offline integrity
> checks pass. Full independent replication and complete case-level
> provenance require original data transformations, checkpoints and logs,
> score-export rules, and matched identifiers not contained here.

If the archived evaluator used score averaging or a different scoring
implementation, publish that exact script, input records, version and
transformation before claiming that the 45,000 feedback rows arise directly
from the released code. Do not fill these gaps by synthesizing run records
from published means.
