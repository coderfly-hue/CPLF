# Reproducibility scope

## Published evidence

The public numerical results and computational protocol accompany the CPLF-LLM
manuscript. File hashes in the protocol's `MANIFEST_SHA256.csv` protect the
published version against accidental edits.

The released materials cover the manuscript-reported learner/session splits,
five-seed aggregates, baseline/ablation comparisons, prompt texts and the
executable computational protocol. Six third-party raw datasets are not
redistributed.

## What can be checked from these files

- Each of the six split lists has the manuscript's Table 1 group counts and
  unique identifiers within the respective dataset.
- The seed-level rows aggregate to the published five-seed summary within
  precision from CSV rounding.
- Main AUC, FQS, leakage and ablation summary values match the revised paper.
- Each feedback case's FQS agrees with the stated weighted formula, subject
  to the displayed numeric precision.
- Release hashes and recorded file inventories are self-consistent.

## Additional information needed for a fully independent rerun

The public prediction case table and the split list use different anonymous
learner identifier namespaces. For example, the OULAD fixed split list has
28,749 learner/session IDs, whereas the prediction file covers 180 distinct
learner identifiers, with no direct exact-ID join. The original selection or
mapping rule is not included in this release.

When the 90,000 public prediction rows are grouped by method, dataset and
seed, their AUC values differ from the five-seed run summary in some groups;
the largest absolute difference observed during package validation was about
0.0129. These may represent different export scopes, but the repository alone
does not establish that explanation. Consequently, do not claim that every
paper metric can be recalculated exactly from this *particular* case table.

The package also does not bundle the six raw third-party source datasets,
original GPU checkpoint weights, LoRA adapters or the full training environment
snapshots. The code and reported research results are public, but
independent end-to-end training reproducibility requires those inputs and
their corresponding licenses.

The manuscript specifies a single 1-5 integer rating for each of the four
rubric-based feedback dimensions, linearly mapped to `{0,.25,.5,.75,1}`.
Every one of the 45,000 public feedback rows instead contains continuous
values off that grid in all four dimensions. These may be aggregated or
postprocessed scores, but their exact transformation is not documented or
provided. The feedback export also omits the generated feedback texts and
raw evaluator JSON. The 15,000-row ablation export explicitly labels
`pred_score_proxy` and `ece_case_proxy` as proxies; those rows do not contain
the raw outcomes needed to recompute aggregate AUC or ECE.

The repository now includes the manuscript's modular `src/cplf_llm/` paths,
which re-export functions from the shared implementation in `core.py`; the
paper-linked monolithic protocol is also retained. The accurate mapping is
`docs/ALGORITHM1_MAPPING.md`. The archived manuscript source originally used
the different path `docs/algorithm1_mapping.md`; the repository's revised
LaTeX source corrects this reference. See `DATA_PROVENANCE_AND_LIMITATIONS.md`
for machine-checkable audit findings.

## Appropriate publication statement

"The repository provides the executable protocol, machine-readable numerical
results, fixed split tables, structured case-level exports and paired-bootstrap
comparison files associated with the revised manuscript. Full independent
training reproduction requires the external datasets and model artifacts."

Published-file integrity checks are not equivalent to an independent training rerun.
