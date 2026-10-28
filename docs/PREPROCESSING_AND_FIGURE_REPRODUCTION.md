# Preprocessing, table and figure reproduction guide

This guide provides executable and auditable steps for the six paper-named
datasets and the principal published empirical tables and figures. The six
third-party raw datasets are not redistributed. The paper does not supply
all original source-schema, label-conversion and anonymization decisions;
these must be verified from the actual experiment preparation records.

## 0. Environment

For offline table and figure rebuilding:

```bash
python -m pip install -r environment/plot_requirements.txt
```

For preprocessing and the manuscript-linked computational protocol:

```bash
python -m pip install -r "CPLF LLM protocols io complete package/CPLF LLM protocols io FORMAL RELEASE/requirements.txt"
```

The reported GPU training environment and model versions are separately
specified in `docs/EXPERIMENT_REPRODUCTION.md`. The plotting environment is
not represented as the original training environment.

## 1. Source preparation commands

`scripts/preprocess_datasets.py` requires genuine source files and fails on
missing correctness labels, timestamps, source field mappings and concept
tags. Each successful output creates a sibling `.provenance.json` file with
input SHA-256 hashes, explicit transformation decisions, output hash and row
counts. If source identifiers must be shared, use `--hash-ids-env NAME` with
a private environment-provided key and keep that key stable across runs.
Do not publish identifiable learner data or the key.

### OULAD

Supply `studentAssessment.csv` and `assessments.csv` from the same licensed
OULAD release. The conversion creates an assessment-level interaction,
joining on `id_assessment`; the assessment type is used as the available
concept/category field and submission day as the relative timestamp. The
manuscript does not state the score-to-binary threshold, so the author must
confirm it. The command below uses an explicit example threshold of 50
**only to illustrate syntax**, not as a claimed original paper choice:

```bash
python scripts/preprocess_datasets.py OULAD \
  --student-assessment /DATA/OULAD/studentAssessment.csv \
  --assessments /DATA/OULAD/assessments.csv \
  --score-threshold 50 \
  --output outputs/standardized/OULAD.csv
```

Change `--score-threshold` to the exact value established by original
preprocessing records before claiming a matched replication. Exercise IDs
include module/presentation to avoid collisions.

### ASSIST2009 Skill Builder

Uses original `user_id`, `problem_id`, `skill_id`, `correct`, and `order_id`
fields. Without a verified timestamp field, the within-source `order_id` is
treated explicitly as an ordering index, not a fabricated wall-clock time.

```bash
python scripts/preprocess_datasets.py ASSIST2009 \
  --input /DATA/ASSIST2009/skill_builder_data.csv \
  --output outputs/standardized/ASSIST2009.csv
```

### EdNet KT1

Expects the per-user KT1 event CSV files and matching `questions.csv`. It
compares each `user_answer` with the source `correct_answer`, joins tags by
`question_id`, and preserves the source timestamp. Obtain both files from
the same official dataset version.

```bash
python scripts/preprocess_datasets.py EdNet \
  --events-dir /DATA/EdNet/KT1 \
  --questions /DATA/EdNet/contents/questions.csv \
  --output outputs/standardized/EdNet.csv
```

### MathDial, CodeWorkout and CodeStream

Their exact raw source formats and the original correctness/feedback
transformations are not established by this release. An author-verified
flattened CSV, JSON/JSONL or Parquet source, plus a JSON field map, is
therefore mandatory. Eight canonical fields must map to *actual* source
columns: `student_id`, `exercise_id`, `concepts`, `correct`, `timestamp`,
`question_text`, `response_text`, `reference_feedback`.

For already standardized input, `configs/standardized_feedback_field_map.json`
is an identity mapping. For raw sources, copy it and replace the mapping
values with the genuine column names. A mapping file is a transformation
configuration, **not** an assertion about the original benchmark schema.

```bash
python scripts/preprocess_datasets.py MathDial \
  --input /DATA/MathDial/verified_records.jsonl \
  --field-map /CONFIGS/mathdial_fields.json \
  --output outputs/standardized/MathDial.csv

python scripts/preprocess_datasets.py CodeWorkout \
  --input /DATA/CodeWorkout/verified_records.csv \
  --field-map /CONFIGS/codeworkout_fields.json \
  --output outputs/standardized/CodeWorkout.csv

python scripts/preprocess_datasets.py CodeStream \
  --input /DATA/CodeStream/verified_records.csv \
  --field-map /CONFIGS/codestream_fields.json \
  --output outputs/standardized/CodeStream.csv
```

The original split-ID mapping must be supplied separately to recreate the
paper's exact 70/10/20 allocation. Creating a new seeded split is a new
experiment, not the exact reported test partition.

## 2. Recreate the published numerical tables

```bash
python scripts/reproduce_tables.py --output-dir outputs/reproduced_tables
```

The command recounts Table 1 from the archived split IDs, recomputes all
267 five-seed group means/SDs from reported seed rows (and checks rounding
against the shipped summaries), and exports manuscript-aligned Tables 2, 4,
6, 7, 8, 12 and 13. It also exports the manuscript's sequence-length and
feedback-error tables from labelled transcriptions. Every output is listed
in `table_provenance.json` with input hashes and evidence class.

Tables generated from published summaries are **not** evidence that their
underlying models were independently retrained. To obtain a raw-computation
reproduction, supply the held-out labels/probabilities and original paired
case records, then use the metric/bootstrap procedures in
`scripts/recompute_prediction_metrics.py` and the formal-release
`scripts/export_reproducibility_results.py`.

## 3. Recreate Figures 4--8

```bash
python scripts/reproduce_figures.py --all --output-dir outputs/reproduced_figures
```

| Figure | Source used by plotting script | Evidence level |
| --- | --- | --- |
| 4 AUC/F1 comparisons | Archived five-seed summary CSV | Published numeric summary |
| 5 learner-concept heatmap | `paper/plot_values/profile_heatmap_display_transcription.csv` and validity summary | Published two-decimal display transcription only |
| 6 sequence-length AUC | Transcribed manuscript sequence-length table | Published table means |
| 7 five-dimension radar | Archived feedback summary CSV; leakage control = `1 - leakage` | Published numeric summary |
| 8 verification errors | Transcribed manuscript feedback-error table | Published table rates |

`figure_provenance.json` records every input and output checksum. The
regenerated images may differ stylistically from the author's archived
PNGs while preserving their reported values. Figures 1--3 are conceptual
schematics archived as `paper/figures/figure1.png` through `figure3.png`;
editable original diagram sources were not supplied and no claim of an
independent diagram-source regeneration is made.

The display transcriptions are **not** raw experiment evidence. The genuine
source values for Figure 5 and original five-seed sensitivity/error traces
remain required for an independent results-level reproduction.

## 4. Validation and publication readiness

```bash
python scripts/verify_formal_release.py
python scripts/verify_root_release_manifest.py
python scripts/audit_published_cases.py
python -m unittest discover -s tests -v
python scripts/verify_canonical_release.py --root . --strict
```

The last command is intentionally nonzero until genuine canonical outputs
are available. Use the Data Availability paragraph in the revised
`paper/main_clean.tex` and `docs/RELEASE_READINESS_AND_SOURCE_EVIDENCE.md`
to describe the package accurately. These scripts do not add fabricated
missing historical run records.
