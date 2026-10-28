# Reproducing the reported CPLF-LLM experiments

## Status and evidence boundary

The **manuscript reports completed benchmark experiments**, including
five runs with seeds 2024-2028, using six public datasets. The release
contains paper-reported result tables and executable reference code. The
supplied archive does **not** contain the complete source-data conversions,
original training logs, optimizer states, model checkpoints, LoRA adapters,
or an identity-preserving join between all public case and split exports.
Therefore the commands below are a reproducible *procedure*, not a claim
that a fresh end-to-end training run has been executed and verified here.

Three distinct validation levels must not be conflated:

1. **Paper-report alignment:** check stored CSV summaries against manuscript
   Tables 1-13 and the reported training settings.
2. **Offline artifact consistency:** check SHA-256 hashes, split counts,
   FQS arithmetic, and five-seed aggregation using the repository verifier.
3. **Independent computational reproduction:** obtain the original permitted
   datasets and archived run evidence, perform the five-seed training and
   evaluation, and compare derived results with the paper. This third level
   cannot be certified from the included archive alone.

## Experiment specification from the 2026-10-28 manuscript

| Item | Manuscript-reported specification |
| --- | --- |
| Prediction benchmarks | OULAD, ASSIST2009, EdNet |
| Feedback benchmarks | MathDial, CodeWorkout, CodeStream |
| Split | Fixed learner/session-level 70/10/20; shared across methods and seeds |
| Prediction training | Sequence length 200; hidden dimension 128; Adam; validation-AUC early stopping with patience 10 |
| Random seeds | 2024, 2025, 2026, 2027, 2028 |
| Generator | `Qwen/Qwen2.5-7B-Instruct`, revision `acbd96531cda22292a3ceaa67e984955d3965282` |
| Adaptation | LoRA rank 16, alpha 32, dropout .05, q/k/v/o projections; 3 epochs |
| LoRA optimization | AdamW; learning rate 0.0002; weight decay .01; warm-up .05; effective batch size 32; max tokens 2048; bfloat16 |
| Inference | Temperature .3; top-p .90; repetition penalty 1.05; limit 150 words in prompt |
| Evaluator | `meta-llama/Llama-3.1-8B-Instruct`, temperature 0; 256 new tokens; 1-5 integer rubric |
| Retrieval and relevance | `BAAI/bge-base-en-v1.5`; top K=5 |
| Verification | FQS threshold eta=.70 and leakage threshold delta=.20 in released config; one revision if required |
| Inference statistics | 10,000 paired cluster bootstrap resamples; Holm correction |
| Environment **reported by the paper** | PyTorch 2.4.1, Transformers 4.46.3, PEFT 0.13.2, CUDA 12.1, NVIDIA A100 80-GB |

The reported hardware is **not** a claim that this repository review ran on
that hardware. Additional system, CUDA driver and package provenance should
be saved with a new run.

## Input preparation

From the manuscript-linked protocol directory, install the requirements in
an environment with an appropriate CUDA stack and permitted model access:

```bash
python -m pip install -r requirements.txt
python cplf_llm_protocol_complete.py --help
```

Download the six original third-party datasets under their licenses. Map each
dataset into a standardized CSV or Parquet table with at least
`student_id,exercise_id,concepts,correct,timestamp`. For feedback training,
provide `question_text,response_text,reference_feedback` and the relevant
learner-profile evidence; the code lists optional fields in its module docstring.
These third-party source datasets and the **original historical** dataset
transformation scripts are not shipped. The repository now includes strict
new source adapters for all six named datasets in
`scripts/preprocess_datasets.py`, documented in
`docs/PREPROCESSING_AND_FIGURE_REPRODUCTION.md`. The author must verify raw
field mappings and any manuscript-unstated label decisions before treating
a new conversion as identical to the reported experimental preparation.

```bash
python cplf_llm_protocol_complete.py validate \
  --input /path/to/OULAD_standardized.parquet \
  --output /path/to/work/OULAD_normalized.parquet
```

The public `fixed_splits/*_splits.csv` files use anonymized `ANON_*` IDs;
the training CLI consumes `train_ids.txt`, `validation_ids.txt` and
`test_ids.txt` containing IDs from the *same* standardized input namespace.
The exact anonymization mapping is absent, so the public lists cannot safely
be joined directly to the source tables. To rerun the **published split**,
obtain the original, matching ID lists under appropriate privacy controls.
To create a *new* deterministic 70/10/20 split (not necessarily the paper's
exact split), use:

```bash
python cplf_llm_protocol_complete.py split \
  --input /path/to/work/OULAD_normalized.parquet \
  --output-dir /path/to/work/OULAD_ids --seed 2024
```

## Five-seed prediction training

Repeat for each prediction dataset with its corresponding standardized file
and original split-ID directory. Keep outputs separate by dataset and seed.

```bash
for seed in 2024 2025 2026 2027 2028; do
  python cplf_llm_protocol_complete.py train-predictor \
    --input /path/to/work/OULAD_normalized.parquet \
    --split-dir /path/to/work/OULAD_ids \
    --output-dir "/path/to/runs/OULAD/seed_${seed}" \
    --config protocol_config.yaml --seed "$seed"
done
```

Each successful run writes a model checkpoint, vocabulary, training history,
test prediction rows and test metric JSON. Retain the original data version,
split lists, software environment and hardware record with these outputs.
The paper reports AUC means .891 (OULAD), .866 (ASSIST2009) and .823
(EdNet); matching these numbers is a result comparison, not proof of
matching data lineage.

## Five-seed LoRA and feedback evaluation

The generator training input is a **training-split** table containing
reference feedback and prompt context. The `--direct` flag removes profile
guidance to create the same-backbone Direct LLM control. Use separate
directories per method and seed, and never use test feedback for fitting.

```bash
for seed in 2024 2025 2026 2027 2028; do
  python cplf_llm_protocol_complete.py train-generator \
    --input /path/to/work/feedback_train.csv \
    --output-dir "/path/to/runs/cplf_adapter_seed_${seed}" \
    --config protocol_config.yaml --seed "$seed"
  python cplf_llm_protocol_complete.py generate-feedback \
    --input /path/to/work/feedback_test.csv \
    --adapter "/path/to/runs/cplf_adapter_seed_${seed}" \
    --resources /path/to/work/pedagogical_resources.csv \
    --output "/path/to/runs/cplf_feedback_seed_${seed}.csv" \
    --config protocol_config.yaml --seed "$seed"
done
```

Run corresponding Direct LLM commands with `train-generator --direct` and
`generate-feedback --direct`, its own adapter and its own outputs. The
paper reports 600 held-out feedback cases from each of MathDial, CodeWorkout
and CodeStream (1,800 unique cases), repeated across methods and seeds.
The published CSV is a score-only export: it does not contain the actual
generated feedback strings or an auditable link to the raw evaluator JSON.
Its continuously valued rubric scores are not direct outputs of the current
single-integer rubric implementation. See `DATA_PROVENANCE_AND_LIMITATIONS.md`
before using it as a raw replay input.

## Metrics, statistical inference and release audit

The paper uses `FQS = max(0, .25R + .25D + .25A + .25P - .40L)`.
For an independent rerun, derive per-case values from source feedback and
evaluator responses, then compute matched-case, 10,000-resample bootstrap
comparisons. `scripts/export_reproducibility_results.py` demonstrates the
export/bootstrap format, but source run archives are needed to regenerate
the published statistics independently.

From the **repository root**, perform the provided offline checks:

```bash
python scripts/verify_formal_release.py
python scripts/audit_published_cases.py
python -m unittest discover -s tests -v
```

The first and third commands check the published files; the audit flags
source-provenance questions without silently rewriting them. A clean
file-integrity result does **not** certify the training history.

## What must be supplied to assert full independent reproduction

- Licensed source-data versions, checksums, preprocessing scripts and
  split/anonymization lineage (with learner privacy safeguarded).
- Predictor checkpoints and logs for all five seeds, plus LoRA adapter
  weights and run metadata for CPLF-LLM and Direct LLM.
- Baseline implementations, exact method-specific settings and run logs
  for LaMP, OPPU, LearnLens and the prediction baselines.
- Raw generated feedback texts, independent evaluator outputs, any
  aggregation/normalization script, and verification/revision trace.
- Case- and learner-level paired keys used for 10,000-resample bootstrap,
  and the programmatically regenerated Tables 2-13.

Until these sources are available and checked, describe this repository as
**paper-aligned published artifacts and an executable reference protocol**,
not a verified independent reproduction of the complete experimental run.

## Published table and figure rebuild commands

From the repository root, install the lightweight plotting requirements and
run both publication-level rebuilders:

```bash
python -m pip install -r environment/plot_requirements.txt
python scripts/reproduce_tables.py --output-dir outputs/reproduced_tables
python scripts/reproduce_figures.py --all --output-dir outputs/reproduced_figures
```

The table helper recomputes all reported seed-summary groups from the archived
seed rows and exports selected manuscript-aligned tables. Figures 4 and 7 use
released summary CSVs; Figures 5, 6 and 8 use explicitly identified published
display/table transcriptions. These recreate the **published numeric display**,
not missing primary experimental observations. For the full source list,
file checksums and exact data-adapter commands, see
`docs/PREPROCESSING_AND_FIGURE_REPRODUCTION.md`.
