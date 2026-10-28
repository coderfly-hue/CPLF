# CPLF-LLM Computational Protocol

## Title

Cognition-Aware Large Language Model Framework for Student Cognitive Profiling and Personalized Learning Feedback

## Purpose

This protocol provides a complete computational workflow for reproducing the CPLF-LLM pipeline: standardized educational interaction processing, fixed learner-level splitting, cognition-aware learner profiling, dynamic profile memory, next-step correctness prediction, profile-guided feedback planning, retrieval of pedagogical evidence, Qwen2.5-7B-Instruct feedback generation, same-backbone Direct LLM comparison, independent feedback verification, FQS computation, human-validation analysis, and paired-bootstrap inference.

The revised manuscript evaluates prediction/profile modeling on OULAD, ASSIST2009, and EdNet, and feedback generation on MathDial, CodeWorkout, and CodeStream. It reports that the framework combines concept mastery, misconception tendency, engagement state, and uncertainty, and evaluates personalized feedback across relevance, diagnostic consistency, actionability, personalization, and answer leakage.

## Attached code

`cplf_llm_protocol_complete.py`

The attached file is a complete executable reference implementation. The manuscript leaves the temporal `Encoder_theta` abstract; this reference protocol instantiates it as a one-layer GRU so that every step is executable.

## Dataset sources

OULAD  
https://analyse.kmi.open.ac.uk/open_dataset

ASSISTments 2009-2010 Skill Builder  
https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010

EdNet  
https://github.com/riiid/ednet

MathDial  
https://github.com/eth-nlped/mathdial

CodeWorkout  
https://pslcdatashop.web.cmu.edu/Files?datasetId=3458

CodeStream  
https://data.mendeley.com/datasets/n77t7z9zcr/1

Do not redistribute third-party raw datasets inside the protocol attachment. Download them under their original terms and transform them into the standardized interaction schema below.

## Standardized interaction schema

Required columns:

- `student_id`
- `exercise_id`
- `concepts`
- `correct`
- `timestamp`

Optional columns used when available:

- `time_spent`
- `attempts`
- `hints`
- `interval`
- `question_text`
- `response_text`
- `reference_feedback`
- `misconception_label`

`concepts` is a pipe-separated string, for example:

`probability|sample_space`

## Software

- Python 3.10 or later
- PyTorch 2.4.1
- Transformers 4.46.3
- PEFT 0.13.2
- CUDA 12.1 for the reported GPU environment
- Sentence Transformers
- pandas
- NumPy
- SciPy
- scikit-learn
- PyYAML
- pyarrow
- huggingface_hub

## Models

Feedback backbone:

`Qwen/Qwen2.5-7B-Instruct`

Fixed revision:

`acbd96531cda22292a3ceaa67e984955d3965282`

Sentence embedding model:

`BAAI/bge-base-en-v1.5`

Independent rubric evaluator:

`meta-llama/Llama-3.1-8B-Instruct`

## Random seeds

Use the same five seeds in all repeated model runs:

- 2024
- 2025
- 2026
- 2027
- 2028

## Procedure

### 1. Install the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python --version
pip freeze > environment_freeze.txt
nvidia-smi > gpu_environment.txt
```

### 2. Download and preserve raw datasets

Create:

```text
data_raw/
data_processed/
splits/
outputs/
resources/
```

Keep the original downloads unchanged.

### 3. Standardize each dataset

Transform each dataset into the schema described above.

Validate a standardized file:

```bash
python cplf_llm_protocol_complete.py validate \
  --input data_processed/OULAD.csv \
  --output data_processed/OULAD_validated.csv
```

The script prints the learner, interaction, task, and concept counts directly from the data. Use these computed counts in the manuscript rather than manually entering dataset-scale numbers.

### 4. Create fixed learner-level splits

```bash
python cplf_llm_protocol_complete.py split \
  --input data_processed/OULAD_validated.csv \
  --output-dir splits/OULAD \
  --seed 2024
```

The split is performed once using 70% training, 10% validation, and 20% test at learner/session level. The IDs are stored as files and reused for every method and every random seed.

### 5. Train the prediction/profile model

The reference implementation uses:

- hidden dimension = 128
- concept embedding dimension = 128
- maximum sequence length = 200
- dropout = 0.2
- Adam learning rate = 5e-4
- early stopping patience = 10

Run each seed:

```bash
for SEED in 2024 2025 2026 2027 2028
do
  python cplf_llm_protocol_complete.py train-predictor \
    --input data_processed/OULAD_validated.csv \
    --split-dir splits/OULAD \
    --output-dir outputs/OULAD \
    --seed $SEED \
    --config protocol_config.yaml
done
```

Repeat for ASSIST2009 and EdNet.

### 6. Cognitive profile construction

For every interaction, the code produces:

- concept mastery
- misconception probability distribution
- engagement representation
- uncertainty representation

The components are concatenated to form the learner cognitive profile.

### 7. Dynamic profile memory

At each time step, a learned update gate combines:

- previous profile memory
- current profile
- current interaction representation

Only history available up to time `t` is used to predict time `t+1`.

### 8. Prediction evaluation

The code exports per-case test predictions and calculates:

- Accuracy
- AUC
- F1
- NLL
- Brier score
- ECE

Use held-out concept-level performance to calculate DOA and Spearman correlation for profile validation.

### 9. Prepare pedagogical resources

Create a resource table with:

```text
resource_id,text,concepts,difficulty
```

`difficulty` is normalized to `[0,1]`.

### 10. Train the CPLF-LLM feedback adapter

The generator configuration is:

- Qwen2.5-7B-Instruct
- revision `acbd96531cda22292a3ceaa67e984955d3965282`
- frozen base weights
- LoRA rank = 16
- LoRA alpha = 32
- LoRA dropout = 0.05
- target modules = q_proj, k_proj, v_proj, o_proj
- 3 epochs
- AdamW
- learning rate = 2e-4
- weight decay = 0.01
- 5% warm-up
- effective batch size = 32
- maximum training sequence length = 2048

```bash
python cplf_llm_protocol_complete.py train-generator \
  --input data_processed/feedback_train.csv \
  --output-dir outputs/cplf_adapter \
  --config protocol_config.yaml
```

### 11. Train the same-backbone Direct LLM adapter

Use exactly the same Qwen checkpoint, LoRA configuration, training split, and optimization budget:

```bash
python cplf_llm_protocol_complete.py train-generator \
  --input data_processed/feedback_train.csv \
  --output-dir outputs/direct_adapter \
  --direct \
  --config protocol_config.yaml
```

The Direct LLM prompt excludes:

- cognitive profile
- profile memory
- retrieved pedagogical evidence
- verification information

### 12. Generate CPLF-LLM feedback

Generation settings:

- temperature = 0.3
- top-p = 0.90
- repetition penalty = 1.05
- maximum new tokens = 256
- manuscript-facing length constraint = 150 words

```bash
python cplf_llm_protocol_complete.py generate-feedback \
  --input data_processed/feedback_test_cases.csv \
  --resources resources/pedagogical_resources.csv \
  --adapter outputs/cplf_adapter \
  --output outputs/cplf_feedback.csv \
  --config protocol_config.yaml
```

### 13. Generate Direct LLM feedback

```bash
python cplf_llm_protocol_complete.py generate-feedback \
  --input data_processed/feedback_test_cases.csv \
  --adapter outputs/direct_adapter \
  --direct \
  --no-revision \
  --output outputs/direct_feedback.csv \
  --config protocol_config.yaml
```

### 14. Feedback scoring

Relevance is computed with `BAAI/bge-base-en-v1.5`.

The independent Llama evaluator scores:

- diagnostic consistency
- actionability
- personalization
- answer leakage

Each rubric score is an integer from 1 to 5 and is normalized to `[0,1]`.

The evaluator is not told which method produced the feedback.

### 15. Feedback Quality Score

For every individual feedback case:

```text
FQS = max(
    0,
    0.25*R
    + 0.25*D
    + 0.25*A
    + 0.25*P
    - 0.40*L
)
```

Calculate FQS at the individual-case level before aggregating. Do not calculate FQS from rounded table means.

### 16. Feedback verification

The default validation-selected thresholds in the complete reference configuration are:

- quality threshold eta = 0.70
- answer leakage threshold delta = 0.20

A candidate is accepted when:

```text
Q >= 0.70
and
L <= 0.20
```

If CPLF-LLM fails verification, exactly one revision step is performed using the failed dimension(s) as the revision instruction. Preserve both pre- and post-verification text and scores.

### 17. Human validation

Use three independent blinded raters.

A long-format human-rating file contains:

```text
case_id,dimension,rater_id,score,automated_score
```

Calculate ICC(2,k) and automated-vs-human Spearman correlation:

```bash
python cplf_llm_protocol_complete.py human-validation \
  --input outputs/human_ratings.csv \
  --output outputs/human_validation_report.csv
```

### 18. Paired bootstrap

Use 10,000 resamples.

For prediction outcomes, bootstrap learners and retain all interactions belonging to each sampled learner.

For feedback outcomes, bootstrap task cases and retain both methods' outputs for every sampled case.

Example for CPLF-LLM vs Direct LLM FQS:

```bash
python cplf_llm_protocol_complete.py bootstrap \
  --input outputs/combined_feedback_scores.csv \
  --method-a CPLF-LLM \
  --method-b Direct_LLM \
  --metric fqs \
  --cluster case_id \
  --n-boot 10000 \
  --output outputs/cplf_vs_direct_fqs_bootstrap.csv
```

### 19. Holm correction

Combine the raw pairwise-comparison rows and run:

```bash
python cplf_llm_protocol_complete.py holm \
  --input outputs/pairwise_tests.csv \
  --p-column p_raw \
  --output outputs/pairwise_tests_holm.csv
```

### 20. Seed-level summaries

```bash
python cplf_llm_protocol_complete.py summary \
  --input outputs/seed_level_results.csv \
  --metrics auc,f1,nll,brier,ece,fqs \
  --output outputs/seed_level_summary.csv
```

### 21. Ablation analyses

Repeat the full evaluation after individually disabling:

1. cognitive profile construction
2. dynamic profile memory
3. retrieval-augmented feedback planning
4. uncertainty-aware updating
5. feedback verification

Interpret each module using the outcome it can directly affect. In particular, feedback verification is downstream of prediction and should be interpreted primarily through FQS and answer leakage rather than a small AUC change.

### 22. Robustness analyses

Prediction:

- history retained: 40%, 60%, 80%, 100%
- label noise: 0%, 5%, 10%, 15%
- missing concept annotations: 10%, 20%, 30%

Feedback:

- mask the corresponding concept/skill/pedagogical metadata at the same missingness levels.

Perturb training inputs only. Keep validation/test targets unchanged.

### 23. Parameter sensitivity

Evaluate:

- retrieved resources `K_r` = 1, 3, 5, 7
- acceptance threshold eta = 0.60, 0.70, 0.80, 0.90
- leakage threshold delta = 0.10, 0.20, 0.30
- generation-loss weight lambda_gen = 0.25, 0.50, 1.00, 2.00

### 24. Machine-readable outputs

Archive:

- fixed split IDs
- vocabulary
- model checkpoints
- five seed-level prediction files
- feedback text before verification
- feedback text after verification
- all rubric component scores
- FQS per case
- human ratings
- bootstrap distributions/results
- Holm-adjusted statistical tables
- environment freeze
- configuration file
- exact prompts

## Expected output structure

```text
outputs/
  OULAD/
    predictor_seed_2024.pt
    predictor_seed_2025.pt
    predictor_seed_2026.pt
    predictor_seed_2027.pt
    predictor_seed_2028.pt
    test_predictions_seed_2024.csv
    ...
  cplf_adapter/
  direct_adapter/
  cplf_feedback.csv
  direct_feedback.csv
  human_validation_report.csv
  pairwise_tests_holm.csv
```

## Quality-control rules

1. Never use test examples for training, prompt selection, hyperparameter selection, threshold selection, or retrieval tuning.
2. Reuse exactly the same fixed split IDs for all methods.
3. Keep case IDs paired across feedback methods.
4. Compute FQS from unrounded per-case component scores.
5. Treat five seeds as optimization-replication summaries, not as the only inferential units.
6. Use learner-level paired bootstrap for prediction and case-level paired bootstrap for feedback.
7. Store the exact Qwen checkpoint revision.
8. Keep the evaluator blind to method identity.
9. Do not overwrite pre-verification outputs after revision.
10. Generate manuscript tables programmatically from archived machine-readable outputs.

## Troubleshooting

### Qwen revision mismatch

Run:

```bash
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct', revision='acbd96531cda22292a3ceaa67e984955d3965282', local_files_only=True))"
```

### FQS does not match a manuscript table

Recalculate per-case FQS from the unrounded component values and then aggregate.

### Data leakage

Check that the intersection between each pair of learner ID files is empty.

### Statistical comparison is not paired

Join outputs by learner/case ID before bootstrap resampling.

### Llama evaluator cannot be downloaded

The model requires access under its provider's distribution terms. Authenticate with the model host and confirm access before running evaluator steps.


## Reproducibility files replacing separate Supporting Information

The public protocols.io package includes the complete CPLF-LLM, Direct LLM, evaluator, and revision prompt templates in the `prompts/` directory. LLM comparison settings are summarized in `llm_baseline_implementation_config.md` and `.csv`. Manuscript-level summary tables are included in `reported_results/`. The final five-seed results are provided in `release_data/seed_level_results.csv`, complete all-baseline paired-bootstrap comparisons are provided in `release_data/all_baseline_pairwise_tests.csv`, and full-model versus ablation comparisons are provided in `release_data/all_ablation_pairwise_tests.csv`. `FORMER_SUPPORTING_INFORMATION_MAP.md` maps these files to the material that would otherwise be supplied as Supporting Information. The manuscript's Algorithm 1 is mapped to the **actual single-file reference implementation** in the repository-level `docs/ALGORITHM1_MAPPING.md`; the modular `src/cplf_llm/` paths named in the manuscript are not packaged here.

The released numerical tables correspond to the final experimental results confirmed by the authors and are provided in machine-readable form for independent inspection and table regeneration.

## Data and code availability

The executable reference implementation is attached directly to this protocol. Raw third-party educational datasets are not redistributed; they are obtained from the dataset sources listed above. When run with suitable source inputs, the protocol can produce split lists, checkpoints, per-case predictions, generated feedback texts, rubric scores and statistical outputs. **The shipped published exports do not include all such raw outputs:** in particular, the feedback-case CSV contains only scores, the ablation-case CSV contains proxy fields, and matching original checkpoint/log and ID-mapping evidence is absent. Therefore the paper-aligned result tables can be checked offline, but complete independent rerunning of the reported experiments needs the sources listed in `docs/EXPERIMENT_REPRODUCTION.md` at the repository root.
