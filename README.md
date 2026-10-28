# CPLF-LLM — Official Computational Release

This repository contains the executable CPLF-LLM computational protocol and machine-readable research result tables accompanying the paper *A Cognition-Aware Large Language Model Framework for Student Cognitive Profiling and Personalized Learning Feedback in Higher Education and Online Learning*.

The paper reports completed five-seed experiments. This archive releases the associated numerical results and reference workflow. Before reusing the result tables, please consult the provenance and independent-reproduction documents listed below.

The code, data schemas, result tables, and protocol documentation are located in the directory referenced in the paper:

`CPLF LLM protocols io complete package/CPLF LLM protocols io FORMAL RELEASE/`

## Repository Contents

| Files | Purpose |
| --- | --- |
| `cplf_llm_protocol_complete.py` | Executable computational workflow |
| `protocol_config.yaml`, `requirements.txt` | Model, training, verification, and environment settings |
| `prompts/` | Eight prompt files for generation, evaluation, and revision |
| `fixed_splits/` | Six learner/session-level train/validation/test allocations |
| `per_case_outputs/` | Prediction, feedback, and ablation records |
| `release_data/` | Five-seed values, summaries, and paired-bootstrap comparisons |
| `reported_results/` | Results reported in the paper |
| `MANIFEST_SHA256.csv` | Published file-integrity hashes |

All of these are located inside the paper-linked protocol directory.

## Verify the Published Release

From the repository root, run:

```bash
python scripts/verify_formal_release.py
python scripts/verify_root_release_manifest.py
python scripts/audit_published_cases.py
python -m unittest discover -s tests -v
```

These offline checks verify file hashes, group splits, five-seed aggregation, feedback score arithmetic, the main published metrics, and structural integrity. They also report known provenance gaps in the case-level exports. The checks do not download large language models or repeat the GPU experiments.

When unresolved source-evidence findings remain, `python scripts/audit_published_cases.py --strict` returns a nonzero exit status.

## Run the Computational Protocol

```bash
cd "CPLF LLM protocols io complete package/CPLF LLM protocols io FORMAL RELEASE"
python -m pip install -r requirements.txt
python cplf_llm_protocol_complete.py --help
```

Third-party benchmark raw datasets, trained model weights, and LoRA adapters are not bundled. Obtain these resources subject to their respective access requirements.

## Manuscript and Evidence Notes

- `docs/PAPER_ALIGNMENT_20261028.md` maps the revised paper's tables, methodology, and data-availability claims to released files.
- `docs/REPRODUCIBILITY_SCOPE.md` explains which checks are independently possible using public files and which require additional primary evidence.
- `docs/EXPERIMENT_REPRODUCTION.md` provides paper-derived training settings, five-seed commands, required inputs, and an evidence checklist.
- `docs/ALGORITHM1_MAPPING.md` maps Algorithm 1 in the paper to functions in the actual monolithic implementation and records a manuscript path mismatch.
- `docs/DATA_PROVENANCE_AND_LIMITATIONS.md` describes limitations related to the score grid, anonymous IDs, missing feedback text, and ablation proxy values.
- `CITATION.cff` contains authorship and citation information.

The package provides the published computational artifacts. An independent end-to-end training replication is beyond the scope of the included offline file-consistency tests. The score-only feedback export and proxy-valued ablation export must not be presented as raw evaluator logs or AUC/ECE rerun logs.

## Canonical Evidence Layer Added for Manuscript Alignment

The repository now includes `src/cplf_llm/` modules matching the implementation paths listed after Algorithm 1, author-provided manuscript source files under `paper/`, and a strict artifact-processing pipeline under `artifacts/` and `scripts/`.

The strict pipeline addresses four release risks:

1. New feedback runs preserve the evaluator's raw 1–5 rubric scores before normalization and FQS calculation.
2. Prediction and ablation AUC and ECE are recomputed from held-out `y_true/y_prob` values rather than proxy columns.
3. A deterministic canonical-ID utility enables fixed splits and per-case outputs to share a common learner/session identifier namespace.
4. Completed training and evaluation runs can include SHA-256 run manifests documenting checkpoints, LoRA adapters, split files, configuration, and software versions.

See `docs/CANONICAL_EVIDENCE_WORKFLOW.md` and run:

```bash
python scripts/verify_canonical_release.py --root .
```

The command above performs a **non-strict structural check**. It may pass even when `artifacts/canonical/` contains no newly archived run records. To require genuine canonical feedback, prediction, and ablation exports, run `python scripts/verify_canonical_release.py --root . --strict`.

## Rebuild Tables, Figures, and Standardized Dataset Inputs

From the repository root, the following commands regenerate the published numerical table exports and the **display-level results** for empirical Figures 4–8, without downloading models or modifying the archived numbers:

```bash
python -m pip install -r environment/plot_requirements.txt
python scripts/reproduce_tables.py --output-dir outputs/reproduced_tables
python scripts/reproduce_figures.py --all --output-dir outputs/reproduced_figures
```

`docs/PREPROCESSING_AND_FIGURE_REPRODUCTION.md` provides the six dataset-specific preparation commands and their required original source inputs.

Figure 5's matrix and Figures 6 and 8 can be reproduced as **published displays** using clearly identified transcribed values in `paper/plot_values/`; these are not recovered raw observations. Figures 1–3 are conceptual illustrations retained as PNG source assets. Evidence for independent raw-to-results reruns remains subject to the limitations described above.
