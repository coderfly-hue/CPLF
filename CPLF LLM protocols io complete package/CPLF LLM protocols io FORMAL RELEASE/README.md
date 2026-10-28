# CPLF-LLM protocols.io formal computational package

> **Computational release:** This directory provides the protocol, executable
> implementation, configuration, prompts, fixed data splits, machine-readable
> results and statistical comparison tables associated with the manuscript.

This package contains the computational protocol, executable implementation, configuration, prompts, baseline settings, and machine-readable result files accompanying the CPLF-LLM manuscript.

## Main files

- `PROTOCOLS_IO_PROTOCOL.md` — protocol text for protocols.io.
- `cplf_llm_protocol_complete.py` — executable CPLF-LLM protocol implementation.
- `protocol_config.yaml` — experiment configuration.
- `requirements.txt` — Python dependencies.
- `prompts/` — CPLF-LLM, Direct LLM, evaluator, and revision prompt templates.
- `llm_baseline_implementation_config.md` / `.csv` — implementation settings for the LLM feedback comparisons.
- `release_data/` — five-seed results and complete paired-bootstrap baseline/ablation comparisons.
- `reported_results/` — manuscript-level summary tables.
- `scripts/export_reproducibility_results.py` — export/regeneration utility.
- `prompts/README.md` — explains the intentional template variables.
- `per_case_outputs/README.md` — numerical export schema and provenance limits.
- `FORMER_SUPPORTING_INFORMATION_MAP.md` — maps manuscript evidence tables to published files.

## Implementation note

The manuscript denotes the temporal encoder abstractly as `Encoder_theta`. The executable protocol instantiates this component as a one-layer GRU so that the released protocol is directly runnable. Dataset statistics are computed from standardized input files rather than hard-coded into the implementation.

## Installation

```bash
pip install -r requirements.txt
```

Use the standardized study datasets and the fixed configuration in `protocol_config.yaml`. Fixed split IDs and experiment outputs should be preserved exactly when reproducing the released analyses.

The manuscript reports A100/CUDA 12.1 training with five seeds (2024-2028),
but the public archive is not the original checkpoint/log archive. The
predictor and LoRA/generation CLIs accept `--seed`; newly completed LoRA
training runs write a `run_metadata.json` with their input checksum and
settings. This does not retroactively establish provenance for the published
CSV tables. See the repository root's
`docs/EXPERIMENT_REPRODUCTION.md` and
`docs/DATA_PROVENANCE_AND_LIMITATIONS.md`.

## Results included in the release

The package includes five-seed numerical results, complete paired-bootstrap comparison tables, prompt templates and LLM baseline settings for the manuscript's statistical reporting and computational protocol.
Published feedback-case files contain only rubric scores, and the ablation
case file contains explicit proxy fields. These are not substitutes for the
raw generated feedback/evaluator logs or raw prediction outputs used to
independently recompute the corresponding manuscript results.
