# CPLF-LLM protocols.io reproducibility package

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
- `FORMER_SUPPORTING_INFORMATION_MAP.md` — mapping from the removed Supporting Information items to their public replacements in this package.

## Implementation note

The manuscript denotes the temporal encoder abstractly as `Encoder_theta`. The executable protocol instantiates this component as a one-layer GRU so that the released protocol is directly runnable. Dataset statistics are computed from standardized input files rather than hard-coded into the implementation.

## Installation

```bash
pip install -r requirements.txt
```

Use the standardized study datasets and the fixed configuration in `protocol_config.yaml`. Fixed split IDs and experiment outputs should be preserved exactly when reproducing the released analyses.

## Results included in the release

The package includes the final author-confirmed five-seed numerical results and the complete paired-bootstrap comparison tables used to support the revised statistical reporting. These files replace the separate S3, S4, and S5 materials previously referenced in the manuscript. Complete prompts and LLM baseline settings replace S1 and Table S1.
