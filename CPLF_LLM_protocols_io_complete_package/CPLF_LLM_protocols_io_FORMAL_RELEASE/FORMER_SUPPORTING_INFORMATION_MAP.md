# Former Supporting Information coverage map

The revised manuscript can omit separate Supporting Information files because the corresponding reproducibility materials are included directly in this protocols.io release.

| Former item | Purpose | Replacement in this package | Status |
|---|---|---|---|
| S1 | Complete generation and evaluation prompts | `prompts/01_...` through `prompts/08_...`; the same templates are implemented in `cplf_llm_protocol_complete.py` | Included |
| Table S1 | LLM baseline implementation choices | `llm_baseline_implementation_config.md` and `llm_baseline_implementation_config.csv` | Included |
| Table S3 | Complete paired-bootstrap comparisons against all applicable baselines | `release_data/all_baseline_pairwise_tests.csv` | Included |
| Table S4 | Full-model vs. every ablation paired-bootstrap comparisons | `release_data/all_ablation_pairwise_tests.csv` | Included |
| Table S5 | Five seed-level results underlying summary tables | `release_data/seed_level_results.csv` and `release_data/seed_level_summary.csv` | Included |

The package also contains manuscript-level summary CSV files in `reported_results/`, exact prompt templates, configuration files, the executable protocol implementation, and utilities for regenerating machine-readable result tables.
