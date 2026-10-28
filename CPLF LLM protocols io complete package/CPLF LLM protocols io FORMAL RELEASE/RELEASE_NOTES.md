# Release notes

## 2026-10-28

This release includes the CPLF-LLM executable protocol, complete prompt
templates, fixed learner/session partitions, LLM baseline settings, five-seed
result rows, paired-bootstrap comparison tables, ablation results and
file-integrity hashes. Numerical tables are in CSV format for downstream
inspection and analysis. Computational environment and model requirements are
documented in `requirements.txt` and `protocol_config.yaml`.

## 2026-10-28 documentation and replay-path clarification

The supplementary mapping file, template-field guide, case-export data
dictionary and repository-level Algorithm 1 and experiment-reproduction guides
are included. The CLI now accepts explicit generator/generation seeds and
records metadata for newly completed LoRA training jobs. Existing published
numeric data are not rewritten. The provenance audit expressly flags
unresolved export lineage and proxy semantics; no original GPU runs or missing
baselines are claimed to have been rerun by this packaging update.
