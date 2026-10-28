# Algorithm 1 implementation map (manuscript dated 2026-10-28)

The repository now exposes the modular `src/cplf_llm/` paths named in the
manuscript while retaining the original single-file protocol implementation for
backward compatibility.  The modular files import the same executable core, so
there is one behavioral implementation rather than duplicated algorithms.

| Algorithm 1 step | Manuscript-aligned module | Executable symbol |
| --- | --- | --- |
| 1--3: iterate learner sequences | `src/cplf_llm/data.py` | `StudentSequenceDataset`, `collate_sequences` |
| 4: encode interaction | `src/cplf_llm/model.py` | `InteractionEncoder` |
| 5: temporal hidden state | `src/cplf_llm/model.py` | `CPLFModel` temporal encoder |
| 6: cognitive profile | `src/cplf_llm/profile.py` | `CPLFModel` profile branches |
| 7: dynamic profile memory | `src/cplf_llm/memory.py` | `CPLFModel` gated memory update |
| 8: next correctness prediction | `src/cplf_llm/prediction.py` | `train_predictor`, `collect_predictions` |
| 9: concept gap | `src/cplf_llm/retrieval.py` | `concept_gap` |
| 10: pedagogical evidence retrieval | `src/cplf_llm/retrieval.py` | `PedagogicalRetriever` |
| 11: structured feedback plan | `src/cplf_llm/planning.py` | `build_cplf_prompt` |
| 12: candidate feedback | `src/cplf_llm/generation.py` | `QwenFeedbackGenerator`, `generate_feedback_file` |
| 13: verification score | `src/cplf_llm/verification.py` | `FeedbackEvaluator`, `fqs` |
| 14--16: accept/revise | `src/cplf_llm/verification.py`, `generation.py` | `failed_dimensions`, revision prompt and rescoring |
| 17--19: outputs | `src/cplf_llm/pipeline.py` | CLI orchestration and CSV/checkpoint exports |

The implementation keeps the evaluator's raw integer 1--5 rubric outputs in new
runs and derives normalized scores from them.  It also enforces the manuscript's
150-word inference bound explicitly.  The fixed release configuration uses ten
equal-width ECE bins; the paper leaves the number of bins `B` symbolic, so the
implementation value is recorded in `protocol_config.yaml` and
`artifacts/experiment_manifest.json`.

The historical `per_case_outputs` are retained as legacy evidence.  Canonical
rerun evidence must be exported through the scripts in `scripts/` and stored in
`artifacts/canonical/`; proxy ablation fields are not accepted as canonical
AUC/ECE evidence.
