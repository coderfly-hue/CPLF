# LLM baseline implementation configuration

This file replaces the implementation-detail material that would otherwise have been placed in a Supporting Information table. It records the comparison rules stated in the revised manuscript without inventing method-specific settings that are not specified there.

## Shared evaluation controls

- Common generator where backbone replacement is permitted: `Qwen/Qwen2.5-7B-Instruct`
- Fixed Qwen revision: `acbd96531cda22292a3ceaa67e984955d3965282`
- Common held-out feedback cases and fixed train/validation/test partitions
- Decoding temperature: 0.3
- top-p: 0.90
- repetition penalty: 1.05
- feedback length limit: 150 words
- Common evaluation rubric: relevance, diagnostic consistency, actionability, personalization, answer leakage
- Independent rubric evaluator: `meta-llama/Llama-3.1-8B-Instruct`, temperature 0
- Relevance/retrieval embedding model: `BAAI/bge-base-en-v1.5`

## Method-level implementation

| Method | Backbone / adaptation | Learner information supplied | Retrieval / planning | Verification | Role in comparison |
|---|---|---|---|---|---|
| Direct LLM | Same Qwen checkpoint and same LoRA budget as CPLF-LLM | Current task and learner response only | No cognitive-profile retrieval/planning | No | Same-backbone general-purpose tutor control |
| LaMP | Common Qwen checkpoint where its published design permits backbone replacement; method-specific mechanism follows the published procedure | User-history/profile information defined by LaMP | LaMP-style personalization/retrieval | No CPLF verifier | General personalization baseline |
| OPPU | Common Qwen checkpoint where its published design permits backbone replacement; method-specific PEFT follows the published procedure | User-specific adaptation signal defined by OPPU | OPPU personalized PEFT mechanism | No CPLF verifier | General personalization baseline |
| LearnLens | Common Qwen checkpoint where its published design permits backbone replacement; curriculum-grounded procedure follows the published method | Task/curriculum and learner-error information supported by the method | Curriculum-grounded feedback structure | No CPLF verifier | Dedicated educational-feedback baseline |
| CPLF-LLM | Qwen checkpoint + LoRA rank 16, alpha 32, dropout 0.05 on q/k/v/o projections | Task, response, cognitive profile, concept gap, misconception cues, uncertainty | BGE retrieval, top-K=5, structured feedback plan | Yes | Proposed framework |

For LaMP, OPPU, and LearnLens, this package deliberately does not fabricate unpublished hyperparameters. The manuscript-level comparison uses the shared split, decoding, and evaluation controls above and follows each method's published mechanism for method-specific components. If the final experiment logs contain additional baseline-specific hyperparameters, add them to this file before public release.
