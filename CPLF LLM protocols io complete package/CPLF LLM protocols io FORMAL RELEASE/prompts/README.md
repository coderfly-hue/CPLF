# Released prompt templates and runtime fields

The eight numbered `.txt` files are **versioned runtime templates**.
Curly-braced spans such as `{question_text}` are intentional variables,
not unfinished editorial placeholders or synthetic feedback examples.

| Template | Input fields / purpose | Code function |
| --- | --- | --- |
| `01_cplf_system_prompt.txt` | Fixed system instructions | `CPLF_SYSTEM_PROMPT` |
| `02_cplf_user_prompt_template.txt` | `question_text`, `response_text`, `correct`, `mastery_summary`, `engagement_summary`, `uncertainty_summary`, `concept_gap_text`, `misconception_cues`, `retrieved_evidence` | `build_cplf_prompt` |
| `03_direct_system_prompt.txt` | Fixed control instructions | `DIRECT_SYSTEM_PROMPT` |
| `04_direct_user_prompt_template.txt` | `question_text`, `response_text`, `correct` | `build_direct_prompt` |
| `05_evaluator_system_prompt.txt` | Fixed blind evaluator instructions | `EVALUATOR_SYSTEM_PROMPT` |
| `06_evaluator_user_prompt_template.txt` | `question_text`, `response_text`, `diagnostic_evidence`, `feedback`; JSON output rubric | `build_evaluator_prompt` |
| `07_revision_system_prompt.txt` | Fixed revision instructions | `REVISION_SYSTEM_PROMPT` |
| `08_revision_user_prompt_template.txt` | `question_text`, `response_text`, `feedback`, `failed_dimensions` | `build_revision_prompt` |

The Python implementation uses its own built-in prompt builders and does not
load the `.txt` files dynamically. These files are the human-readable
archival counterparts. For an actual inference case, the Python builders
render the fields from the standardized case data; the `concept_gap_text`,
`misconception_cues` and `retrieved_evidence` fields are computed from
the profile and optional resource retriever. Missing source case data
should **not** be filled with invented student responses or model outputs.

Do not run unguarded Python `str.format()` on the evaluator template:
its last line contains a literal JSON example with braces. Use
`build_evaluator_prompt` or a renderer that explicitly escapes this JSON.
The evaluator returns an integer 1-5 for each of `diagnostic`,
`actionability`, `personalization`, and `leakage`; each is normalized by
`(rating - 1)/4`. See the published case-level export caveat in
`../per_case_outputs/README.md`.
