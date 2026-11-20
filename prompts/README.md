# Prompt bundles

`base_prompts.json` is the *initial* prompt bundle — the starting point of the
FGPR optimisation procedure (`candidate_id: pre_feedback_original_prompt`).
It defines:

| key | purpose |
| --- | --- |
| `extraction_system_prompt` | system message for element extraction |
| `extraction_prompt` | template with `[[CONTENT]]` / `[[HIGH_LEVEL_LAWS]]` slots |
| `classification_system_prompt` | system message for conflict classification |
| `classification_prompt` | template with candidate and element slots |

Placeholders are filled by `legalsem.prompts`:

| placeholder | filled with |
| --- | --- |
| `[[CONTENT]]` | subordinate-law clause text |
| `[[HIGH_LEVEL_LAWS]]` | numbered superior-law clauses |
| `[[CANDIDATE_LABELS]]` | comma-separated admissible label ids |
| `[[CANDIDATE_EXPLANATIONS]]` | `id: name - rationale` lines |
| `[[CONTENT_ELEMENTS]]` | JSON of the extracted elements |
| `[[TASK_TYPE]]` | trace of which elements triggered the reduction |

## Why only the initial bundle is included

The optimised bundle produced by FGPR is not part of this release. Running the
pipeline with `base_prompts.json` reproduces the *method*, not the exact
numbers reported in the paper, which were obtained after prompt optimisation on
the development split. To reproduce those numbers, supply your own optimised
bundle with the same schema and point `--prompt-bundle` at it.

## Adding a bundle

Any JSON file with the four required keys above works. Validate it with:

```bash
PYTHONPATH=src python -c "from legalsem.prompts import load_bundle; print(load_bundle('prompts/base_prompts.json')['candidate_id'])"
```
