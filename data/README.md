# data

The full datasets are **not** redistributed in this repository. This directory
holds only 20-case samples that document the field layout.

```text
data/
  lcr-cn/
    sample.jsonl        20 cases, field layout for the conflict task
    train.jsonl         ← place here (4895 cases)
    val.jsonl           ← place here (696 cases)
    test.jsonl          ← place here (1404 cases)
  slard-31/
    sample.json         20 cases, field layout for the topic task
    train.json             ← place here (5569 cases)
    val.json               ← place here (986 cases)
    test.json              ← place here (1639 cases)
```

## Field reference

LCR-CN (`conflict_superior_laws`):

| field | role |
| --- | --- |
| `id`, `title`, `level`, `category` | metadata |
| `content` | subordinate-law clause (model input) |
| `high_level_laws` | superior-law clauses, list (model input) |
| `conflict_type` | gold class name — **scoring only, never in the prompt** |
| `conflict_description`, `distractors` | annotation rationale and negative examples — **never in the prompt** |

SLARD-31 (`SLARD_classification`):

| field | role |
| --- | --- |
| `id`, `title` | metadata |
| `clause_content` | clause text (model input) |
| `regulation` | regulation title (model input) |
| `category` | gold topic label — **scoring only** |

## Split discipline

- `train` is the feedback set: it is used to aggregate errors and propose prompt
  candidates. It is never used to fit the element-to-label mapping.
- `val` is used only to select among prompt candidates by macro-F1.
- `test` is run exactly once, after the prompt is frozen. Test labels never enter
  extraction, candidate construction, or prompt selection.

## Obtaining the full data

`SLARD_classification/dataset/README.md` documents how the development split was
derived from the publicly released SLARD data, including the
`prepare_dev_split.py` reproduction path. LCR-CN is distributed separately by the
authors under its own terms; see that README for the contact and license notes.
