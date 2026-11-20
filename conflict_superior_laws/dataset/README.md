# dataset

- `train.jsonl`：训练反馈集，4895条；仅用于错误聚合和候选 Prompt 生成。
- `val.jsonl`：独立验证集，696条；仅用于指标选择候选 Prompt。
- `test.jsonl`：held-out 测试集，1404条；Prompt 冻结后运行一次。
- `10.jsonl`：小规模调试样本，不用于正式指标。

主要字段包括 `id`、`content`、`high_level_laws`、`conflict_type`。不要把 `conflict_type` 或 `conflict_description` 放入模型推理提示词。
