# dataset

- `train.json`：5569条，开发集中的反馈集（由 6555 条原始开发数据分层划分得到），用于错误聚合和候选 Prompt 生成。
- `val.json`：986条，开发集中的验证集，只用于 Macro-F1 选择。
- `test.json`：1639条 held-out 测试数据，来自官方发布划分，未重采样。

原始字段：`id`、`title`、`clause_content`、`regulation`、`category`。

如需从原始 6555 条重新划分，在 `../3.feedback_refinement/` 中运行 `prepare_dev_split.py`（输入为原始开发数据，输出 feedback/val）。不要从 test 中切验证集。
