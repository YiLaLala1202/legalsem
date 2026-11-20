# output

实验产物的落盘位置。`run.py` 按下列路径写入：

```text
output/
  baseline/{backbone}_{split}_baseline_output.jsonl       Direct 31 分类
  full/elements/{backbone}_{split}_extract.jsonl          31 类主题证据抽取
  full/pred_classfication/{backbone}_{split}_class.jsonl  动态候选分类
  ablation/{backbone}_{split}_class_no_dynamic.jsonl      静态候选集消融
```

用 `--output` 可覆盖默认路径。不同 backbone 使用独立文件名；重跑前先确认输出是
续跑还是覆盖模式，避免把旧结果与新结果混合。
