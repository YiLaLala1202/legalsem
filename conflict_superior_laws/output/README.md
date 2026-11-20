# output

实验产物的落盘位置，不放运行代码。`run.py` 按下列路径写入，`analysis/` 按同样的
路径 glob 读取，两边保持一致：

```text
output/
  baseline/{backbone}_{split}_baseline_output.jsonl      Direct 五分类
  full/elements/{backbone}_{split}_extract.jsonl         语义元素抽取
  full/pred_classfication/{backbone}_{split}_class.jsonl 动态候选分类
  ablation/{backbone}_{split}_class_no_dynamic.jsonl     静态候选集消融
  verfication/                                           FGPR 中间产物
```

用 `--output` 可覆盖默认路径；此时需要同步修改 `analysis/` 中的读取路径。

`pred_classfication` 与 `verfication` 的拼写为历史遗留，改名会破坏既有预测文件和
分析脚本的路径解析，因此保留。不同 backbone 使用独立文件名，避免相互覆盖。
