# FGPR 实验补充代码

这套代码只实现论文需要的主流程：

```text
反馈集预测
  → 程序聚合错误
  → 固定外部大模型生成 K 个 Prompt 候选
  → 目标模型在验证集逐一运行
  → 按 Macro-F1 接受最佳候选
  → 保存 frozen_prompt.json
  → 冻结后单独运行测试集
```

## Prompt 的关系

- `baseline_prompt.json`：最初的 Direct Prompt 五分类提示，仅用于 baseline 对照。
- `base_prompts.json`：反馈优化前的简版 extraction Prompt 和 dynamic classification Prompt，是 FGPR 的起点，也是本仓库唯一分发的 bundle。
- 反馈优化后的最终 Prompt 不随仓库分发；FGPR 的目标就是从训练反馈生成更明确的要素定义和类别边界，并由验证集选择最终版本。若需复现论文数值，请以同 schema 自备优化后的 bundle。

## 主要文件

- `run_fgpr.py`：运行 Prompt 优化，或使用冻结 Prompt 运行测试集。
- `run_target_openai.py`：OpenAI-compatible 目标模型运行器，可连接远程 API 或 vLLM server。
- `fgpr/aggregate.py`：聚合候选构造错误与分类错误。
- `fgpr/optimizer.py`：调用固定外部模型生成 K 个完整 Prompt 候选。
- `fgpr/evaluate.py`：计算固定五类 Macro-F1。
- `fgpr/orchestrator.py`：执行迭代选择、早停和 Prompt 冻结。

## 错误划分

```text
gold 不在 candidate_labels
    → element/candidate construction error

gold 在 candidate_labels 中，但预测错误
    → classification error
```

这里不把第一类严格称为 extraction error，因为它也可能来自候选映射规则。

## 配置

复制 `config.example.json`，为每个 backbone 修改：

- `target_model_name`；
- `target_runner.command` 中的模型名和服务地址；
- `work_dir`。

API Key 只通过环境变量提供，不写进源码。

```bash
export SILICONFLOW_API_KEY='...'
export VLLM_API_KEY='...'
```

## 开发阶段优化

```bash
python3 run_fgpr.py optimize --config config.qwen2.5-7b.json
```

每一轮执行：

1. 用当前 Prompt 在 feedback/train 集上预测；
2. 聚合错误和高频混淆；
3. 固定外部模型生成 K 个候选；
4. 当前 Prompt 与 K 个候选分别在 validation 集上运行；
5. 只有最佳候选 Macro-F1 严格提升时才接受；
6. 达到最大轮数或连续若干轮无提升后停止。

最终 Prompt 保存为：

```text
<work_dir>/frozen_prompt.json
```

## 冻结后测试

```bash
python3 run_fgpr.py test \
  --config config.qwen2.5-7b.json \
  --frozen-prompt runs/qwen2.5-7b/frozen_prompt.json \
  --test-data ../dataset/test.jsonl \
  --output runs/qwen2.5-7b/test_predictions.jsonl \
  --metrics runs/qwen2.5-7b/test_metrics.json
```

优化配置中没有 test 路径。测试标签不会进入抽取、候选生成或 Prompt 选择。

## 关于“100% 覆盖率”

要求 100% 的只是预测文件完整性：验证集每条样本都必须产生一条合法预测，否则不同候选无法公平比较。动态候选集的 candidate recall 不要求达到 100%，真实标签未进入候选集的样本会正常计为错误。

## 关于 gold 泄露

正式 FGPR 选择中，目标模型不会收到 gold 字段：优化器只看到训练反馈的聚合结果，候选 Prompt 在验证集上按 Macro-F1 评估。逐样本读取 gold、强制加入正确候选并提示“正确类别”的做法只可能作为调试手段，不能用于任何会产生论文数值的流程。
