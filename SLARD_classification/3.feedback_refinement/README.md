# SLARD-31 反馈引导 Prompt 优化

本目录补充 SLARD-31 尚未实现的开发阶段反馈流程，不修改原来的 `baseline`、`1.abstract`、`2.dynamic_classification` 和消融脚本。

## 流程

```text
train.json
  → 按31类分层划分 feedback.json / val.json
  → 当前抽取 Prompt 在 feedback 上抽取31类证据
  → 非空证据字段构造动态候选类别
  → 当前分类 Prompt 预测主题
  → 程序聚合候选遗漏与类别混淆
  → 固定外部大模型生成 K 个中文候选 Prompt
  → 目标模型在 val 上运行当前 Prompt 与全部候选
  → Macro-F1 严格提升才接受
  → frozen_prompt.json
  → 冻结后单独运行 test.json
```

## 第一步：划分开发数据

SLARD 当前只有 `train.json` 和 `test.json`。不能使用 test 选择 Prompt，因此先执行：

```bash
python3 prepare_dev_split.py \
  --input ../dataset/train.json \
  --feedback-output ../dataset/feedback.json \
  --validation-output ../dataset/val.json \
  --validation-ratio 0.15 \
  --seed 2026
```

程序按 `category` 分层抽样，并保证每个类别在 feedback 和 validation 中至少保留一条。已有输出默认不会覆盖。

## 第二步：配置模型

复制 `config.example.json`，修改目标模型名称、OpenAI-compatible 服务地址和 `work_dir`。密钥通过环境变量提供：

```bash
export SILICONFLOW_API_KEY='...'
export VLLM_API_KEY='...'
```

本地 Qwen、DeepSeek、LegalOne 可以使用 vLLM OpenAI server；其他模型也可以把 `target_runner.command` 换成自己的运行脚本，只需接收 input、prompt bundle 和 output 三个路径。

## 第三步：优化并冻结

```bash
python3 run_fgpr.py optimize --config config.qwen2.5-7b.json
```

错误划分：

- 真实类别不在 `candidate_categories`：`element_candidate_construction_error`；
- 真实类别在候选集中但预测错误：`classification_error`。

每轮当前 Prompt 和 K 个候选都在同一个 validation split 上运行。候选只有在 Macro-F1 严格提升时才被接受。最终文件为：

```text
<work_dir>/frozen_prompt.json
```

## 第四步：冻结后测试

```bash
python3 run_fgpr.py test \
  --config config.qwen2.5-7b.json \
  --frozen-prompt runs/qwen2.5-7b/frozen_prompt.json \
  --test-data ../dataset/test.json \
  --output runs/qwen2.5-7b/test_predictions.jsonl \
  --metrics runs/qwen2.5-7b/test_metrics.json
```

测试标签只在推理完成后用于计算指标，不参与错误聚合、候选生成、Prompt 选择或停止判断。

## 说明

- 抽取字段就是31个主题类别；非空字段直接形成动态候选集。
- 抽取全空时回退到全部31类，避免无法分类。
- 要求100%的只是验证预测文件完整性，不是 candidate recall。
- `base_prompts.json` 使用中文提示词；固定 Prompt optimizer 的系统指令和候选生成要求也全部使用中文。
