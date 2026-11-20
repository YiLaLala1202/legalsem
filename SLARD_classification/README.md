# SLARD-31 法律条文主题分类

本目录是从 SLARD 数据构建的31类法律条文主题分类实验。整理时保留现有代码和结果路径，只新增说明文档与 `3.feedback_refinement`。

## 31个类别

环境保护、民政、能源、教育、人权、牧业、工业管理、水利、法制工作、知识产权、农业、旅游、土地、林业、房地产、体育、工商管理、文物文史、合同、商贸物资、营商环境优化、国家安全、企业、广告、劳动工会、测绘、军事、渔业、价格、建筑业、文化。

## 目录结构

```text
dataset/                     数据放置说明（数据本体不随仓库分发）
output/                      预测结果落盘位置（运行后生成）
3.feedback_refinement/       FGPR：错误聚合、候选Prompt、验证选择
```

各模型与各消融设置不再各占一个脚本，统一由仓库根目录的 `run.py` 驱动，
任务范式由 `--dataset slard-31` 自动切换为 31 类主题分类。

## 数据

- `train.json`：6555条。
- `test.json`：1639条，必须保持为 held-out test。
- `feedback.json`、`val.json`：运行 `3.feedback_refinement/prepare_dev_split.py` 后从 train 按类别分层产生。

不能使用 `test.json` 生成或选择 Prompt。

## 主实验顺序

```bash
# Direct baseline：不做证据抽取，直接 31 分类
python run.py baseline --dataset slard-31 --backbone qwen3-8b --split test

# Evidence + Dynamic：先抽取主题证据，再按非空类别收缩候选集
python run.py extract  --dataset slard-31 --backbone qwen3-8b --split test
python run.py classify --dataset slard-31 --backbone qwen3-8b \
    --elements SLARD_classification/output/full/elements/qwen3-8b_test_extract.jsonl --mode dynamic

# 消融：保留证据抽取，但不收缩候选集
python run.py classify --dataset slard-31 --backbone qwen3-8b \
    --elements SLARD_classification/output/full/elements/qwen3-8b_test_extract.jsonl --mode static

# FGPR
python3 3.feedback_refinement/prepare_dev_split.py
python3 3.feedback_refinement/run_fgpr.py optimize --config 3.feedback_refinement/config.example.json
python3 3.feedback_refinement/run_fgpr.py test --config 3.feedback_refinement/config.example.json \
    --frozen-prompt runs/qwen2.5-7b/frozen_prompt.json
```

## 动态候选规则

抽取结果包含31个主题字段。所有非空字段直接组成 `candidate_categories`；若全部为空，则回退到完整31类。

## 消融对应关系

| 设置 | 主题证据抽取 | 动态候选 | FGPR |
|---|---:|---:|---:|
| Direct | 否 | 否 | 否 |
| No Dynamic | 是 | 否 | 否 |
| No Verification | 是 | 是 | 否 |
| Full | 是 | 是 | 是 |

## 运行注意事项

1. 六个 backbone 的配置和输出目录应相互独立。
2. 验证集 Macro-F1 必须固定在全部31类上计算，不能只平均验证集中表现较好的类别。
3. 验证预测必须覆盖 validation 的全部 ID；candidate recall 本身不要求100%。
4. 模型服务地址与凭据均在 `configs/backbones.yaml` 中声明，密钥只从环境变量读取。
5. `pred_classfication`、`no_verificaion` 等既有拼写暂不修改，以免破坏历史路径。
