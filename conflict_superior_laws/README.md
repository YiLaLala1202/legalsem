# superior–subordinate law conflict classification（LCR-CN）

本目录是五分类上下位法规范冲突实验。整理时保留了现有脚本、模型实现和输出路径，没有批量移动或重命名文件。

## 任务与标签

| ID | 类别 |
|---:|---|
| 0 | 无冲突 |
| 1 | 职权或责任划分不符 |
| 2 | 概念或定义范围不符 |
| 3 | 处罚幅度或范围不符 |
| 4 | 增设或变更适用条件 |

数据划分位于 `dataset/`：train 4895条、validation 696条、test 1404条。

## 目录结构

```text
dataset/                     数据放置说明（数据本体不随仓库分发）
output/                      预测结果落盘位置（运行后生成）
3.verification/              FGPR：错误聚合、候选Prompt、验证选择、冻结
```

各模型与各消融设置不再各占一个脚本，统一由仓库根目录的 `run.py` 驱动，
模型差异只体现在 `configs/backbones.yaml`。

## 主实验顺序

```bash
# Direct baseline：不做元素抽取，直接五分类
python run.py baseline --dataset lcr-cn --backbone legalone-8b --split test

# Element + Dynamic：先抽取，再按元素收缩候选集分类
python run.py extract  --dataset lcr-cn --backbone qwen3-8b --split test
python run.py classify --dataset lcr-cn --backbone qwen3-8b \
    --elements conflict_superior_laws/output/full/elements/qwen3-8b_test_extract.jsonl --mode dynamic

# 消融：保留元素抽取，但不收缩候选集
python run.py classify --dataset lcr-cn --backbone qwen3-8b \
    --elements conflict_superior_laws/output/full/elements/qwen3-8b_test_extract.jsonl --mode static

# FGPR：开发期提示词优化，冻结后测试（细节见 3.verification/README.md）
python3 3.verification/run_fgpr.py optimize --config 3.verification/config.example.json
python3 3.verification/run_fgpr.py test --config 3.verification/config.example.json \
    --frozen-prompt runs/qwen2.5-7b/frozen_prompt.json
```

「固定 GPT 抽取结果后比较分类模型」这一组设置同样由 `run.py` 表达：
`extract` 用 `--backbone gpt-5.4`，`classify` 换成其他 backbone 并指向该抽取
文件即可。该设置不要与 self-extraction 主实验混合统计。

## 提示词版本

`prompts/base_prompts.json` 保存反馈优化前的简版提示词，是 FGPR 的起点。
反馈优化后的最终提示词不随本仓库分发，因此本仓库运行得到的是方法本身，
而非论文表中的具体 Macro-F1 数值。以同一 schema 提供自备的优化版 bundle，
并用 `--prompt-bundle` 指向它即可复现论文数值。

## 动态候选规则

- 类别0始终保留；
- Subject、Action_Behavior 或 Responsibility 非空时加入类别1；
- Definition 非空时加入类别2；
- Sanction 非空时加入类别3；
- Condition 非空时加入类别4；
- 所有元素为空时回退到五个完整类别。

## 消融对应关系

| 设置 | 元素抽取 | 动态候选 | FGPR |
|---|---:|---:|---:|
| Direct | 否 | 否 | 否 |
| No Dynamic | 是 | 否 | 否 |
| No Verification | 是 | 是 | 否 |
| Full | 是 | 是 | 是 |

## 运行注意事项

1. 不要使用 test 标签生成、筛选或停止 Prompt 优化。
2. `3.verification` 中要求100%的是预测文件对 validation ID 的完整覆盖，不是 candidate recall。
3. 每个 backbone 应使用独立 `work_dir`，避免候选预测相互覆盖。
4. 模型服务地址与凭据均在 `configs/backbones.yaml` 中声明，密钥只从环境变量读取，不存在于源码内。
5. 输出目录中的 `pred_classfication` 和 `verfication` 是既有拼写，为避免破坏历史路径暂不改名。
