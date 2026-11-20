from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from fgpr.categories import CATEGORIES


def main() -> None:
    parser = argparse.ArgumentParser(description="从 SLARD train 中分层划分 feedback/validation")
    parser.add_argument("--input", default="../dataset/train.json")
    parser.add_argument("--feedback-output", default="../dataset/feedback.json")
    parser.add_argument("--validation-output", default="../dataset/val.json")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 1:
        raise ValueError("validation-ratio 必须在 0 与 1 之间")
    input_path = Path(args.input).resolve()
    feedback_path = Path(args.feedback_output).resolve()
    validation_path = Path(args.validation_output).resolve()
    if not args.overwrite and (feedback_path.exists() or validation_path.exists()):
        raise FileExistsError("输出文件已存在；确认后使用 --overwrite")
    rows = json.loads(input_path.read_text(encoding="utf-8-sig"))
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        category = str(row.get("category", "")).strip()
        if category not in CATEGORIES:
            raise ValueError(f"第 {index + 1} 条记录类别非法：{category!r}")
        grouped[category].append((index, row))
    rng = random.Random(args.seed)
    validation_indices = set()
    for category in CATEGORIES:
        values = list(grouped[category])
        if len(values) < 2:
            raise ValueError(f"类别 {category} 样本不足2条，无法同时进入 feedback 和 validation")
        rng.shuffle(values)
        count = max(1, min(len(values) - 1, round(len(values) * args.validation_ratio)))
        validation_indices.update(index for index, _ in values[:count])
    feedback = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation = [row for index, row in enumerate(rows) if index in validation_indices]
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    feedback_path.write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"feedback={len(feedback)}，validation={len(validation)}，seed={args.seed}")


if __name__ == "__main__":
    main()
