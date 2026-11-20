from __future__ import annotations

from typing import Any

from .categories import CATEGORIES
from .common import index_by_id, read_records


def evaluate_predictions(gold_path: str, prediction_path: str) -> dict[str, Any]:
    gold = index_by_id(read_records(gold_path), gold_path)
    pred = index_by_id(read_records(prediction_path), prediction_path)
    missing = set(gold) - set(pred)
    extra = set(pred) - set(gold)
    if missing or extra:
        raise ValueError(f"验证集预测必须完整：缺失={len(missing)}，多出={len(extra)}")
    pairs = []
    for item_id, row in gold.items():
        true_category = str(row.get("category", "")).strip()
        pred_category = str(pred[item_id].get("pred_category", pred[item_id].get("category", ""))).strip()
        if true_category not in CATEGORIES or pred_category not in CATEGORIES:
            raise ValueError(f"id={item_id} 的类别非法")
        pairs.append((true_category, pred_category))

    per_class = []
    f1_values = []
    for category in CATEGORIES:
        tp = sum(gold == category and guess == category for gold, guess in pairs)
        fp = sum(gold != category and guess == category for gold, guess in pairs)
        fn = sum(gold == category and guess != category for gold, guess in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "category": category,
                "support": sum(gold == category for gold, _ in pairs),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "samples": len(pairs),
        "coverage": 1.0,
        "accuracy": sum(gold == guess for gold, guess in pairs) / len(pairs) if pairs else 0.0,
        "macro_f1": sum(f1_values) / len(CATEGORIES),
        "per_class": per_class,
    }
