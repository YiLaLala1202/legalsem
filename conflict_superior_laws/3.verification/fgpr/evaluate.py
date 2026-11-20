from __future__ import annotations

import math
from typing import Any

from .common import PRED_KEYS, get_first_label, index_by_id, read_jsonl


def evaluate_predictions(
    gold_path: str,
    prediction_path: str,
    label_map: dict[int, str],
) -> dict[str, Any]:
    gold_by_id = index_by_id(read_jsonl(gold_path), gold_path)
    pred_by_id = index_by_id(read_jsonl(prediction_path), prediction_path)
    missing = sorted(set(gold_by_id) - set(pred_by_id))
    extra = sorted(set(pred_by_id) - set(gold_by_id))
    if missing or extra:
        raise ValueError(
            f"Validation coverage must be 100%: missing={len(missing)}, extra={len(extra)}"
        )

    name_to_id = {name: label for label, name in label_map.items()}
    y_true: list[int] = []
    y_pred: list[int] = []
    for item_id, row in gold_by_id.items():
        raw_gold = row.get("gold_label_id", row.get("conflict_type"))
        if isinstance(raw_gold, str) and raw_gold in name_to_id:
            gold = name_to_id[raw_gold]
        else:
            try:
                gold = int(raw_gold)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid gold label for id={item_id}: {raw_gold!r}") from exc
        pred = get_first_label(pred_by_id[item_id], PRED_KEYS, label_map)
        if gold not in label_map or pred not in label_map:
            raise ValueError(f"Invalid label for id={item_id}: gold={gold!r}, pred={pred!r}")
        y_true.append(gold)
        y_pred.append(pred)

    per_class = []
    f1_values = []
    for label in sorted(label_map):
        tp = sum(gold == label and pred == label for gold, pred in zip(y_true, y_pred))
        fp = sum(gold != label and pred == label for gold, pred in zip(y_true, y_pred))
        fn = sum(gold == label and pred != label for gold, pred in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append(
            {
                "label": label,
                "name": label_map[label],
                "support": sum(gold == label for gold in y_true),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    correct = sum(gold == pred for gold, pred in zip(y_true, y_pred))
    macro_f1 = sum(f1_values) / len(f1_values)
    if not math.isfinite(macro_f1):
        raise ValueError("Macro-F1 is not finite")
    return {
        "schema_version": "fgpr.metrics.v1",
        "samples": len(y_true),
        "coverage": 1.0,
        "accuracy": correct / len(y_true) if y_true else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
