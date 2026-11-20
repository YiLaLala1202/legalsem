from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .common import (
    GOLD_KEYS,
    PRED_KEYS,
    get_candidates,
    get_first_label,
    index_by_id,
    iter_jsonl,
    read_jsonl,
    truncate_text,
)


def aggregate_errors(
    gold_path: str,
    prediction_path: str,
    label_map: dict[int, str],
    representatives_per_cluster: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Join predictions to gold data and build deterministic, dataset-level feedback."""
    prediction_rows = read_jsonl(prediction_path)
    pred_by_id = index_by_id(prediction_rows, prediction_path)

    records: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    gold_counts: Counter[int] = Counter()
    pred_counts: Counter[int] = Counter()
    confusion: Counter[tuple[int, int]] = Counter()
    missing_gold_by_label: Counter[int] = Counter()
    candidate_known = 0
    candidate_hits = 0

    seen_gold_ids: set[str] = set()
    missing_ids: list[str] = []
    for gold_row in iter_jsonl(gold_path):
        if gold_row.get("id") is None:
            raise ValueError(f"A row in {gold_path} has no id")
        item_id = str(gold_row["id"])
        if item_id in seen_gold_ids:
            raise ValueError(f"Duplicate id={item_id!r} in {gold_path}")
        seen_gold_ids.add(item_id)
        pred_row = pred_by_id.get(item_id)
        if pred_row is None:
            missing_ids.append(item_id)
            continue
        gold = get_first_label(gold_row, GOLD_KEYS, label_map)
        pred = get_first_label(pred_row, PRED_KEYS, label_map)
        if gold is None:
            raise ValueError(f"Cannot read gold label for id={item_id}")
        if pred is None:
            raise ValueError(f"Cannot read predicted label for id={item_id}")

        candidates = get_candidates(pred_row, label_map)
        if candidates is not None:
            candidate_known += 1
            if gold in candidates:
                candidate_hits += 1

        if candidates is not None and gold not in candidates:
            stage = "element_candidate_construction_error"
            missing_gold_by_label[gold] += 1
        elif pred != gold:
            stage = "classification_error"
        else:
            stage = "correct"

        stage_counts[stage] += 1
        gold_counts[gold] += 1
        pred_counts[pred] += 1
        if pred != gold:
            confusion[(gold, pred)] += 1

        merged_elements = pred_row.get("extracted_elements") or gold_row.get("extracted_elements") or {}
        record = {
            "id": item_id,
            "gold_label": gold,
            "gold_name": label_map[gold],
            "pred_label": pred,
            "pred_name": label_map[pred],
            "candidate_labels": candidates,
            "candidate_contains_gold": None if candidates is None else gold in candidates,
            "error_stage": stage,
            "cluster_key": f"{stage}|gold={gold}|pred={pred}",
            "content": truncate_text(gold_row.get("content")),
            "high_level_laws": [
                truncate_text(value, 400) for value in gold_row.get("high_level_laws", [])[:3]
            ],
            "extracted_elements": merged_elements,
        }
        records.append(record)

    extra_ids = sorted(set(pred_by_id) - seen_gold_ids)
    if missing_ids or extra_ids:
        raise ValueError(
            "Prediction/gold id mismatch: "
            f"missing_predictions={len(missing_ids)}, unknown_predictions={len(extra_ids)}"
        )

    clustered: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["error_stage"] != "correct":
            clustered[record["cluster_key"]].append(record)

    cluster_summaries = []
    for key, values in sorted(clustered.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        cluster_summaries.append(
            {
                "cluster_key": key,
                "count": len(values),
                "error_stage": values[0]["error_stage"],
                "gold_label": values[0]["gold_label"],
                "pred_label": values[0]["pred_label"],
                "representatives": values[:representatives_per_cluster],
            }
        )

    total = len(records)
    summary = {
        "schema_version": "fgpr.error_aggregation.v1",
        "total_samples": total,
        "correct": stage_counts["correct"],
        "errors": total - stage_counts["correct"],
        "stage_counts": dict(stage_counts),
        "candidate_set_available": candidate_known,
        "candidate_recall": (candidate_hits / candidate_known) if candidate_known else None,
        "per_gold_label": [
            {
                "label": label,
                "name": label_map[label],
                "count": gold_counts[label],
                "missing_from_candidate": missing_gold_by_label[label],
            }
            for label in sorted(label_map)
        ],
        "prediction_distribution": [
            {"label": label, "name": label_map[label], "count": pred_counts[label]}
            for label in sorted(label_map)
        ],
        "confusion_pairs": [
            {
                "gold_label": gold,
                "gold_name": label_map[gold],
                "pred_label": pred,
                "pred_name": label_map[pred],
                "count": count,
            }
            for (gold, pred), count in confusion.most_common()
        ],
        "clusters": cluster_summaries,
    }
    return summary, records
