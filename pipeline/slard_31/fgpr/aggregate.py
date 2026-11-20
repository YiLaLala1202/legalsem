from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .categories import CATEGORIES
from .common import index_by_id, read_records


def aggregate_errors(
    gold_path: str,
    prediction_path: str,
    representatives_per_cluster: int = 3,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold_by_id = index_by_id(read_records(gold_path), gold_path)
    pred_by_id = index_by_id(read_records(prediction_path), prediction_path)
    missing = sorted(set(gold_by_id) - set(pred_by_id))
    extra = sorted(set(pred_by_id) - set(gold_by_id))
    if missing or extra:
        raise ValueError(f"预测与数据 ID 不一致：缺失={len(missing)}，多出={len(extra)}")

    stage_counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    missing_gold: Counter[str] = Counter()
    gold_counts: Counter[str] = Counter()
    records = []
    candidate_hits = 0

    for item_id, gold_row in gold_by_id.items():
        pred_row = pred_by_id[item_id]
        gold = str(gold_row.get("category", "")).strip()
        pred = str(pred_row.get("pred_category", pred_row.get("category", ""))).strip()
        candidates = pred_row.get("candidate_categories")
        if gold not in CATEGORIES or pred not in CATEGORIES:
            raise ValueError(f"id={item_id} 的类别非法：gold={gold!r}, pred={pred!r}")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"id={item_id} 缺少 candidate_categories")
        candidate_hits += int(gold in candidates)
        if gold not in candidates:
            stage = "element_candidate_construction_error"
            missing_gold[gold] += 1
        elif pred != gold:
            stage = "classification_error"
        else:
            stage = "correct"
        stage_counts[stage] += 1
        gold_counts[gold] += 1
        if pred != gold:
            confusion[(gold, pred)] += 1
        records.append(
            {
                "id": item_id,
                "gold_category": gold,
                "pred_category": pred,
                "candidate_categories": candidates,
                "candidate_contains_gold": gold in candidates,
                "error_stage": stage,
                "cluster_key": f"{stage}|gold={gold}|pred={pred}",
                "title": str(gold_row.get("title", ""))[:200],
                "clause_content": str(gold_row.get("clause_content", ""))[:600],
                "extracted_fields": pred_row.get("extracted_fields", {}),
            }
        )

    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["error_stage"] != "correct":
            clusters[record["cluster_key"]].append(record)
    cluster_rows = [
        {
            "cluster_key": key,
            "count": len(values),
            "error_stage": values[0]["error_stage"],
            "gold_category": values[0]["gold_category"],
            "pred_category": values[0]["pred_category"],
            "representatives": values[:representatives_per_cluster],
        }
        for key, values in sorted(clusters.items(), key=lambda pair: (-len(pair[1]), pair[0]))
    ]
    total = len(records)
    summary = {
        "total_samples": total,
        "stage_counts": dict(stage_counts),
        "candidate_recall": candidate_hits / total if total else 0.0,
        "per_category": [
            {
                "category": category,
                "count": gold_counts[category],
                "missing_from_candidate": missing_gold[category],
            }
            for category in CATEGORIES
        ],
        "confusion_pairs": [
            {"gold_category": gold, "pred_category": pred, "count": count}
            for (gold, pred), count in confusion.most_common()
        ],
        "clusters": cluster_rows,
    }
    return summary, records
