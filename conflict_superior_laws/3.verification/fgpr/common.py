from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_LABEL_MAP = {
    0: "无冲突",
    1: "职权或责任划分不符",
    2: "概念或定义范围不符",
    3: "处罚幅度或范围不符",
    4: "增设或变更适用条件",
}

ELEMENT_FIELDS = (
    "Subject",
    "Action_Behavior",
    "Condition",
    "Responsibility",
    "Sanction",
    "Definition",
)

GOLD_KEYS = ("gold_label_id", "label_id", "conflict_type", "gold_label")
PRED_KEYS = (
    "pred_conflict_label_id",
    "pred_label_id",
    "prediction",
    "pred_label",
)
CANDIDATE_KEYS = ("candidate_labels", "step3_candidate_labels", "candidates")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, target)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, target)


def label_map_from_config(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("labels") or DEFAULT_LABEL_MAP
    return {int(key): str(value) for key, value in raw.items()}


def normalize_label(value: Any, label_map: dict[int, str]) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in label_map else None
    if isinstance(value, float) and value.is_integer():
        integer = int(value)
        return integer if integer in label_map else None
    text = str(value).strip()
    if text in {str(label) for label in label_map}:
        return int(text)
    reverse = {name: label for label, name in label_map.items()}
    if text in reverse:
        return reverse[text]
    match = re.search(r"(?<!\d)(\d+)(?!\d)", text)
    if match and int(match.group(1)) in label_map:
        return int(match.group(1))
    return None


def get_first_label(
    row: dict[str, Any], keys: Iterable[str], label_map: dict[int, str]
) -> int | None:
    for key in keys:
        if key in row:
            label = normalize_label(row.get(key), label_map)
            if label is not None:
                return label
    return None


def get_candidates(row: dict[str, Any], label_map: dict[int, str]) -> list[int] | None:
    for key in CANDIDATE_KEYS:
        raw = row.get(key)
        if not isinstance(raw, (list, tuple, set)):
            continue
        normalized = [normalize_label(value, label_map) for value in raw]
        return sorted({value for value in normalized if value is not None})
    return None


def index_by_id(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("id") is None:
            raise ValueError(f"A row in {source} has no id")
        item_id = str(row["id"])
        if item_id in indexed:
            raise ValueError(f"Duplicate id={item_id!r} in {source}")
        indexed[item_id] = row
    return indexed


def truncate_text(value: Any, limit: int = 600) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def replace_tokens(template: str, values: dict[str, str]) -> str:
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace(f"[[{name}]]", value)
    unresolved = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", rendered)))
    if unresolved:
        raise ValueError(f"Unresolved prompt tokens: {', '.join(unresolved)}")
    return rendered


def validate_prompt_bundle(bundle: dict[str, Any]) -> None:
    extraction = bundle.get("extraction_prompt")
    classification = bundle.get("classification_prompt")
    if not isinstance(extraction, str) or not extraction.strip():
        raise ValueError("Prompt bundle needs a non-empty extraction_prompt")
    if not isinstance(classification, str) or not classification.strip():
        raise ValueError("Prompt bundle needs a non-empty classification_prompt")
    required_extraction = ("[[CONTENT]]", "[[HIGH_LEVEL_LAWS]]")
    required_classification = (
        "[[CONTENT]]",
        "[[HIGH_LEVEL_LAWS]]",
        "[[CONTENT_ELEMENTS]]",
        "[[CANDIDATE_LABELS]]",
        "[[CANDIDATE_EXPLANATIONS]]",
    )
    missing = [token for token in required_extraction if token not in extraction]
    missing += [token for token in required_classification if token not in classification]
    if missing:
        raise ValueError(f"Prompt bundle is missing required tokens: {', '.join(missing)}")


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def blind_dataset(source: str | Path, destination: str | Path) -> None:
    """Remove supervision-only fields before a target-model runner sees a split."""
    blocked = {
        "conflict_type",
        "conflict_description",
        "gold_label",
        "gold_label_id",
        "label",
        "label_id",
        "target",
        "golden_content",
        "distractors",
    }
    def blinded_rows() -> Iterator[dict[str, Any]]:
        for row in iter_jsonl(source):
            yield {key: value for key, value in row.items() if key not in blocked}

    write_jsonl(destination, blinded_rows())
