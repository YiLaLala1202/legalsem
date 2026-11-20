from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        value = read_json(source)
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError(f"{source} 必须是 JSON 对象数组")
        return value
    rows = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_no} 不是 JSON 对象")
            rows.append(row)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, target)


def index_by_id(rows: Iterable[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        if row.get("id") is None:
            raise ValueError(f"{source} 中存在缺少 id 的记录")
        item_id = str(row["id"])
        if item_id in result:
            raise ValueError(f"{source} 中存在重复 id={item_id}")
        result[item_id] = row
    return result


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def replace_tokens(template: str, values: dict[str, str]) -> str:
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace(f"[[{name}]]", value)
    unresolved = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", rendered)))
    if unresolved:
        raise ValueError(f"提示词中仍有未替换占位符：{', '.join(unresolved)}")
    return rendered


def validate_prompt_bundle(bundle: dict[str, Any]) -> None:
    extraction = bundle.get("extraction_prompt")
    classification = bundle.get("classification_prompt")
    if not isinstance(extraction, str) or not extraction.strip():
        raise ValueError("缺少 extraction_prompt")
    if not isinstance(classification, str) or not classification.strip():
        raise ValueError("缺少 classification_prompt")
    required_extraction = (
        "[[TITLE]]", "[[CONTENT]]", "[[CATEGORY_DEFINITIONS]]", "[[EMPTY_SCHEMA]]"
    )
    required_classification = (
        "[[TITLE]]", "[[CONTENT]]", "[[EXTRACTED_FIELDS]]", "[[CANDIDATE_CATEGORIES]]"
    )
    missing = [token for token in required_extraction if token not in extraction]
    missing += [token for token in required_classification if token not in classification]
    if missing:
        raise ValueError(f"提示词缺少必要占位符：{', '.join(missing)}")


def blind_dataset(source: str | Path, destination: str | Path) -> None:
    blocked = {"category", "gold_category", "gold_label", "label", "target"}
    rows = (
        {key: value for key, value in row.items() if key not in blocked}
        for row in read_records(source)
    )
    write_jsonl(destination, rows)
