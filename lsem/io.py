"""JSONL I/O helpers with crash-safe, resumable output.

Every experiment stage streams one JSON object per line and must survive being
interrupted in the middle of a long run. The helpers here implement the same
contract used throughout the paper's experiments:

1. Results are appended as soon as they are produced.
2. On restart, already-successful rows are skipped.
3. At the end the file is rewritten in the original input order so that
   downstream scripts can align predictions with gold labels positionally.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    """Overwrite ``path`` with ``items``, one JSON object per line."""
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    """Append a single row and flush immediately."""
    ensure_parent(path)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()


def ensure_parent(path: str) -> None:
    """Create the parent directory of ``path`` when it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_existing(path: str, index_key: str = "_index") -> Dict[int, Dict[str, Any]]:
    """Load previously written rows keyed by their positional index."""
    if not os.path.exists(path):
        return {}

    existing: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = row.get(index_key)
            if isinstance(index, int):
                existing[index] = row
    return existing


def rewrite_in_order(
    rows: Dict[int, Dict[str, Any]],
    items: List[Dict[str, Any]],
    path: str,
    placeholder: Callable[[Dict[str, Any], int], Dict[str, Any]],
) -> None:
    """Rewrite ``path`` in input order, filling gaps with ``placeholder``."""
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        for index, item in enumerate(items):
            row = rows.get(index) or placeholder(item, index)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_lines(path: str) -> int:
    """Count non-empty lines without loading the file into memory."""
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_path(value: Optional[str], root: str) -> Optional[str]:
    """Resolve ``value`` against ``root`` unless it is already absolute."""
    if value is None:
        return None
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)
