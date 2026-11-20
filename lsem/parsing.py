"""Tolerant parsing of model output.

LLM responses are not guaranteed to be clean JSON: they may be wrapped in
Markdown fences, prefixed with prose, or truncated. The helpers below recover
the payload where possible and otherwise return ``None`` so the caller can
retry, matching the behaviour of the original experiment scripts.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

ELEMENT_FIELDS: Sequence[str] = (
    "Subject",
    "Action_Behavior",
    "Condition",
    "Responsibility",
    "Sanction",
    "Definition",
)

ELEMENT_PARTS: Sequence[str] = ("content_elements", "high_level_law_elements")


def empty_elements() -> Dict[str, List[str]]:
    """Return the canonical empty element dictionary."""
    return {field: [] for field in ELEMENT_FIELDS}


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort recovery of a JSON object from a free-form response."""
    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1]

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def normalize_elements(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Coerce an arbitrary payload into the expected element schema.

    Missing or malformed parts are replaced with empty arrays so downstream
    stages never have to guard against ``None``.
    """
    result: Dict[str, Any] = payload if isinstance(payload, dict) else {}

    for part in ELEMENT_PARTS:
        section = result.get(part)
        if not isinstance(section, dict):
            section = {}
            result[part] = section

        for field in ELEMENT_FIELDS:
            value = section.get(field, [])
            if value is None:
                value = []
            elif isinstance(value, str):
                value = [value]
            elif not isinstance(value, list):
                value = [str(value)]
            section[field] = value

    return result


def failed_elements(message: str = "LLM call or JSON parse failed") -> Dict[str, Any]:
    """Element payload used as a placeholder for rows that never succeeded."""
    payload: Dict[str, Any] = {part: empty_elements() for part in ELEMENT_PARTS}
    payload["error"] = message
    return payload


_NUMBER_PATTERN = re.compile(r"-?\d+")


def parse_label_id(text: str, candidates: Sequence[int]) -> Optional[int]:
    """Extract a single label id from a model response.

    Only identifiers inside the candidate set are accepted; the first match
    wins, which mirrors the strict one-token output format used in training
    and evaluation.
    """
    if not text:
        return None

    allowed = {int(c) for c in candidates}

    stripped = text.strip()
    try:
        value = int(stripped)
        return value if value in allowed else None
    except ValueError:
        pass

    for match in _NUMBER_PATTERN.finditer(stripped):
        value = int(match.group())
        if value in allowed:
            return value

    return None
