"""Stage implementations: element extraction, conflict classification, baseline.

Each function returns a per-case processor compatible with
:func:`lsem.runner.run_batch`. Output rows keep the field names used by the
analysis scripts in ``analysis/`` so that evaluation code does not need to
change.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from . import io, labels, parsing, prompts
from .backends import BaseBackend

EXTRACT_KEEP_FIELDS = (
    "id",
    "title",
    "level",
    "category",
    "content",
    "high_level_laws",
    "conflict_type",
)


def _gold_label(item: Dict[str, Any]) -> Optional[int]:
    """Map the gold conflict type string onto its integer id."""
    gold = item.get("conflict_type")
    for label_id, name in labels.LABEL_MAP.items():
        if name == gold:
            return label_id
    return item.get("gold_label") if isinstance(item.get("gold_label"), int) else None


# --------------------------------------------------------------------------- #
# Stage 1: semantic element extraction
# --------------------------------------------------------------------------- #
def extract_row(
    backend: BaseBackend,
    bundle: Dict[str, Any],
    item: Dict[str, Any],
    index: int,
) -> Optional[Dict[str, Any]]:
    """Extract elements for one case; ``None`` signals a retryable failure."""
    prompt = prompts.build_extraction_prompt(
        bundle,
        content=item.get("content", ""),
        high_level_laws=item.get("high_level_laws", []),
    )

    try:
        raw = backend.generate(bundle["extraction_system_prompt"], prompt)
    except Exception:  # noqa: BLE001 - backend already retries internally
        return None

    payload = parsing.extract_json_object(raw)
    if payload is None:
        return None

    row = {field: item.get(field) for field in EXTRACT_KEEP_FIELDS}
    row["_index"] = index
    row["extracted_elements"] = parsing.normalize_elements(payload)
    return row


def extract_placeholder(item: Dict[str, Any], index: int, reason: str) -> Dict[str, Any]:
    row = {field: item.get(field) for field in EXTRACT_KEEP_FIELDS}
    row["_index"] = index
    row["extracted_elements"] = parsing.failed_elements(reason)
    return row


def extract_succeeded(row: Dict[str, Any]) -> bool:
    extracted = row.get("extracted_elements")
    if not isinstance(extracted, dict) or "error" in extracted:
        return False
    return isinstance(extracted.get("content_elements"), dict) and isinstance(
        extracted.get("high_level_law_elements"), dict
    )


# --------------------------------------------------------------------------- #
# Stage 2: conflict classification
# --------------------------------------------------------------------------- #
def classify_row(
    backend: BaseBackend,
    bundle: Dict[str, Any],
    item: Dict[str, Any],
    index: int,
    mode: str = "dynamic",
) -> Optional[Dict[str, Any]]:
    """Classify one case using the candidate set implied by its elements."""
    extracted = item.get("extracted_elements") or {}
    content_elements = extracted.get("content_elements", {}) if isinstance(extracted, dict) else {}

    candidates, trace = labels.choose_candidate_labels(content_elements, mode=mode)
    _, explanations = labels.describe_candidates(candidates)

    prompt = prompts.build_classification_prompt(
        bundle,
        content=item.get("content", ""),
        high_level_laws=item.get("high_level_laws", []),
        content_elements=content_elements,
        candidate_ids=candidates,
        candidate_explanations=explanations,
        task_type=trace,
    )

    try:
        raw = backend.generate(bundle["classification_system_prompt"], prompt)
    except Exception:  # noqa: BLE001
        return None

    label_id = parsing.parse_label_id(raw, candidates)
    if label_id is None:
        return None

    row = dict(item)
    row["pred_conflict_label_id"] = label_id
    row["pred_conflict_type"] = labels.LABEL_MAP.get(label_id)
    row["gold_label_id"] = _gold_label(item)
    row["candidate_labels"] = candidates
    row["task_type"] = trace
    row["_index"] = index
    return row


def classify_placeholder(item: Dict[str, Any], index: int, reason: str) -> Dict[str, Any]:
    row = dict(item)
    row["pred_conflict_label_id"] = None
    row["pred_conflict_type"] = None
    row["gold_label_id"] = _gold_label(item)
    row["candidate_labels"] = labels.ALL_LABELS
    row["task_type"] = f"failed:{reason}"
    row["_index"] = index
    return row


def classify_succeeded(row: Dict[str, Any]) -> bool:
    return isinstance(row.get("pred_conflict_label_id"), int)


# --------------------------------------------------------------------------- #
# Baseline: end-to-end classification without element extraction
# --------------------------------------------------------------------------- #
def _baseline_inputs(item: Dict[str, Any]) -> tuple:
    """Accept either pre-paired fields or the raw subordinate/superior pair."""
    if item.get("input1") is not None and item.get("input2") is not None:
        return item["input1"], item["input2"]

    content = item.get("content", "")
    laws = prompts.format_high_level_laws(item.get("high_level_laws", []))
    return content, laws


def baseline_row(
    backend: BaseBackend,
    item: Dict[str, Any],
    index: int,
) -> Optional[Dict[str, Any]]:
    """Direct classification with all five labels available at once."""
    subordinate, superior = _baseline_inputs(item)
    label_text = prompts.label_lines_for(labels.ALL_LABELS, labels.LABEL_MAP)
    user_prompt = prompts.build_baseline_prompt(subordinate, superior or "", label_text)

    system_prompt = (
        "你是一个法律冲突分类器。"
        "必须严格输出合法 JSON，不要解释，不要输出多余文字。"
    )
    try:
        raw = backend.generate(system_prompt, user_prompt)
    except Exception:  # noqa: BLE001
        return None

    label_id = parsing.parse_label_id(raw, labels.ALL_LABELS)
    if label_id is None:
        return None

    row = dict(item)
    row["pred_conflict_label_id"] = label_id
    row["pred_conflict_type"] = labels.LABEL_MAP.get(label_id)
    row["gold_label_id"] = _gold_label(item)
    row["_index"] = index
    return row


def baseline_placeholder(item: Dict[str, Any], index: int, reason: str) -> Dict[str, Any]:
    row = dict(item)
    row["pred_conflict_label_id"] = None
    row["pred_conflict_type"] = None
    row["gold_label_id"] = _gold_label(item)
    row["_index"] = index
    return row


def baseline_succeeded(row: Dict[str, Any]) -> bool:
    return isinstance(row.get("pred_conflict_label_id"), int)
