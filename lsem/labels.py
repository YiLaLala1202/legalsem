"""Conflict label space and semantic-element-driven candidate reduction.

This module is the implementation of the paper's central inference-time
mechanism: the set of classes offered to the model is narrowed according to
which semantic elements were actually extracted from the subordinate clause.

The mapping is fixed a priori (it is *not* learned from the training split),
which is what makes the control layer inspectable and auditable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

LABEL_MAP: Dict[int, str] = {
    0: "无冲突",
    1: "职权或责任划分不符",
    2: "概念或定义范围不符",
    3: "处罚幅度或范围不符",
    4: "增设或变更适用条件",
}

LABEL_EXPLAIN: Dict[int, str] = {
    0: "下位法与上位法含义一致，无削弱、扩大、变更、增设条件等情况",
    1: "下位法改变主体职责、权限、义务、责任范围",
    2: "下位法改变概念、定义、对象范围",
    3: "下位法改变处罚种类、处罚对象或处罚范围",
    4: "下位法增设、删除、放宽或收紧适用条件、前提、例外情形",
}

ALL_LABELS: List[int] = sorted(LABEL_MAP)

# Which element triggers which candidate class.
# A non-empty element set corresponds to one or more admissible classes.
ELEMENT_TO_LABEL: Dict[str, Tuple[int, ...]] = {
    "Subject": (1,),
    "Action_Behavior": (1,),
    "Responsibility": (1,),
    "Definition": (2,),
    "Sanction": (3,),
    "Condition": (4,),
}

NO_CONFLICT_LABEL = 0


def is_non_empty(value: Any) -> bool:
    """True when ``value`` is a non-empty list."""
    return isinstance(value, list) and len(value) > 0


def choose_candidate_labels(
    content_elements: Any,
    mode: str = "dynamic",
) -> Tuple[List[int], str]:
    """Return the admissible label set and a trace of the triggering elements.

    Parameters
    ----------
    content_elements:
        The ``content_elements`` dictionary produced by the extraction stage.
    mode:
        ``"dynamic"`` applies the element-to-label reduction; ``"static"``
        always returns the full label space and is used by the ablation study.

    Returns
    -------
    (candidates, trace)
        ``candidates`` is a sorted list of label ids. ``trace`` records which
        element fields were non-empty, or ``"all_empty"`` when the reduction
        could not fire; the trace is stored next to each prediction so the
        decision path remains auditable after the fact.
    """
    if mode == "static":
        return list(ALL_LABELS), "static"

    if not isinstance(content_elements, dict):
        return list(ALL_LABELS), "all_empty"

    non_empty_fields = [key for key, value in content_elements.items() if is_non_empty(value)]
    if not non_empty_fields:
        return list(ALL_LABELS), "all_empty"

    candidates = {NO_CONFLICT_LABEL}
    for field in non_empty_fields:
        candidates.update(ELEMENT_TO_LABEL.get(field, ()))

    return sorted(candidates), ",".join(non_empty_fields)


def describe_candidates(candidates: Sequence[int]) -> Tuple[str, str]:
    """Render the candidate id list and their explanations for the prompt."""
    ids = ", ".join(str(int(c)) for c in candidates)
    explanations = "\n".join(
        f"{int(c)}: {LABEL_MAP[int(c)]} - {LABEL_EXPLAIN[int(c)]}" for c in candidates
    )
    return ids, explanations
