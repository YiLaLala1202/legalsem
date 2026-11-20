"""Prompt bundle loading and placeholder filling.

A prompt bundle is a JSON document holding the system prompts and the two
task templates (element extraction and conflict classification). Templates use
``[[PLACEHOLDER]]`` markers rather than ``str.format`` fields, because the
templates themselves contain literal JSON braces.

Only the *initial* bundle (``candidate_id = pre_feedback_original_prompt``) is
distributed with this repository; see ``prompts/README.md``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

CONTENT = "[[CONTENT]]"
HIGH_LEVEL_LAWS = "[[HIGH_LEVEL_LAWS]]"
CANDIDATE_LABELS = "[[CANDIDATE_LABELS]]"
CANDIDATE_EXPLANATIONS = "[[CANDIDATE_EXPLANATIONS]]"
CONTENT_ELEMENTS = "[[CONTENT_ELEMENTS]]"
TASK_TYPE = "[[TASK_TYPE]]"

REQUIRED_KEYS = (
    "extraction_system_prompt",
    "extraction_prompt",
    "classification_system_prompt",
    "classification_prompt",
)


def load_bundle(path: str) -> Dict[str, Any]:
    """Load a prompt bundle and verify that all required keys are present."""
    with open(path, "r", encoding="utf-8") as handle:
        bundle = json.load(handle)

    missing = [key for key in REQUIRED_KEYS if key not in bundle]
    if missing:
        raise ValueError(f"prompt bundle is missing required keys: {missing}")

    return bundle


def format_high_level_laws(laws: Any) -> str:
    """Render the superior-law list as a numbered block."""
    if not laws:
        return ""
    if isinstance(laws, str):
        return laws
    return "\n".join(f"{i + 1}. {law}" for i, law in enumerate(laws))


def build_extraction_prompt(
    bundle: Dict[str, Any],
    content: str,
    high_level_laws: Any,
) -> str:
    """Fill the extraction template for a single case."""
    return (
        bundle["extraction_prompt"]
        .replace(CONTENT, content or "")
        .replace(HIGH_LEVEL_LAWS, format_high_level_laws(high_level_laws))
    )


def build_classification_prompt(
    bundle: Dict[str, Any],
    content: str,
    high_level_laws: Any,
    content_elements: Dict[str, Any],
    candidate_ids: Sequence[int],
    candidate_explanations: str,
    task_type: str,
) -> str:
    """Fill the classification template for a single case."""
    return (
        bundle["classification_prompt"]
        .replace(CONTENT, content or "")
        .replace(HIGH_LEVEL_LAWS, format_high_level_laws(high_level_laws))
        .replace(CONTENT_ELEMENTS, json.dumps(content_elements, ensure_ascii=False, indent=2))
        .replace(CANDIDATE_LABELS, ", ".join(str(int(c)) for c in candidate_ids))
        .replace(CANDIDATE_EXPLANATIONS, candidate_explanations)
        .replace(TASK_TYPE, task_type)
    )


def build_baseline_prompt(content: str, high_level_laws: Any, label_lines: str) -> str:
    """Prompt used by the end-to-end baseline, which skips element extraction."""
    laws = format_high_level_laws(high_level_laws)
    return f"""你是法律规范冲突分类助手。

任务：
比较下位法 content 与上位法 high_level_laws，
判断二者针对同一规范事项是否存在实质性差异。

候选类别：
{label_lines}

【下位法 content】
{content}

【上位法 high_level_laws】
{laws}

输出要求：
只输出一个类别数字，不要解释，不要输出其他文字。
""".strip()


def label_lines_for(ids: List[int], label_map: Dict[int, str]) -> str:
    """Render ``id: name`` lines for the baseline prompt."""
    return "\n".join(f"{int(i)}: {label_map[int(i)]}" for i in ids)
