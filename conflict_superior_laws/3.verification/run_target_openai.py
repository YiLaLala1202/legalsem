from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from fgpr.common import (
    DEFAULT_LABEL_MAP,
    ELEMENT_FIELDS,
    read_json,
    read_jsonl,
    replace_tokens,
    validate_prompt_bundle,
)


LABEL_EXPLANATIONS = {
    0: "下位法与上位法含义一致，无实质性规范冲突",
    1: "下位法改变主体职责、权限、义务或责任范围",
    2: "下位法改变概念、定义、对象边界或适用对象范围",
    3: "下位法改变处罚种类、幅度、对象、范围或法律后果",
    4: "下位法增设、删除、放宽或收紧适用条件、前提或例外",
}


def empty_elements() -> dict[str, list[str]]:
    return {field: [] for field in ELEMENT_FIELDS}


def normalize_elements(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    normalized: dict[str, Any] = {}
    for side in ("content_elements", "high_level_law_elements"):
        source = value.get(side) if isinstance(value.get(side), dict) else {}
        normalized[side] = {}
        for field in ELEMENT_FIELDS:
            raw = source.get(field, [])
            if isinstance(raw, list):
                normalized[side][field] = [str(item).strip() for item in raw if str(item).strip()]
            elif raw is None or str(raw).strip() == "":
                normalized[side][field] = []
            else:
                normalized[side][field] = [str(raw).strip()]
    return normalized


def extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Extraction response is not a JSON object")
    return value


def is_non_empty(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def choose_candidate_labels(content_elements: dict[str, Any]) -> tuple[list[int], str]:
    """Fixed programmatic mapping; the optimizer is not allowed to modify it."""
    if not isinstance(content_elements, dict):
        return list(DEFAULT_LABEL_MAP), "all_empty"
    non_empty_fields = [
        field for field in ELEMENT_FIELDS if is_non_empty(content_elements.get(field, []))
    ]
    if not non_empty_fields:
        return list(DEFAULT_LABEL_MAP), "all_empty"
    candidates = {0}
    if any(
        is_non_empty(content_elements.get(field, []))
        for field in ("Subject", "Action_Behavior", "Responsibility")
    ):
        candidates.add(1)
    if is_non_empty(content_elements.get("Definition", [])):
        candidates.add(2)
    if is_non_empty(content_elements.get("Sanction", [])):
        candidates.add(3)
    if is_non_empty(content_elements.get("Condition", [])):
        candidates.add(4)
    return sorted(candidates), ",".join(non_empty_fields)


def parse_prediction(text: str, candidates: list[int]) -> int:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    exact = re.fullmatch(r"\s*(\d+)\s*", cleaned)
    if exact and int(exact.group(1)) in candidates:
        return int(exact.group(1))
    found = [int(value) for value in re.findall(r"(?<!\d)(\d+)(?!\d)", cleaned)]
    valid = [value for value in found if value in candidates]
    if len(set(valid)) == 1:
        return valid[0]
    raise ValueError(f"Cannot parse one candidate label from: {text[:200]!r}")


class ChatClient:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to use this runner") from exc
        api_key = os.getenv(args.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Environment variable {args.api_key_env} is not set")
        self.client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
        self.model = args.model
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_retries = args.max_retries
        self.sleep_seconds = args.sleep_seconds

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=max_tokens,
                )
                if self.sleep_seconds:
                    time.sleep(self.sleep_seconds)
                return (response.choices[0].message.content or "").strip()
            except Exception:
                if attempt + 1 == self.max_retries:
                    raise
                time.sleep(min(60.0, (2**attempt) + random.random()))
        raise RuntimeError("Unreachable retry state")


def high_law_text(item: dict[str, Any]) -> str:
    laws = item.get("high_level_laws") or []
    return "\n".join(f"{index}. {law}" for index, law in enumerate(laws, 1))


def process_item(
    item: dict[str, Any],
    bundle: dict[str, Any],
    client: ChatClient,
    extraction_max_tokens: int,
    classification_max_tokens: int,
) -> dict[str, Any]:
    common = {
        "CONTENT": str(item.get("content") or ""),
        "HIGH_LEVEL_LAWS": high_law_text(item),
    }
    extraction_prompt = replace_tokens(bundle["extraction_prompt"], common)
    extraction_raw = client.complete(
        bundle.get("extraction_system_prompt", "只输出合法 JSON。"),
        extraction_prompt,
        extraction_max_tokens,
    )
    extracted = normalize_elements(extract_json(extraction_raw))
    content_elements = extracted["content_elements"]
    candidates, task_type = choose_candidate_labels(content_elements)
    explanations = "\n".join(
        f"{label}: {DEFAULT_LABEL_MAP[label]} - {LABEL_EXPLANATIONS[label]}"
        for label in candidates
    )
    classification_prompt = replace_tokens(
        bundle["classification_prompt"],
        {
            **common,
            "CONTENT_ELEMENTS": json.dumps(content_elements, ensure_ascii=False, indent=2),
            "HIGH_LEVEL_LAW_ELEMENTS": json.dumps(
                extracted["high_level_law_elements"], ensure_ascii=False, indent=2
            ),
            "CANDIDATE_LABELS": ", ".join(str(label) for label in candidates),
            "CANDIDATE_EXPLANATIONS": explanations,
            "TASK_TYPE": task_type,
        },
    )
    classification_raw = client.complete(
        bundle.get("classification_system_prompt", "只输出一个候选类别数字。"),
        classification_prompt,
        classification_max_tokens,
    )
    prediction = parse_prediction(classification_raw, candidates)
    return {
        "id": str(item["id"]),
        "extracted_elements": extracted,
        "candidate_labels": candidates,
        "task_type": task_type,
        "pred_conflict_label_id": prediction,
        "pred_conflict_type": DEFAULT_LABEL_MAP[prediction],
        "raw_extraction_output": extraction_raw,
        "raw_classification_output": classification_raw,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one target model with an FGPR prompt bundle")
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompt-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="TARGET_MODEL_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--extraction-max-tokens", type=int, default=2048)
    parser.add_argument("--classification-max-tokens", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = read_json(args.prompt_bundle)
    validate_prompt_bundle(bundle)
    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict[str, Any]] = {}
    if output.exists():
        for row in read_jsonl(output):
            done[str(row["id"])] = row
    client = ChatClient(args)
    with output.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(rows, 1):
            item_id = str(item.get("id"))
            if item_id in done:
                continue
            result = process_item(
                item,
                bundle,
                client,
                args.extraction_max_tokens,
                args.classification_max_tokens,
            )
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] id={item_id} pred={result['pred_conflict_label_id']}")


if __name__ == "__main__":
    main()
