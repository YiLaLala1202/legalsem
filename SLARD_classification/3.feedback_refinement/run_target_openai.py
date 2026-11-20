from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from fgpr.categories import CATEGORIES, category_definition_text
from fgpr.common import read_json, read_records, replace_tokens, validate_prompt_bundle


def normalize_fields(value: Any) -> dict[str, list[str]]:
    source = value if isinstance(value, dict) else {}
    result = {}
    for category in CATEGORIES:
        raw = source.get(category, [])
        if isinstance(raw, list):
            result[category] = [str(item).strip() for item in raw if str(item).strip()]
        elif raw is None or not str(raw).strip():
            result[category] = []
        else:
            result[category] = [str(raw).strip()]
    return result


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
        raise ValueError("抽取输出不是 JSON 对象")
    return value


def choose_candidates(fields: dict[str, list[str]]) -> tuple[list[str], str]:
    candidates = [category for category in CATEGORIES if fields.get(category)]
    if not candidates:
        return list(CATEGORIES), "all_empty"
    return candidates, ",".join(candidates)


def parse_category(text: str, candidates: list[str]) -> str:
    cleaned = text.strip().strip('"\'').strip("。；;，,")
    if cleaned in candidates:
        return cleaned
    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            candidate = str(value.get("category", "")).strip()
            if candidate in candidates:
                return candidate
    except json.JSONDecodeError:
        pass
    matches = [category for category in candidates if category in cleaned]
    if len(set(matches)) == 1:
        return matches[0]
    raise ValueError(f"无法从输出中解析唯一候选类别：{text[:200]!r}")


class ChatClient:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("请先安装 openai 包") from exc
        key = os.getenv(args.api_key_env, "").strip()
        if not key:
            raise RuntimeError(f"环境变量 {args.api_key_env} 未设置")
        self.client = OpenAI(api_key=key, base_url=args.base_url, timeout=args.timeout)
        self.args = args

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        for attempt in range(self.args.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.args.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    max_tokens=max_tokens,
                )
                if self.args.sleep_seconds:
                    time.sleep(self.args.sleep_seconds)
                return (response.choices[0].message.content or "").strip()
            except Exception:
                if attempt + 1 == self.args.max_retries:
                    raise
                time.sleep(min(60.0, 2**attempt + random.random()))
        raise RuntimeError("模型调用重试失败")


def process_item(
    item: dict[str, Any], bundle: dict[str, Any], client: ChatClient, args: argparse.Namespace
) -> dict[str, Any]:
    title = str(item.get("title", ""))
    content = str(item.get("clause_content", item.get("content", "")))
    extraction_prompt = replace_tokens(
        bundle["extraction_prompt"],
        {
            "TITLE": title,
            "CONTENT": content,
            "CATEGORY_DEFINITIONS": category_definition_text(),
            "EMPTY_SCHEMA": json.dumps(
                {category: [] for category in CATEGORIES}, ensure_ascii=False, indent=2
            ),
        },
    )
    extraction_raw = client.complete(
        bundle.get("extraction_system_prompt", "只输出合法 JSON。"),
        extraction_prompt,
        args.extraction_max_tokens,
    )
    fields = normalize_fields(extract_json(extraction_raw))
    candidates, task_type = choose_candidates(fields)
    classification_prompt = replace_tokens(
        bundle["classification_prompt"],
        {
            "TITLE": title,
            "CONTENT": content,
            "EXTRACTED_FIELDS": json.dumps(fields, ensure_ascii=False, indent=2),
            "CANDIDATE_CATEGORIES": "\n".join(f"- {value}" for value in candidates),
            "TASK_TYPE": task_type,
        },
    )
    classification_raw = client.complete(
        bundle.get("classification_system_prompt", "只输出一个候选类别名称。"),
        classification_prompt,
        args.classification_max_tokens,
    )
    prediction = parse_category(classification_raw, candidates)
    return {
        "id": str(item["id"]),
        "extracted_fields": fields,
        "candidate_categories": candidates,
        "task_type": task_type,
        "pred_category": prediction,
        "raw_extraction_output": extraction_raw,
        "raw_classification_output": classification_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="使用一组 SLARD-31 Prompt 运行目标模型")
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompt-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", default="TARGET_MODEL_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--extraction-max-tokens", type=int, default=4096)
    parser.add_argument("--classification-max-tokens", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    bundle = read_json(args.prompt_bundle)
    validate_prompt_bundle(bundle)
    rows = read_records(args.input)
    if args.limit:
        rows = rows[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if output.exists():
        completed = {str(row["id"]): row for row in read_records(output)}
    client = ChatClient(args)
    with output.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(rows, 1):
            item_id = str(item["id"])
            if item_id in completed:
                continue
            result = process_item(item, bundle, client, args)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index}/{len(rows)}] id={item_id} 预测={result['pred_category']}")


if __name__ == "__main__":
    main()
