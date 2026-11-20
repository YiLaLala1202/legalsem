from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from .common import validate_prompt_bundle, write_json


OPTIMIZER_SYSTEM_PROMPT = """你是法律文本分类实验中固定使用的开发阶段提示词优化器。
你的职责是根据训练反馈集的聚合错误，生成完整的候选提示词；你不负责选择最终候选，也不能接触验证集或测试集。
请只输出合法 JSON，不要输出 Markdown 代码块或额外解释。"""


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Optimizer response must be a JSON object")
    return value


def build_optimizer_request(
    current_bundle: dict[str, Any],
    aggregate: dict[str, Any],
    candidate_count: int,
    round_index: int,
) -> dict[str, Any]:
    compact_feedback = {
        "total_samples": aggregate["total_samples"],
        "stage_counts": aggregate["stage_counts"],
        "candidate_recall": aggregate["candidate_recall"],
        "per_gold_label": aggregate["per_gold_label"],
        "top_confusion_pairs": aggregate["confusion_pairs"][:12],
        "top_error_clusters": aggregate["clusters"][:12],
    }
    return {
        "task": "为反馈引导的提示词优化流程生成完整的抽取与分类候选提示词",
        "round": round_index,
        "candidate_count": candidate_count,
        "constraints": [
            "必须返回 candidate_count 指定数量的候选。",
            "只能依据给出的训练反馈聚合中反复出现的错误模式进行修改。",
            "以当前提示词为基础，保留原任务、输出格式和所有占位符。",
            "重点澄清要素定义、遗漏规则和类别判定边界，不得加入针对某个样本的答案。",
            "不得把样本 ID 或样本特有事实复制到运行时提示词中。",
            "运行时提示词不得提及真实标签、验证集、测试集或上一轮具体预测。",
            "抽取提示词必须保留 [[CONTENT]] 和 [[HIGH_LEVEL_LAWS]]。",
            "分类提示词必须保留 [[CONTENT]]、[[HIGH_LEVEL_LAWS]]、[[CONTENT_ELEMENTS]]、[[CANDIDATE_LABELS]] 和 [[CANDIDATE_EXPLANATIONS]]。",
            "动态候选集由程序构造，不得修改候选集构造规则。",
            "不同候选应重点解决不同的高频遗漏或类别混淆边界。",
        ],
        "output_schema": {
            "candidates": [
                {
                    "candidate_id": "r<round>_c<index>",
                    "target_stage": "extraction|classification|joint",
                    "rationale": "简要说明该候选针对的聚合错误",
                    "extraction_prompt": "完整的中文抽取提示词模板",
                    "classification_prompt": "完整的中文分类提示词模板",
                }
            ]
        },
        "current_prompt_bundle": {
            "extraction_prompt": current_bundle["extraction_prompt"],
            "classification_prompt": current_bundle["classification_prompt"],
        },
        "training_feedback_aggregate": compact_feedback,
    }


def _reject_feedback_leakage(bundle: dict[str, Any], aggregate: dict[str, Any]) -> None:
    combined = bundle["extraction_prompt"] + "\n" + bundle["classification_prompt"]
    representative_ids = {
        str(rep["id"])
        for cluster in aggregate.get("clusters", [])
        for rep in cluster.get("representatives", [])
    }
    leaked = sorted(item_id for item_id in representative_ids if item_id and item_id in combined)
    if leaked:
        raise ValueError(f"Candidate prompt contains feedback sample IDs: {leaked[:5]}")


def validate_candidates(
    response: dict[str, Any],
    current_bundle: dict[str, Any],
    aggregate: dict[str, Any],
    expected_count: int,
) -> list[dict[str, Any]]:
    raw_candidates = response.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != expected_count:
        raise ValueError(
            f"Optimizer must return exactly {expected_count} candidates; got "
            f"{len(raw_candidates) if isinstance(raw_candidates, list) else 'non-list'}"
        )
    candidates = []
    seen_ids = set()
    for index, raw in enumerate(raw_candidates, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"Candidate {index} is not an object")
        candidate = {
            "schema_version": "fgpr.prompt_bundle.v1",
            "candidate_id": str(raw.get("candidate_id") or f"candidate_{index}"),
            "target_stage": str(raw.get("target_stage") or "joint"),
            "rationale": str(raw.get("rationale") or ""),
            "extraction_system_prompt": current_bundle.get(
                "extraction_system_prompt", "你是法律文本结构化信息抽取助手。只输出合法 JSON。"
            ),
            "classification_system_prompt": current_bundle.get(
                "classification_system_prompt", "你是法律规范冲突分类助手。只输出一个候选类别数字。"
            ),
            "extraction_prompt": raw.get("extraction_prompt"),
            "classification_prompt": raw.get("classification_prompt"),
        }
        if candidate["candidate_id"] in seen_ids:
            raise ValueError(f"Duplicate candidate_id={candidate['candidate_id']}")
        seen_ids.add(candidate["candidate_id"])
        if candidate["target_stage"] not in {"extraction", "classification", "joint"}:
            raise ValueError(f"Invalid target_stage={candidate['target_stage']}")
        validate_prompt_bundle(candidate)
        _reject_feedback_leakage(candidate, aggregate)
        candidates.append(candidate)
    return candidates


def generate_candidates(
    current_bundle: dict[str, Any],
    aggregate: dict[str, Any],
    optimizer_config: dict[str, Any],
    round_index: int,
    artifact_dir: Path,
    response_file: str | None = None,
) -> list[dict[str, Any]]:
    candidate_count = int(optimizer_config.get("candidate_count", 3))
    request = build_optimizer_request(current_bundle, aggregate, candidate_count, round_index)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "optimizer_request.json", request)

    if response_file:
        response = json.loads(Path(response_file).read_text(encoding="utf-8"))
        raw_text = json.dumps(response, ensure_ascii=False)
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'openai' package to call the fixed optimizer") from exc
        key_env = str(optimizer_config.get("api_key_env", "SILICONFLOW_API_KEY"))
        api_key = os.getenv(key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Environment variable {key_env} is not set")
        client = OpenAI(
            api_key=api_key,
            base_url=optimizer_config.get("base_url"),
            timeout=float(optimizer_config.get("timeout_seconds", 180)),
        )
        retries = int(optimizer_config.get("max_retries", 5))
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                result = client.chat.completions.create(
                    model=str(optimizer_config["model"]),
                    messages=[
                        {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                    ],
                    temperature=float(optimizer_config.get("temperature", 0.3)),
                    top_p=float(optimizer_config.get("top_p", 0.9)),
                    max_tokens=int(optimizer_config.get("max_tokens", 12000)),
                    seed=optimizer_config.get("seed", 2026),
                )
                raw_text = result.choices[0].message.content or ""
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 == retries:
                    raise
                time.sleep(min(60.0, (2**attempt) + random.random()))
        else:  # pragma: no cover
            raise RuntimeError("Optimizer call failed") from last_error
        response = _extract_json(raw_text)

    (artifact_dir / "optimizer_raw_response.txt").write_text(raw_text, encoding="utf-8")
    write_json(artifact_dir / "optimizer_response.json", response)
    candidates = validate_candidates(response, current_bundle, aggregate, candidate_count)
    for candidate in candidates:
        write_json(artifact_dir / f"{candidate['candidate_id']}.json", candidate)
    return candidates
