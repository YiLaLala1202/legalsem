from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from .common import validate_prompt_bundle, write_json


SYSTEM_PROMPT = """你是中文法律法规主题分类实验中固定使用的提示词优化器。
你只能根据训练反馈集的聚合错误生成候选提示词，不能接触验证集或测试集，也不能决定最终采用哪个候选。
请只输出合法 JSON，不要输出 Markdown 或额外解释。"""


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
        raise ValueError("优化器必须返回 JSON 对象")
    return value


def build_request(
    current_bundle: dict[str, Any], aggregate: dict[str, Any], candidate_count: int, round_index: int
) -> dict[str, Any]:
    return {
        "任务": "根据 SLARD-31 训练反馈聚合生成完整的中文抽取与分类候选提示词",
        "轮次": round_index,
        "候选数量": candidate_count,
        "要求": [
            "严格返回指定数量的候选。",
            "只能依据反复出现的候选遗漏和类别混淆进行修改。",
            "必须保留当前任务、输出格式和所有 [[...]] 占位符。",
            "抽取提示词重点减少真实主题未进入动态候选集的问题。",
            "分类提示词重点澄清高频混淆主题之间的边界。",
            "不得加入样本 ID、样本特有事实、真实答案或验证集信息。",
            "不得修改程序化的动态候选集构造规则。",
        ],
        "输出格式": {
            "candidates": [
                {
                    "candidate_id": "r<轮次>_c<序号>",
                    "target_stage": "extraction|classification|joint",
                    "rationale": "该候选针对的聚合错误",
                    "extraction_prompt": "完整中文抽取提示词",
                    "classification_prompt": "完整中文分类提示词",
                }
            ]
        },
        "当前提示词": {
            "extraction_prompt": current_bundle["extraction_prompt"],
            "classification_prompt": current_bundle["classification_prompt"],
        },
        "训练反馈聚合": {
            "total_samples": aggregate["total_samples"],
            "stage_counts": aggregate["stage_counts"],
            "candidate_recall": aggregate["candidate_recall"],
            "per_category": aggregate["per_category"],
            "top_confusion_pairs": aggregate["confusion_pairs"][:20],
            "top_error_clusters": aggregate["clusters"][:15],
        },
    }


def validate_candidates(
    response: dict[str, Any], aggregate: dict[str, Any], expected_count: int
) -> list[dict[str, Any]]:
    raw_candidates = response.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != expected_count:
        raise ValueError(f"优化器必须返回 {expected_count} 个候选")
    sample_ids = {
        str(rep["id"])
        for cluster in aggregate.get("clusters", [])
        for rep in cluster.get("representatives", [])
    }
    candidates = []
    seen = set()
    for index, raw in enumerate(raw_candidates, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index} 个候选不是 JSON 对象")
        candidate = {
            "candidate_id": str(raw.get("candidate_id") or f"candidate_{index}"),
            "target_stage": str(raw.get("target_stage") or "joint"),
            "rationale": str(raw.get("rationale") or ""),
            "extraction_system_prompt": "你是中文法律法规主题分类的证据抽取助手，只输出合法 JSON。",
            "classification_system_prompt": "你是中文法律法规条文主题分类助手，只输出一个候选类别名称。",
            "extraction_prompt": raw.get("extraction_prompt"),
            "classification_prompt": raw.get("classification_prompt"),
        }
        if candidate["candidate_id"] in seen:
            raise ValueError(f"候选 ID 重复：{candidate['candidate_id']}")
        seen.add(candidate["candidate_id"])
        if candidate["target_stage"] not in {"extraction", "classification", "joint"}:
            raise ValueError(f"target_stage 非法：{candidate['target_stage']}")
        validate_prompt_bundle(candidate)
        combined = candidate["extraction_prompt"] + candidate["classification_prompt"]
        leaked = [item_id for item_id in sample_ids if item_id and item_id in combined]
        if leaked:
            raise ValueError(f"候选提示词包含训练样本 ID：{leaked[:5]}")
        candidates.append(candidate)
    return candidates


def generate_candidates(
    current_bundle: dict[str, Any],
    aggregate: dict[str, Any],
    config: dict[str, Any],
    round_index: int,
    artifact_dir: Path,
    response_file: str | None = None,
) -> list[dict[str, Any]]:
    count = int(config.get("candidate_count", 3))
    request = build_request(current_bundle, aggregate, count, round_index)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifact_dir / "optimizer_request.json", request)
    if response_file:
        response = json.loads(Path(response_file).read_text(encoding="utf-8"))
        raw_text = json.dumps(response, ensure_ascii=False)
    else:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("请先安装 openai 包") from exc
        key_env = str(config.get("api_key_env", "SILICONFLOW_API_KEY"))
        api_key = os.getenv(key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"环境变量 {key_env} 未设置")
        client = OpenAI(
            api_key=api_key,
            base_url=config.get("base_url"),
            timeout=float(config.get("timeout_seconds", 180)),
        )
        retries = int(config.get("max_retries", 5))
        for attempt in range(retries):
            try:
                result = client.chat.completions.create(
                    model=str(config["model"]),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                    ],
                    temperature=float(config.get("temperature", 0.3)),
                    top_p=float(config.get("top_p", 0.9)),
                    max_tokens=int(config.get("max_tokens", 16000)),
                    seed=config.get("seed", 2026),
                )
                raw_text = result.choices[0].message.content or ""
                response = extract_json(raw_text)
                break
            except Exception:
                if attempt + 1 == retries:
                    raise
                time.sleep(min(60.0, 2**attempt + random.random()))
    write_json(artifact_dir / "optimizer_response.json", response)
    candidates = validate_candidates(response, aggregate, count)
    for candidate in candidates:
        write_json(artifact_dir / f"{candidate['candidate_id']}.json", candidate)
    return candidates
