from __future__ import annotations

import os
import shlex
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .aggregate import aggregate_errors
from .common import (
    blind_dataset,
    index_by_id,
    read_json,
    read_records,
    resolve_path,
    validate_prompt_bundle,
    write_json,
    write_jsonl,
)
from .evaluate import evaluate_predictions
from .optimizer import generate_candidates


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_target(
    runner: dict[str, Any],
    input_path: Path,
    prompt_path: Path,
    output_path: Path,
    split: str,
    round_index: int,
    candidate_id: str,
    config_dir: Path,
) -> None:
    marker = output_path.with_suffix(output_path.suffix + ".complete")
    if marker.exists() and output_path.exists() and output_path.stat().st_size:
        return
    values = {
        "input": str(input_path),
        "prompt_bundle": str(prompt_path),
        "output": str(output_path),
        "split": split,
        "round": str(round_index),
        "candidate_id": candidate_id,
    }
    command = []
    for part in runner["command"]:
        rendered = str(part)
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value)
        command.append(rendered)
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in runner.get("env", {}).items()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("[运行目标模型]", shlex.join(command), flush=True)
    subprocess.run(
        command,
        cwd=resolve_path(config_dir, runner.get("cwd", ".")),
        env=env,
        check=True,
        shell=False,
    )
    input_ids = set(index_by_id(read_records(input_path), str(input_path)))
    output_ids = set(index_by_id(read_records(output_path), str(output_path)))
    if input_ids != output_ids:
        raise RuntimeError(
            f"目标模型输出不完整：缺失={len(input_ids-output_ids)}，多出={len(output_ids-input_ids)}"
        )
    marker.write_text(utc_now() + "\n", encoding="utf-8")


def save_bundle(directory: Path, bundle: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{bundle['candidate_id']}.json"
    write_json(path, bundle)
    return path


def optimize(config_path: str, offline_response_pattern: str | None = None) -> Path:
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    config = read_json(config_file)
    feedback_gold = resolve_path(config_dir, config["feedback_data"])
    validation_gold = resolve_path(config_dir, config["validation_data"])
    work_dir = resolve_path(config_dir, config["work_dir"])
    current = read_json(resolve_path(config_dir, config["initial_prompt_bundle"]))
    validate_prompt_bundle(current)
    selection = config.get("selection", {})
    max_rounds = int(selection.get("max_rounds", 3))
    patience = int(selection.get("patience", 2))
    min_delta = float(selection.get("min_delta", 0.0))
    representatives = int(selection.get("representatives_per_cluster", 3))
    work_dir.mkdir(parents=True, exist_ok=True)

    blinded_dir = work_dir / "blinded_inputs"
    blinded_feedback = blinded_dir / "feedback.jsonl"
    blinded_validation = blinded_dir / "validation.jsonl"
    blind_dataset(feedback_gold, blinded_feedback)
    blind_dataset(validation_gold, blinded_validation)

    history = []
    no_improvement = 0
    accepted_score = None
    for round_index in range(1, max_rounds + 1):
        round_dir = work_dir / f"round_{round_index:02d}"
        candidate_dir = round_dir / "candidates"
        prediction_dir = round_dir / "predictions"
        metric_dir = round_dir / "metrics"
        baseline = deepcopy(current)
        baseline["candidate_id"] = f"r{round_index}_baseline"
        baseline["target_stage"] = "none"
        baseline_path = save_bundle(candidate_dir, baseline)

        feedback_prediction = prediction_dir / f"{baseline['candidate_id']}_feedback.jsonl"
        run_target(
            config["target_runner"], blinded_feedback, baseline_path, feedback_prediction,
            "feedback", round_index, baseline["candidate_id"], config_dir,
        )
        aggregate, records = aggregate_errors(
            str(feedback_gold), str(feedback_prediction), representatives
        )
        write_json(round_dir / "error_aggregate.json", aggregate)
        write_jsonl(round_dir / "error_records.jsonl", records)

        response_file = (
            offline_response_pattern.format(round=round_index)
            if offline_response_pattern
            else None
        )
        generated = generate_candidates(
            current, aggregate, config["optimizer"], round_index,
            round_dir / "optimizer", response_file,
        )
        scored = []
        for candidate in [baseline] + generated:
            candidate_path = save_bundle(candidate_dir, candidate)
            prediction_path = prediction_dir / f"{candidate['candidate_id']}_validation.jsonl"
            run_target(
                config["target_runner"], blinded_validation, candidate_path, prediction_path,
                "validation", round_index, candidate["candidate_id"], config_dir,
            )
            metrics = evaluate_predictions(str(validation_gold), str(prediction_path))
            metrics.update(
                {"candidate_id": candidate["candidate_id"], "is_baseline": candidate is baseline}
            )
            write_json(metric_dir / f"{candidate['candidate_id']}.json", metrics)
            scored.append((candidate, metrics))

        baseline_score = scored[0][1]["macro_f1"]
        best_candidate, best_metrics = max(
            scored, key=lambda pair: (pair[1]["macro_f1"], pair[0] is baseline)
        )
        improved = best_candidate is not baseline and best_metrics["macro_f1"] > baseline_score + min_delta
        if improved:
            current = best_candidate
            accepted_score = best_metrics["macro_f1"]
            no_improvement = 0
        else:
            accepted_score = baseline_score
            no_improvement += 1
        result = {
            "round": round_index,
            "baseline_macro_f1": baseline_score,
            "best_candidate_id": best_candidate["candidate_id"],
            "best_macro_f1": best_metrics["macro_f1"],
            "accepted": improved,
            "current_candidate_id": current["candidate_id"],
            "scores": [metrics for _, metrics in scored],
        }
        history.append(result)
        write_json(round_dir / "selection.json", result)
        if no_improvement >= patience:
            break

    frozen = deepcopy(current)
    frozen.update(
        {
            "frozen": True,
            "frozen_at_utc": utc_now(),
            "target_model": config.get("target_model_name"),
            "validation_macro_f1": accepted_score,
            "history": history,
        }
    )
    frozen_path = work_dir / "frozen_prompt.json"
    write_json(frozen_path, frozen)
    return frozen_path


def run_frozen_test(
    config_path: str,
    frozen_prompt_path: str,
    test_data_path: str,
    output_path: str,
    metrics_path: str | None,
) -> dict[str, Any] | None:
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    config = read_json(config_file)
    frozen_path = Path(frozen_prompt_path).resolve()
    frozen = read_json(frozen_path)
    if frozen.get("frozen") is not True:
        raise ValueError("测试必须使用 frozen=true 的提示词文件")
    test_gold = Path(test_data_path).resolve()
    output = Path(output_path).resolve()
    blinded = output.parent / (output.stem + ".blinded_input.jsonl")
    blind_dataset(test_gold, blinded)
    run_target(
        config["target_runner"], blinded, frozen_path, output, "test",
        len(frozen.get("history", [])), str(frozen.get("candidate_id", "frozen")), config_dir,
    )
    if not metrics_path:
        return None
    metrics = evaluate_predictions(str(test_gold), str(output))
    write_json(metrics_path, metrics)
    return metrics
