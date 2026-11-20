from __future__ import annotations

import json
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
    label_map_from_config,
    read_json,
    read_jsonl,
    resolve_path,
    validate_prompt_bundle,
    write_json,
    write_jsonl,
)
from .evaluate import evaluate_predictions
from .optimizer import generate_candidates


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_command(parts: list[str], values: dict[str, str]) -> list[str]:
    command = []
    for part in parts:
        rendered = str(part)
        for key, value in values.items():
            rendered = rendered.replace("{" + key + "}", value)
        command.append(rendered)
    return command


def run_target(
    runner: dict[str, Any],
    input_path: Path,
    prompt_bundle_path: Path,
    output_path: Path,
    split: str,
    round_index: int,
    candidate_id: str,
    config_dir: Path,
) -> None:
    complete_marker = output_path.with_suffix(output_path.suffix + ".complete")
    if complete_marker.exists() and output_path.exists() and output_path.stat().st_size > 0:
        return
    command_template = runner.get("command")
    if not isinstance(command_template, list) or not command_template:
        raise ValueError("target_runner.command must be a non-empty JSON array")
    values = {
        "input": str(input_path),
        "prompt_bundle": str(prompt_bundle_path),
        "output": str(output_path),
        "split": split,
        "round": str(round_index),
        "candidate_id": candidate_id,
    }
    command = _format_command(command_template, values)
    env = os.environ.copy()
    for key, value in (runner.get("env") or {}).items():
        env[str(key)] = str(value)
    cwd = resolve_path(config_dir, runner.get("cwd", "."))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print("[runner]", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True, shell=False)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Target runner did not create output: {output_path}")
    input_ids = set(index_by_id(read_jsonl(input_path), str(input_path)))
    output_ids = set(index_by_id(read_jsonl(output_path), str(output_path)))
    if input_ids != output_ids:
        raise RuntimeError(
            "Target runner output is incomplete: "
            f"missing={len(input_ids - output_ids)}, extra={len(output_ids - input_ids)}"
        )
    complete_marker.write_text(utc_now() + "\n", encoding="utf-8")


def _save_bundle(directory: Path, bundle: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{bundle['candidate_id']}.json"
    write_json(path, bundle)
    return path


def _baseline_bundle(current: dict[str, Any], round_index: int) -> dict[str, Any]:
    baseline = deepcopy(current)
    baseline["candidate_id"] = f"r{round_index}_baseline"
    baseline["target_stage"] = "none"
    baseline["rationale"] = "Current accepted prompt; validation control candidate."
    return baseline


def optimize(config_path: str, offline_response_pattern: str | None = None) -> Path:
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    config = read_json(config_file)
    labels = label_map_from_config(config)
    feedback_gold = resolve_path(config_dir, config["feedback_data"])
    validation_gold = resolve_path(config_dir, config["validation_data"])
    initial_bundle_path = resolve_path(config_dir, config["initial_prompt_bundle"])
    work_dir = resolve_path(config_dir, config["work_dir"])
    runner = config["target_runner"]
    selection = config.get("selection", {})
    max_rounds = int(selection.get("max_rounds", 3))
    patience = int(selection.get("patience", 2))
    min_delta = float(selection.get("min_delta", 0.0))
    representatives = int(selection.get("representatives_per_cluster", 3))

    current = read_json(initial_bundle_path)
    validate_prompt_bundle(current)
    current.setdefault("schema_version", "fgpr.prompt_bundle.v1")
    current.setdefault("candidate_id", "initial")
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json(work_dir / "resolved_config.json", config)

    blinded_dir = work_dir / "blinded_inputs"
    blinded_feedback = blinded_dir / "feedback.jsonl"
    blinded_validation = blinded_dir / "validation.jsonl"
    blind_dataset(feedback_gold, blinded_feedback)
    blind_dataset(validation_gold, blinded_validation)

    history: list[dict[str, Any]] = []
    no_improvement = 0
    accepted_score: float | None = None

    for round_index in range(1, max_rounds + 1):
        round_dir = work_dir / f"round_{round_index:02d}"
        bundle_dir = round_dir / "candidates"
        prediction_dir = round_dir / "predictions"
        metric_dir = round_dir / "metrics"
        round_dir.mkdir(parents=True, exist_ok=True)

        baseline = _baseline_bundle(current, round_index)
        baseline_path = _save_bundle(bundle_dir, baseline)
        feedback_prediction = prediction_dir / f"{baseline['candidate_id']}_feedback.jsonl"
        run_target(
            runner,
            blinded_feedback,
            baseline_path,
            feedback_prediction,
            "feedback",
            round_index,
            baseline["candidate_id"],
            config_dir,
        )
        aggregate, records = aggregate_errors(
            str(feedback_gold), str(feedback_prediction), labels, representatives
        )
        write_json(round_dir / "error_aggregate.json", aggregate)
        write_jsonl(round_dir / "error_records.jsonl", records)

        response_file = None
        if offline_response_pattern:
            response_file = offline_response_pattern.format(round=round_index)
        generated = generate_candidates(
            current,
            aggregate,
            config["optimizer"],
            round_index,
            round_dir / "optimizer",
            response_file=response_file,
        )
        candidate_bundles = [baseline] + generated

        scored = []
        for candidate in candidate_bundles:
            candidate_path = _save_bundle(bundle_dir, candidate)
            output_path = prediction_dir / f"{candidate['candidate_id']}_validation.jsonl"
            run_target(
                runner,
                blinded_validation,
                candidate_path,
                output_path,
                "validation",
                round_index,
                candidate["candidate_id"],
                config_dir,
            )
            metrics = evaluate_predictions(str(validation_gold), str(output_path), labels)
            metrics.update(
                {
                    "candidate_id": candidate["candidate_id"],
                    "is_baseline": candidate is baseline,
                }
            )
            write_json(metric_dir / f"{candidate['candidate_id']}.json", metrics)
            scored.append((candidate, metrics))

        baseline_score = scored[0][1]["macro_f1"]
        # Stable tie breaking keeps the current prompt when scores are identical.
        best_candidate, best_metrics = max(
            scored, key=lambda pair: (pair[1]["macro_f1"], pair[0] is baseline)
        )
        improved = (
            best_candidate is not baseline
            and best_metrics["macro_f1"] > baseline_score + min_delta
        )
        if improved:
            current = best_candidate
            accepted_score = best_metrics["macro_f1"]
            no_improvement = 0
        else:
            accepted_score = baseline_score
            no_improvement += 1

        round_result = {
            "round": round_index,
            "baseline_candidate_id": baseline["candidate_id"],
            "baseline_macro_f1": baseline_score,
            "best_candidate_id": best_candidate["candidate_id"],
            "best_macro_f1": best_metrics["macro_f1"],
            "min_delta": min_delta,
            "accepted": improved,
            "accepted_candidate_id": current["candidate_id"],
            "no_improvement_rounds": no_improvement,
            "scores": [metrics for _, metrics in scored],
        }
        history.append(round_result)
        write_json(round_dir / "selection.json", round_result)
        if no_improvement >= patience:
            break

    frozen = deepcopy(current)
    frozen.update(
        {
            "frozen": True,
            "frozen_at_utc": utc_now(),
            "selection_metric": "macro_f1",
            "validation_macro_f1": accepted_score,
            "optimizer_model": config["optimizer"].get("model"),
            "target_model": config.get("target_model_name"),
            "development_data": {
                "feedback": str(feedback_gold),
                "validation": str(validation_gold),
                "test_used_during_optimization": False,
            },
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
    metrics_path: str | None = None,
) -> dict[str, Any] | None:
    config_file = Path(config_path).resolve()
    config_dir = config_file.parent
    config = read_json(config_file)
    labels = label_map_from_config(config)
    frozen_path = Path(frozen_prompt_path).resolve()
    frozen = read_json(frozen_path)
    if frozen.get("frozen") is not True:
        raise ValueError("Test execution requires a prompt bundle with frozen=true")
    test_gold = Path(test_data_path).resolve()
    output = Path(output_path).resolve()
    blinded = output.parent / (output.stem + ".blinded_input.jsonl")
    blind_dataset(test_gold, blinded)
    run_target(
        config["target_runner"],
        blinded,
        frozen_path,
        output,
        "test",
        len(frozen.get("history", [])),
        str(frozen.get("candidate_id", "frozen")),
        config_dir,
    )
    if not metrics_path:
        return None
    metrics = evaluate_predictions(str(test_gold), str(output), labels)
    metrics.update(
        {
            "test_output": str(output),
            "evaluated_at_utc": utc_now(),
        }
    )
    write_json(metrics_path, metrics)
    return metrics
