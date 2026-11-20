#!/usr/bin/env python
"""Unified entry point for every experiment stage.

Examples
--------
Element extraction on the held-out LCR-CN split, local vLLM backend::

    python run.py extract --dataset lcr-cn --backbone qwen3-8b --split test

Dynamic classification using the extracted elements, hosted API backend::

    python run.py classify --dataset lcr-cn --backbone gpt-5.4 \\
        --elements outputs/qwen3-8b_test_extract.jsonl --mode dynamic

Static ablation (full label space, no element-driven reduction)::

    python run.py classify --dataset lcr-cn --backbone gpt-5.4 \\
        --elements outputs/qwen3-8b_test_extract.jsonl --mode static

End-to-end baseline without element extraction::

    python run.py baseline --dataset lcr-cn --backbone legalone-8b --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import yaml

from lsem import io, prompts, tasks, topics
from lsem.backends import BackendError, SamplingConfig, build_backend
from lsem.runner import run_batch

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "backbones.yaml")
DEFAULT_DATA_ROOT = os.path.join(ROOT, "data")
DEFAULT_BUNDLES = {
    "lcr-cn": os.path.join(ROOT, "prompts", "base_prompts.json"),
    "slard-31": os.path.join(ROOT, "prompts", "base_prompts_slard31.json"),
}

DATASET_FILES = {
    "lcr-cn": {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"},
    "slard-31": {"train": "train.json", "val": "val.json", "test": "test.json"},
}


def load_backbone_config(path: str, name: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    backbones = config.get("backbones", {})
    if name not in backbones:
        available = ", ".join(sorted(backbones)) or "(none)"
        raise SystemExit(f"unknown backbone {name!r}; available: {available}")

    return backbones[name]


def read_dataset(path: str) -> List[Dict[str, Any]]:
    """Read either a JSON array or a JSONL file."""
    if path.endswith(".jsonl"):
        return io.read_jsonl(path)

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise SystemExit(f"expected a JSON array in {path}")
    return payload


DATASET_DIRS = {"lcr-cn": "conflict_superior_laws", "slard-31": "SLARD_classification"}


def default_output(task: str, args: argparse.Namespace) -> str:
    """Mirror the on-disk layout the analysis scripts expect.

    Keeping these paths stable is what lets ``analysis/`` glob the prediction
    files without configuration.
    """
    base = os.path.join(ROOT, DATASET_DIRS[args.dataset], "output")
    name = args.backbone
    split = args.split

    if task == "extract":
        return os.path.join(base, "full", "elements", f"{name}_{split}_extract.jsonl")

    if task == "classify":
        if args.mode == "static":
            return os.path.join(base, "ablation", f"{name}_{split}_class_no_dynamic.jsonl")
        return os.path.join(base, "full", "pred_classfication", f"{name}_{split}_class.jsonl")

    return os.path.join(base, "baseline", f"{name}_{split}_baseline_output.jsonl")


def report_accuracy(rows: List[Dict[str, Any]], dataset: str) -> None:
    """Print accuracy over rows that carry both a prediction and a gold label."""
    if dataset == "slard-31":
        scored = [
            row
            for row in rows
            if isinstance(row.get("category"), str) and isinstance(row.get("gold_category"), str)
        ]
        if not scored:
            print("no scored rows (missing gold labels or all predictions failed)")
            return
        correct = sum(1 for row in scored if row["category"] == row["gold_category"])
    else:
        scored = [
            row
            for row in rows
            if isinstance(row.get("pred_conflict_label_id"), int)
            and isinstance(row.get("gold_label_id"), int)
        ]
        if not scored:
            print("no scored rows (missing gold labels or all predictions failed)")
            return
        correct = sum(
            1 for row in scored if row["pred_conflict_label_id"] == row["gold_label_id"]
        )

    print(f"evaluated={len(scored)} correct={correct} accuracy={correct / len(scored):.4f}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["extract", "classify", "baseline"])
    parser.add_argument("--dataset", default="lcr-cn", choices=sorted(DATASET_FILES))
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--input", help="explicit input file, overrides --dataset/--split")
    parser.add_argument("--output", help="explicit output file")
    parser.add_argument("--elements", help="extraction output consumed by the classify task")
    parser.add_argument("--mode", default="dynamic", choices=["dynamic", "static"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--prompt-bundle", default=None, help="defaults to the bundle for --dataset")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerun-rounds", type=int, default=2)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve inputs, print the plan, and exit without calling any model",
    )

    args = parser.parse_args(argv)

    if args.task == "classify":
        if not args.elements:
            parser.error("--elements is required for the classify task")
        input_path = args.elements
    else:
        input_path = args.input or os.path.join(
            args.data_root, args.dataset, DATASET_FILES[args.dataset][args.split]
        )

    if not os.path.exists(input_path):
        raise SystemExit(
            f"input not found: {input_path}\n"
            f"place the dataset under {args.data_root}/<dataset>/ or pass --input"
        )

    output_path = args.output or default_output(args.task, args)
    items = read_dataset(input_path)

    backbone_cfg = load_backbone_config(args.config, args.backbone)

    print(f"task       : {args.task}" + (f" (mode={args.mode})" if args.task == "classify" else ""))
    print(f"backbone   : {args.backbone} -> {backbone_cfg.get('model')} via {backbone_cfg.get('backend', 'api')}")
    print(f"input      : {input_path} ({len(items)} cases)")
    print(f"output     : {output_path}")

    if args.dry_run:
        print("dry run: no model call performed")
        return 0

    sampling = SamplingConfig.from_dict(backbone_cfg.get("sampling"))
    backend_kwargs = {key: value for key, value in backbone_cfg.items() if key not in {"backend", "model", "sampling"}}

    try:
        backend = build_backend(
            backbone_cfg.get("backend", "api"),
            model=backbone_cfg["model"],
            sampling=sampling,
            **backend_kwargs,
        )
    except BackendError as exc:
        raise SystemExit(f"backend error: {exc}") from exc

    is_topic_task = args.dataset == "slard-31"
    bundle_path = args.prompt_bundle or DEFAULT_BUNDLES[args.dataset]

    if args.task == "extract":
        bundle = prompts.load_bundle(bundle_path)
        process_one = (
            (lambda item, index: topics.topic_extract_row(backend, bundle, item, index))
            if is_topic_task
            else (lambda item, index: tasks.extract_row(backend, bundle, item, index))
        )
        stats = run_batch(
            items,
            output_path,
            process_one=process_one,
            is_success=topics.topic_extract_succeeded if is_topic_task else tasks.extract_succeeded,
            make_placeholder=topics.topic_extract_placeholder if is_topic_task else tasks.extract_placeholder,
            rerun_rounds=args.rerun_rounds,
            limit=args.limit,
            quiet=args.quiet,
        )
    elif args.task == "classify":
        bundle = prompts.load_bundle(bundle_path)
        process_one = (
            (lambda item, index: topics.topic_classify_row(backend, bundle, item, index, mode=args.mode))
            if is_topic_task
            else (lambda item, index: tasks.classify_row(backend, bundle, item, index, mode=args.mode))
        )
        stats = run_batch(
            items,
            output_path,
            process_one=process_one,
            is_success=topics.topic_classify_succeeded if is_topic_task else tasks.classify_succeeded,
            make_placeholder=topics.topic_classify_placeholder if is_topic_task else tasks.classify_placeholder,
            rerun_rounds=args.rerun_rounds,
            limit=args.limit,
            quiet=args.quiet,
        )
    else:
        stats = run_batch(
            items,
            output_path,
            process_one=(
                (lambda item, index: topics.baseline_row(backend, item, index))
                if is_topic_task
                else (lambda item, index: tasks.baseline_row(backend, item, index))
            ),
            is_success=topics.baseline_succeeded if is_topic_task else tasks.baseline_succeeded,
            make_placeholder=topics.baseline_placeholder if is_topic_task else tasks.baseline_placeholder,
            rerun_rounds=args.rerun_rounds,
            limit=args.limit,
            quiet=args.quiet,
        )

    if args.task in {"classify", "baseline"}:
        report_accuracy(io.read_jsonl(output_path), args.dataset)

    print(f"done: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
