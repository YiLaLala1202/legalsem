from __future__ import annotations

import argparse

from fgpr.orchestrator import optimize, run_frozen_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feedback-Guided Prompt Refinement with metric-based validation selection"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize_parser = subparsers.add_parser(
        "optimize", help="aggregate feedback, generate candidates, validate, and freeze"
    )
    optimize_parser.add_argument("--config", required=True)
    optimize_parser.add_argument(
        "--offline-response-pattern",
        help="Optional JSON response path; may contain {round} for reproducible dry runs",
    )

    test_parser = subparsers.add_parser(
        "test", help="run a previously frozen prompt once on a held-out test set"
    )
    test_parser.add_argument("--config", required=True)
    test_parser.add_argument("--frozen-prompt", required=True)
    test_parser.add_argument("--test-data", required=True)
    test_parser.add_argument("--output", required=True)
    test_parser.add_argument("--metrics", help="Optional post-inference metrics JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "optimize":
        frozen_path = optimize(args.config, args.offline_response_pattern)
        print(f"Frozen prompt: {frozen_path}")
    else:
        metrics = run_frozen_test(
            args.config,
            args.frozen_prompt,
            args.test_data,
            args.output,
            args.metrics,
        )
        if metrics:
            print(f"Test Macro-F1: {metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
