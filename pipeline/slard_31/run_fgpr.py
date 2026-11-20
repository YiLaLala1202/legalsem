from __future__ import annotations

import argparse

from fgpr.orchestrator import optimize, run_frozen_test


def main() -> None:
    parser = argparse.ArgumentParser(description="SLARD-31 反馈引导提示词优化")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dev = subparsers.add_parser("optimize", help="训练反馈聚合、候选生成、验证选择与冻结")
    dev.add_argument("--config", required=True)
    dev.add_argument("--offline-response-pattern")
    test = subparsers.add_parser("test", help="使用冻结提示词运行一次测试集")
    test.add_argument("--config", required=True)
    test.add_argument("--frozen-prompt", required=True)
    test.add_argument("--test-data", required=True)
    test.add_argument("--output", required=True)
    test.add_argument("--metrics")
    args = parser.parse_args()
    if args.command == "optimize":
        print("冻结提示词：", optimize(args.config, args.offline_response_pattern))
    else:
        metrics = run_frozen_test(
            args.config, args.frozen_prompt, args.test_data, args.output, args.metrics
        )
        if metrics:
            print(f"测试集 Macro-F1：{metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
