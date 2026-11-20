from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from fgpr.aggregate import aggregate_errors
from fgpr.categories import CATEGORIES
from fgpr.common import read_json, validate_prompt_bundle
from fgpr.evaluate import evaluate_predictions
from fgpr.orchestrator import optimize


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


class SlardFgprTests(unittest.TestCase):
    def test_base_prompt_is_valid(self) -> None:
        bundle = read_json(Path(__file__).parents[1] / "base_prompts.json")
        validate_prompt_bundle(bundle)

    def test_aggregate_and_macro_f1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "gold.json"
            pred = root / "pred.jsonl"
            gold_rows = [
                {"id": f"case_{index}", "category": category, "title": "t", "clause_content": "c"}
                for index, category in enumerate(CATEGORIES)
            ]
            pred_rows = [
                {
                    "id": f"case_{index}",
                    "pred_category": category,
                    "candidate_categories": list(CATEGORIES),
                    "extracted_fields": {},
                }
                for index, category in enumerate(CATEGORIES)
            ]
            write_json(gold, gold_rows)
            write_jsonl(pred, pred_rows)
            summary, records = aggregate_errors(str(gold), str(pred))
            metrics = evaluate_predictions(str(gold), str(pred))
            self.assertEqual(summary["candidate_recall"], 1.0)
            self.assertEqual(len(records), 31)
            self.assertEqual(metrics["macro_f1"], 1.0)

    def test_offline_end_to_end_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "id": f"case_{index}",
                    "title": "title",
                    "clause_content": "content",
                    "category": category,
                }
                for index, category in enumerate(CATEGORIES)
            ]
            feedback = root / "feedback.json"
            validation = root / "validation.json"
            write_json(feedback, rows)
            write_json(validation, rows)
            base = read_json(Path(__file__).parents[1] / "base_prompts.json")
            base_path = root / "base.json"
            write_json(base_path, base)
            candidate = dict(base)
            candidate.update(
                {
                    "candidate_id": "r1_c1",
                    "target_stage": "classification",
                    "rationale": "mock",
                    "classification_prompt": base["classification_prompt"] + "\nMOCK_GOOD",
                }
            )
            response = root / "response_1.json"
            write_json(response, {"candidates": [candidate]})
            mock_runner = Path(__file__).with_name("mock_target_runner.py").resolve()
            config = root / "config.json"
            write_json(
                config,
                {
                    "target_model_name": "mock",
                    "feedback_data": str(feedback),
                    "validation_data": str(validation),
                    "initial_prompt_bundle": str(base_path),
                    "work_dir": str(root / "run"),
                    "optimizer": {"candidate_count": 1, "model": "mock"},
                    "target_runner": {
                        "cwd": str(root),
                        "command": [
                            sys.executable,
                            str(mock_runner),
                            "--input", "{input}",
                            "--prompt-bundle", "{prompt_bundle}",
                            "--output", "{output}",
                        ],
                    },
                    "selection": {"max_rounds": 1, "patience": 1},
                },
            )
            frozen = read_json(optimize(str(config), str(root / "response_{round}.json")))
            self.assertTrue(frozen["frozen"])
            self.assertEqual(frozen["candidate_id"], "r1_c1")
            self.assertEqual(frozen["validation_macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
