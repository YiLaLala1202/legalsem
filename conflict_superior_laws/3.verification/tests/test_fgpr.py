from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from fgpr.aggregate import aggregate_errors
from fgpr.common import DEFAULT_LABEL_MAP, blind_dataset
from fgpr.evaluate import evaluate_predictions
from fgpr.orchestrator import optimize


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class FgprTests(unittest.TestCase):
    def test_error_stage_and_candidate_recall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "gold.jsonl"
            pred = root / "pred.jsonl"
            write_jsonl(
                gold,
                [
                    {"id": "a", "conflict_type": "职权或责任划分不符", "content": "a"},
                    {"id": "b", "conflict_type": "概念或定义范围不符", "content": "b"},
                    {"id": "c", "conflict_type": "无冲突", "content": "c"},
                ],
            )
            write_jsonl(
                pred,
                [
                    {"id": "a", "pred_conflict_label_id": 0, "candidate_labels": [0, 4]},
                    {"id": "b", "pred_conflict_label_id": 0, "candidate_labels": [0, 2]},
                    {"id": "c", "pred_conflict_label_id": 0, "candidate_labels": [0, 1]},
                ],
            )
            summary, records = aggregate_errors(str(gold), str(pred), DEFAULT_LABEL_MAP)
            self.assertEqual(summary["stage_counts"]["element_candidate_construction_error"], 1)
            self.assertEqual(summary["stage_counts"]["classification_error"], 1)
            self.assertEqual(summary["stage_counts"]["correct"], 1)
            self.assertAlmostEqual(summary["candidate_recall"], 2 / 3)
            self.assertEqual(len(records), 3)

    def test_macro_f1_uses_all_fixed_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / "gold.jsonl"
            pred = root / "pred.jsonl"
            write_jsonl(
                gold,
                [{"id": str(i), "conflict_type": DEFAULT_LABEL_MAP[i]} for i in range(5)],
            )
            write_jsonl(
                pred,
                [{"id": str(i), "pred_conflict_label_id": i} for i in range(5)],
            )
            metrics = evaluate_predictions(str(gold), str(pred), DEFAULT_LABEL_MAP)
            self.assertEqual(metrics["macro_f1"], 1.0)

    def test_blinding_removes_supervision_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            destination = root / "blinded.jsonl"
            write_jsonl(
                source,
                [
                    {
                        "id": "a",
                        "content": "text",
                        "conflict_type": "无冲突",
                        "conflict_description": "gold explanation",
                    }
                ],
            )
            blind_dataset(source, destination)
            row = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(row["content"], "text")
            self.assertNotIn("conflict_type", row)
            self.assertNotIn("conflict_description", row)

    def test_offline_end_to_end_selection_and_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feedback = root / "feedback.jsonl"
            validation = root / "validation.jsonl"
            rows = [
                {
                    "id": f"case_{label}",
                    "content": f"content {label}",
                    "high_level_laws": ["law"],
                    "conflict_type": DEFAULT_LABEL_MAP[label],
                }
                for label in range(5)
            ]
            write_jsonl(feedback, rows)
            write_jsonl(validation, rows)
            extraction = "[[CONTENT]]\n[[HIGH_LEVEL_LAWS]]"
            classification = (
                "[[CONTENT]]\n[[HIGH_LEVEL_LAWS]]\n[[CONTENT_ELEMENTS]]\n"
                "[[CANDIDATE_LABELS]]\n[[CANDIDATE_EXPLANATIONS]]\n[[TASK_TYPE]]"
            )
            base = root / "base.json"
            base.write_text(
                json.dumps(
                    {
                        "candidate_id": "initial",
                        "extraction_prompt": extraction,
                        "classification_prompt": classification,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            response = root / "response_1.json"
            response.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_id": "r1_c1",
                                "target_stage": "classification",
                                "rationale": "mock automatic candidate",
                                "extraction_prompt": extraction,
                                "classification_prompt": classification + "\nMOCK_GOOD_CANDIDATE",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            mock_runner = Path(__file__).with_name("mock_target_runner.py").resolve()
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "target_model_name": "mock",
                        "feedback_data": str(feedback),
                        "validation_data": str(validation),
                        "initial_prompt_bundle": str(base),
                        "work_dir": str(root / "run"),
                        "optimizer": {"candidate_count": 1, "model": "mock-optimizer"},
                        "target_runner": {
                            "cwd": str(root),
                            "command": [
                                sys.executable,
                                str(mock_runner),
                                "--input",
                                "{input}",
                                "--prompt-bundle",
                                "{prompt_bundle}",
                                "--output",
                                "{output}",
                            ],
                        },
                        "selection": {"max_rounds": 1, "patience": 1},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            frozen_path = optimize(str(config), str(root / "response_{round}.json"))
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
            self.assertTrue(frozen["frozen"])
            self.assertEqual(frozen["candidate_id"], "r1_c1")
            self.assertEqual(frozen["validation_macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
