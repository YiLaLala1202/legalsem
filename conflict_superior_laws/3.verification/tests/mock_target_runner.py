from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--prompt-bundle", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

bundle = json.loads(Path(args.prompt_bundle).read_text(encoding="utf-8"))
perfect = "MOCK_GOOD_CANDIDATE" in bundle["classification_prompt"]
rows = []
for line in Path(args.input).read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    prediction = int(str(item["id"]).split("_")[-1]) if perfect else 0
    rows.append(
        {
            "id": item["id"],
            "pred_conflict_label_id": prediction,
            "candidate_labels": [0, 1, 2, 3, 4],
            "extracted_elements": {"content_elements": {}},
        }
    )
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
