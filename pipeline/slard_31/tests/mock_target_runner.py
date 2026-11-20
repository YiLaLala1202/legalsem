from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from fgpr.categories import CATEGORIES
from fgpr.common import read_json, read_records


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--prompt-bundle", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
bundle = read_json(args.prompt_bundle)
good = "MOCK_GOOD" in bundle["classification_prompt"]
rows = []
for item in read_records(args.input):
    label_index = int(str(item["id"]).split("_")[-1])
    prediction = CATEGORIES[label_index] if good else CATEGORIES[0]
    rows.append(
        {
            "id": str(item["id"]),
            "extracted_fields": {category: ["evidence"] for category in CATEGORIES},
            "candidate_categories": list(CATEGORIES),
            "pred_category": prediction,
        }
    )
target = Path(args.output)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
)
