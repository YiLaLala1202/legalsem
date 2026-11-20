"""The batch loop shared by every experiment stage.

The loop is deliberately identical across extraction, classification and the
end-to-end baseline so that differences between tables come from the model and
the prompt, never from the harness:

1. stream results to disk as soon as each case finishes;
2. skip cases already recorded as successful (resume after interruption);
3. retry failures in dedicated rounds;
4. rewrite the file in input order, inserting error placeholders for cases
   that never succeeded so downstream alignment stays positional.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from . import io

ProcessFn = Callable[[Dict[str, Any], int], Optional[Dict[str, Any]]]
SuccessFn = Callable[[Dict[str, Any]], bool]
PlaceholderFn = Callable[[Dict[str, Any], int, str], Dict[str, Any]]


def run_batch(
    items: List[Dict[str, Any]],
    output_path: str,
    process_one: ProcessFn,
    is_success: SuccessFn,
    make_placeholder: PlaceholderFn,
    rerun_rounds: int = 2,
    limit: Optional[int] = None,
    quiet: bool = False,
) -> Dict[str, int]:
    """Execute ``process_one`` over ``items`` with resume and retry support."""
    if limit is not None:
        items = items[:limit]

    total = len(items)
    rows = io.load_existing(output_path)
    done = {index for index, row in rows.items() if is_success(row)}

    if not quiet:
        print(f"total={total} already_done={len(done)} pending={total - len(done)}")

    started = time.time()
    failed: List[int] = []

    for index, item in enumerate(items):
        if index in done:
            continue

        try:
            result = process_one(item, index)
        except Exception as exc:  # noqa: BLE001 - keep the run alive on single-case errors
            if not quiet:
                print(f"[warn] case {index} raised: {exc}")
            result = None

        if result is None:
            failed.append(index)
            io.append_jsonl(output_path, make_placeholder(item, index, "first round failed"))
            continue

        rows[index] = result
        io.append_jsonl(output_path, result)

    for round_id in range(1, rerun_rounds + 1):
        if not failed:
            break

        current, failed = failed, []
        if not quiet:
            print(f"retry round {round_id}: {len(current)} case(s)")

        for index in current:
            item = items[index]
            try:
                result = process_one(item, index)
            except Exception as exc:  # noqa: BLE001
                if not quiet:
                    print(f"[warn] case {index} raised on retry: {exc}")
                result = None

            if result is None:
                failed.append(index)
                continue

            rows[index] = result
            io.append_jsonl(output_path, result)

    for index in failed:
        rows[index] = make_placeholder(items[index], index, "failed after all rounds")

    io.rewrite_in_order(rows, items, output_path, lambda item, idx: make_placeholder(item, idx, "missing"))

    elapsed = time.time() - started
    stats = {
        "total": total,
        "skipped": len(done),
        "failed": len(failed),
        "seconds": int(elapsed),
    }

    if not quiet:
        rate = total / elapsed if elapsed > 0 else 0.0
        print(f"finished: {stats['failed']} failure(s), {elapsed:.1f}s, {rate:.2f} case/s")
        print(f"output: {output_path}")

    return stats
