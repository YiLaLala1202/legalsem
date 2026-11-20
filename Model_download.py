"""Download the checkpoints referenced by ``configs/backbones.yaml``.

Nothing is written outside ``--local-dir``. Optionally export a mirror endpoint
before running, e.g. ``export HF_ENDPOINT=https://hf-mirror.com``.

Examples
--------
    python Model_download.py qwen3-8b --local-dir ./models/Qwen3-8B
    python Model_download.py legalone-8b --local-dir ./models/LegalOne-8B
    python Model_download.py --list
"""

from __future__ import annotations

import argparse
import os

# repo_id, source ("hf" or "ms"), suggested local folder
CHECKPOINTS = {
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "ms", "Qwen2.5-7B"),
    "qwen3-8b": ("Qwen/Qwen3-8B", "ms", "Qwen3-8B"),
    "deepseek-r1-qwen3-8b": ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "ms", "DeepSeek-R1-0528-Qwen3-8B"),
    "legalone-8b": ("CSHaitao/LegalOne-8B", "hf", "LegalOne-8B"),
    "lawllm-7b": ("ShengbinYue/LawLLM-7B", "hf", "LawLLM-7B"),
}

ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "*.safetensors",
    "*.safetensors.index.json",
]


def download_hf(repo_id: str, local_dir: str, workers: int) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        allow_patterns=ALLOW_PATTERNS,
        max_workers=workers,
    )


def download_ms(repo_id: str, local_dir: str) -> None:
    from modelscope import snapshot_download

    snapshot_download(repo_id, cache_dir=local_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="checkpoint key, see --list")
    parser.add_argument("--local-dir", help="target directory (required unless --list)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--list", action="store_true", help="print the available keys and exit")
    args = parser.parse_args()

    if args.list or not args.name:
        for key, (repo_id, source, folder) in CHECKPOINTS.items():
            print(f"{key:22s} {repo_id:42s} source={source}  suggested=./models/{folder}")
        return 0

    if args.name not in CHECKPOINTS:
        raise SystemExit(f"unknown checkpoint {args.name!r}; use --list to see the options")

    if not args.local_dir:
        raise SystemExit("--local-dir is required")

    repo_id, source, _ = CHECKPOINTS[args.name]
    local_dir = os.path.abspath(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"downloading {repo_id} from {source} -> {local_dir}")
    if source == "hf":
        download_hf(repo_id, local_dir, args.workers)
    else:
        download_ms(repo_id, local_dir)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
