"""Lightweight toolkit for knowledge-guided legal text classification.

The package bundles the reusable pieces shared by every experiment stage:

* :mod:`legalsem.io`       -- JSONL reading/writing with resumable output
* :mod:`legalsem.parsing`  -- tolerant JSON / label extraction from model output
* :mod:`legalsem.prompts`  -- prompt bundle loading and placeholder filling
* :mod:`legalsem.labels`   -- conflict label space and candidate reduction
* :mod:`legalsem.backends` -- pluggable LLM backends (remote API / local vLLM)
* :mod:`legalsem.runner`   -- the batch loop shared by all tasks
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
