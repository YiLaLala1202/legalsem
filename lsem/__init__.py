"""Lightweight toolkit for knowledge-guided legal text classification.

The package bundles the reusable pieces shared by every experiment stage:

* :mod:`lsem.io`       -- JSONL reading/writing with resumable output
* :mod:`lsem.parsing`  -- tolerant JSON / label extraction from model output
* :mod:`lsem.prompts`  -- prompt bundle loading and placeholder filling
* :mod:`lsem.labels`   -- conflict label space and candidate reduction
* :mod:`lsem.backends` -- pluggable LLM backends (remote API / local vLLM)
* :mod:`lsem.runner`   -- the batch loop shared by all tasks
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
