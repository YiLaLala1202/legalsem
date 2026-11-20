"""Pluggable LLM backends.

Three execution modes are supported, all behind one ``generate`` call:

``api``
    Any OpenAI-compatible HTTP endpoint (hosted vendor API, third-party relay,
    or a locally served ``vllm serve``). Credentials are read from an
    environment variable, never from the source tree.

``vllm``
    In-process offline inference with vLLM, for GPUs you control. Imported
    lazily so that the package remains usable without vLLM installed.

Sampling parameters follow the paper's fixed setting: greedy decoding
(``temperature=0``) for classification, and ``temperature=0`` / ``top_p=0.1``
for extraction.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class BackendError(RuntimeError):
    """Raised when a backend cannot be constructed or a request fails."""


@dataclass
class SamplingConfig:
    temperature: float = 0.0
    top_p: float = 0.1
    max_tokens: int = 1024

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SamplingConfig":
        data = data or {}
        return cls(
            temperature=float(data.get("temperature", 0.0)),
            top_p=float(data.get("top_p", 0.1)),
            max_tokens=int(data.get("max_tokens", 1024)),
        )


class BaseBackend:
    """Interface implemented by every backend."""

    name = "base"

    def __init__(self, model: str, sampling: Optional[SamplingConfig] = None) -> None:
        self.model = model
        self.sampling = sampling or SamplingConfig()

    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources; overridden where needed."""


class OpenAIChatBackend(BaseBackend):
    """OpenAI-compatible chat completions over HTTP."""

    name = "api"

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        sampling: Optional[SamplingConfig] = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: float = 180.0,
    ) -> None:
        super().__init__(model, sampling)

        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise BackendError(
                f"environment variable {api_key_env!r} is empty; "
                f"export it before running (see README, section Configuration)"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BackendError("the 'openai' package is required for API backends") from exc

        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def generate(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.sampling.temperature,
                    top_p=self.sampling.top_p,
                    max_tokens=self.sampling.max_tokens,
                    stream=False,
                )
                content = response.choices[0].message.content
                return content or ""
            except Exception as exc:  # noqa: BLE001 - backend retries on any failure
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)

        raise BackendError(f"request failed after {self.max_retries} attempts: {last_error}")


class VLLMOfflineBackend(BaseBackend):
    """In-process vLLM generation on local GPUs."""

    name = "vllm"

    def __init__(
        self,
        model: str,
        tokenizer_path: Optional[str] = None,
        sampling: Optional[SamplingConfig] = None,
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "float16",
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        super().__init__(model, sampling)

        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise BackendError(
                "the 'vllm' and 'transformers' packages are required for local backends"
            ) from exc

        self.enable_thinking = enable_thinking
        tokenizer_id = tokenizer_path or model

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer_id,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            max_tokens=self.sampling.max_tokens,
        )

    def _apply_template(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Qwen3-style tokenizers accept enable_thinking; others do not.
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, system: str, user: str) -> str:
        prompt = self._apply_template(system, user)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text.strip()


def build_backend(
    kind: str,
    model: str,
    sampling: Optional[SamplingConfig] = None,
    **kwargs: Any,
) -> BaseBackend:
    """Factory dispatching on the backend kind recorded in a config file."""
    kind = (kind or "api").lower()

    if kind in {"api", "openai", "openai-compatible"}:
        allowed = {
            key: kwargs[key]
            for key in ("base_url", "api_key_env", "max_retries", "timeout")
            if key in kwargs and kwargs[key] is not None
        }
        return OpenAIChatBackend(model=model, sampling=sampling, **allowed)

    if kind in {"vllm", "local"}:
        allowed = {
            key: kwargs[key]
            for key in (
                "tokenizer_path",
                "tensor_parallel_size",
                "max_model_len",
                "gpu_memory_utilization",
                "dtype",
                "trust_remote_code",
                "enable_thinking",
            )
            if key in kwargs and kwargs[key] is not None
        }
        return VLLMOfflineBackend(model=model, sampling=sampling, **allowed)

    raise BackendError(f"unknown backend kind: {kind!r}")
