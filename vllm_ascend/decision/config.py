# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Versioned configuration shared by the exporter, workers, and HTTP endpoint."""

import math
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class KevConfig:
    schema_version: int = 2
    head_dim: int = 256
    max_context: int = 16384
    max_questions: int = 64
    max_parent_tokens: int = 131072
    max_concurrent_parents: int = 32
    question_concurrency: int = 8
    timeout_seconds: float = 120.0
    temperature: float = 1.0
    strict_length: bool = False
    date_facts: bool = False
    dp_affinity: bool = True

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise TypeError("Exported model must contain a kev_config object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown kev_config fields: {sorted(unknown)}")
        result = cls(**data)
        if result.schema_version != 2:
            raise ValueError("Unsupported Kev schema version")
        for name in (
            "head_dim",
            "max_context",
            "max_questions",
            "max_parent_tokens",
            "max_concurrent_parents",
            "question_concurrency",
        ):
            value = getattr(result, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if result.max_context < 2:
            raise ValueError("max_context must be at least two")
        for name in ("timeout_seconds", "temperature"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("strict_length", "date_facts", "dp_affinity"):
            if type(getattr(result, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        return result


def validate_runtime(config):
    """Reject combinations that cannot preserve the first implementation's contract."""
    if config.model_config.runner_type != "pooling":
        raise ValueError("Kev requires --runner pooling")
    if config.use_v2_model_runner:
        raise ValueError("Kev currently requires VLLM_USE_V2_MODEL_RUNNER=0")
    if not config.model_config.enforce_eager:
        raise ValueError("Kev currently requires --enforce-eager")
    hybrid = config.model_config.is_hybrid
    mode = config.cache_config.mamba_cache_mode
    if hybrid and config.cache_config.enable_prefix_caching:
        if mode != "align" or not config.scheduler_config.enable_chunked_prefill:
            raise ValueError("Kev hybrid APC requires mamba-cache-mode align and chunked prefill")
    elif mode not in (None, "none"):
        raise ValueError("Without hybrid APC, Kev requires mamba-cache-mode none")
    if config.scheduler_config.async_scheduling:
        raise ValueError("Kev currently requires --no-async-scheduling")
    if config.speculative_config is not None or config.lora_config is not None:
        raise ValueError("Kev requires merged weights without speculative decoding or dynamic LoRA")
    if config.quant_config is not None:
        raise ValueError("Kev quantization has not been enabled")
    if config.kv_transfer_config is not None:
        raise ValueError("Kev does not yet support KV transfer")
    parallel = config.parallel_config
    if parallel.pipeline_parallel_size != 1:
        raise ValueError("Kev pipeline parallelism is not yet enabled")
    for name in ("decode_context_parallel_size", "prefill_context_parallel_size"):
        if getattr(parallel, name, 1) != 1:
            raise ValueError("Kev context parallelism is not yet enabled")
    if config.compilation_config.pass_config.enable_sp:
        raise ValueError("Kev sequence parallelism is not yet enabled")


KEV_ARCHITECTURES = (
    "AscendKevQwen2ForDecision",
    "AscendKevQwen3ForDecision",
    "AscendKevQwen35ForDecision",
)


def is_kev_model(model_config):
    return any(name in KEV_ARCHITECTURES for name in model_config.architectures)
