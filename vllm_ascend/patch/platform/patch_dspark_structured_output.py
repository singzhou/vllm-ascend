# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep grammar-invalid DSpark proposals available for residual sampling."""

from functools import wraps
from typing import Any

from vllm.v1.structured_output import StructuredOutputManager

_ORIGINAL_VALIDATE_ATTR = "_vllm_ascend_dspark_original_validate_tokens"
_ORIGINAL_BITMASK_ATTR = "_vllm_ascend_dspark_original_grammar_bitmask"

_original_validate_tokens = getattr(
    StructuredOutputManager, _ORIGINAL_VALIDATE_ATTR, StructuredOutputManager.validate_tokens
)
_original_grammar_bitmask = getattr(
    StructuredOutputManager, _ORIGINAL_BITMASK_ATTR, StructuredOutputManager.grammar_bitmask
)


def _uses_probabilistic_dspark(manager: StructuredOutputManager) -> bool:
    config = manager.vllm_config
    spec = getattr(config, "speculative_config", None)
    return (
        spec is not None
        and not getattr(config, "use_v2_model_runner", True)
        and spec.method == "dspark"
        and spec.draft_sample_method == "probabilistic"
        and spec.rejection_sample_method == "standard"
    )


@wraps(_original_validate_tokens)
def _validate_tokens(self: StructuredOutputManager, request: Any, spec_tokens: list[int]) -> list[int]:
    validated = _original_validate_tokens(self, request, spec_tokens)
    if (
        _uses_probabilistic_dspark(self)
        and request.use_structured_output
        and len(validated) < len(spec_tokens)
        and spec_tokens[len(validated)] >= 0
    ):
        # Truncating before this proposal would sample p directly on rejection,
        # rather than (p-q)+. Keep its real id so the cached q is still verified.
        return spec_tokens[: len(validated) + 1]
    return validated


@wraps(_original_grammar_bitmask)
def _grammar_bitmask(
    self: StructuredOutputManager,
    requests: dict[str, Any],
    structured_output_request_ids: list[str],
    scheduled_spec_decode_tokens: dict[str, list[int]],
) -> Any:
    mask_tokens = scheduled_spec_decode_tokens
    if _uses_probabilistic_dspark(self):
        for req_id in structured_output_request_ids:
            tokens = scheduled_spec_decode_tokens.get(req_id, [])
            if not tokens:
                continue
            prefix = _original_validate_tokens(self, requests[req_id], tokens)
            if len(prefix) < len(tokens) and tokens[len(prefix)] >= 0:
                if mask_tokens is scheduled_spec_decode_tokens:
                    mask_tokens = scheduled_spec_decode_tokens.copy()
                # Upstream fills this position's mask before seeing -1, then
                # skips grammar advancement and the unused suffix. Only the
                # mask-generation copy uses padding: the worker receives the
                # real proposal and rejects it with its original q and p(x)=0.
                mask_tokens[req_id] = prefix + [-1] * (len(tokens) - len(prefix))
    return _original_grammar_bitmask(self, requests, structured_output_request_ids, mask_tokens)


setattr(StructuredOutputManager, _ORIGINAL_VALIDATE_ATTR, _original_validate_tokens)
setattr(StructuredOutputManager, _ORIGINAL_BITMASK_ATTR, _original_grammar_bitmask)
StructuredOutputManager.validate_tokens = _validate_tokens
StructuredOutputManager.grammar_bitmask = _grammar_bitmask
