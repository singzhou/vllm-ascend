# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bound Kev APC reads before lookup without altering scheduler token accounting.

Remove this adapter once upstream exposes a per-request prefix-read limit.
Only requests carrying validated Kev readout metadata enter this path. Shared
blocks, references, eviction and hybrid boundary selection remain upstream.
"""

from functools import wraps

from vllm.config import ModelConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager

from vllm_ascend.decision.config import is_kev_model
from vllm_ascend.decision.planner import ReadoutSpec


class _PrefixLookupView:
    """Read-only Request view for upstream's prompt_length - 1 lookup bound."""

    def __init__(self, request, prefix_length):
        self._request = request
        self.num_tokens = min(request.num_tokens, prefix_length + 1)

    def __getattr__(self, name):
        return getattr(self._request, name)


def _read_limit(request):
    params = request.pooling_params
    if params is None or params.task != "plugin":
        return None
    extra = params.extra_kwargs or {}
    if "kev_readout" not in extra:
        return None
    return ReadoutSpec.parse(extra["kev_readout"], request.num_prompt_tokens).state_length


def _install():
    original_lookup = KVCacheManager.get_computed_blocks
    if getattr(original_lookup, "_ascend_kev_bounded", False):
        return

    @wraps(original_lookup)
    def get_computed_blocks(manager, request):
        limit = _read_limit(request)
        if limit is None:
            return original_lookup(manager, request)
        result = original_lookup(manager, _PrefixLookupView(request, limit))
        if result[1] > limit or result[2] > limit:
            raise RuntimeError("Kev prefix lookup exceeded the readout-safe boundary")
        return result

    get_computed_blocks._ascend_kev_bounded = True
    KVCacheManager.get_computed_blocks = get_computed_blocks

    original_connector = KVCacheManager.get_computed_blocks_for_connector

    @wraps(original_connector)
    def get_computed_blocks_for_connector(manager, request):
        if _read_limit(request) is not None:
            raise ValueError("Kev remote KV transfer is not yet supported")
        return original_connector(manager, request)

    KVCacheManager.get_computed_blocks_for_connector = get_computed_blocks_for_connector

    original_capability = ModelConfig.is_prefix_caching_supported.fget

    def is_prefix_caching_supported(model_config):
        if is_kev_model(model_config) and model_config.attn_type == "hybrid":
            # Native GDN writes its state in forward. No speculative decoding
            # postprocess is needed; chunk alignment and cache ownership remain
            # the responsibility of the existing hybrid scheduler/coordinator.
            return True
        return original_capability(model_config)

    ModelConfig.is_prefix_caching_supported = property(is_prefix_caching_supported)


_install()
