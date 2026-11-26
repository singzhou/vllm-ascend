# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
from functools import wraps

import vllm.envs as envs_vllm
import vllm.v1.attention.selector as attention_selector
import vllm.v1.worker.gpu.spec_decode.dflash.cudagraph as cudagraph_module
import vllm.v1.worker.gpu.spec_decode.dflash.speculator as speculator_module
import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as speculator2_module
from vllm.compilation import backends
from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.kv_cache_interface import KVCacheSpecKind

from vllm_ascend import envs
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.fa4_v1 import AscendFA4WindowBackend, fa4_target_compatible
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    _selector_walk_kernel_ascend,
)

_original_cached_get_attn_backend = attention_selector._cached_get_attn_backend


@wraps(_original_cached_get_attn_backend)
def _get_dflash2_attn_backend(
    backend: AttentionBackendEnum | None,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    default_backend = _original_cached_get_attn_backend(
        backend=backend, attn_selector_config=attn_selector_config, num_heads=num_heads
    )
    if (
        backends.model_tag != "dflash_head"
        or not attn_selector_config.has_sliding_window
        or default_backend is not AscendAttentionBackend
    ):
        return default_backend

    config = get_current_vllm_config()
    spec = config.speculative_config
    if (
        not config.use_v2_model_runner
        or spec is None
        or not spec.parallel_drafting
        or spec.draft_model_config is None
        or "DFlash2DraftModel" not in (spec.draft_model_config.architectures or ())
        or spec.attention_backend not in (None, AttentionBackendEnum.CUSTOM)
    ):
        return default_backend

    target_fa4 = (
        envs.VLLM_ASCEND_FA4_TARGET_ATTN
        and config.attention_config.backend_per_kind.get(KVCacheSpecKind.FULL_ATTENTION.value)
        is AttentionBackendEnum.CUSTOM
    )
    if (
        (target_fa4 or envs.VLLM_ASCEND_ENABLE_DSPARK_FIA_SINK == 2)
        and envs_vllm.VLLM_KV_CACHE_LAYOUT not in ("LBHNC", "HND")
        and get_current_hardware_profile().supports(HardwareCapability.FA4_CAUSAL_DEVICE_LENGTHS)
        and fa4_target_compatible(attn_selector_config, windowed_draft=True)
    ):
        return AscendFA4WindowBackend
    return default_backend


# Keep the draft decision outside vLLM's selector cache. The cache key can be
# identical for the target and draft, even when their attention differs.
_get_dflash2_attn_backend.cache_clear = _original_cached_get_attn_backend.cache_clear
_get_dflash2_attn_backend.cache_info = _original_cached_get_attn_backend.cache_info
attention_selector._cached_get_attn_backend = _get_dflash2_attn_backend

cudagraph_module.build_attn_metadata = build_attn_metadata
speculator_module.DFlashCudaGraphManager = DFlashAclGraphManager

# triton-ascend cannot lower tldevice.log1p in the upstream selector walk;
# swap in the algebraically equivalent log(1 - u) variant.
speculator2_module._selector_walk_kernel = _selector_walk_kernel_ascend
