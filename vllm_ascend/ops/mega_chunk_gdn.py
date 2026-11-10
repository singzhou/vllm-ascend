# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache, lru_cache

import torch
from vllm.logger import init_logger
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

logger = init_logger(__name__)

_CHUNK_SIZE = 128
_HEAD_DIM = 128
_MAX_VALUE_HEADS = 64


def supports_mega_chunk_gdn(q: torch.Tensor, v: torch.Tensor, cu_seqlens_host: tuple[int, ...]) -> bool:
    return (
        q.device.type == "npu"
        and q.dtype == v.dtype == torch.bfloat16
        and q.shape[-1] == v.shape[-1] == _HEAD_DIM
        and 0 < v.shape[2] <= _MAX_VALUE_HEADS
        and v.shape[2] % q.shape[2] == 0
        and len(cu_seqlens_host) >= 2
        and cu_seqlens_host[0] == 0
        and cu_seqlens_host[-1] == q.shape[1]
        and all(end > start for start, end in zip(cu_seqlens_host, cu_seqlens_host[1:]))
    )


@lru_cache(maxsize=1)
def _get_mega_chunk_gdn_op():
    try:
        from custom_ops_lib import mega_chunk_gdn
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_MEGA_CHUNK_GDN=1 requires the xLLM custom_ops_lib "
            "binding and the MegaChunkGdn Ascend910B operator package."
        ) from exc
    logger.info("Using xLLM MegaChunkGdn for supported GDN prefill batches.")
    return mega_chunk_gdn


@cache
def _get_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ones = torch.ones((_CHUNK_SIZE, _CHUNK_SIZE), device=device, dtype=torch.float32)
    return ones.tril(-1), ones.tril(), -torch.eye(_CHUNK_SIZE, device=device, dtype=torch.bfloat16)


def mega_chunk_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    op = _get_mega_chunk_gdn_op()
    num_value_heads = v.shape[2]
    num_chunks = sum(
        (end - start + _CHUNK_SIZE - 1) // _CHUNK_SIZE for start, end in zip(cu_seqlens_host, cu_seqlens_host[1:])
    )
    # The cache uses [V, K] while the xLLM kernel consumes and returns [K, V].
    state = initial_state.transpose(-1, -2).to(v.dtype).contiguous()
    out, *_, final_state = op(
        l2norm_fwd(q.contiguous()).contiguous(),
        l2norm_fwd(k.contiguous()).contiguous(),
        v.contiguous(),
        g.to(torch.float32).contiguous(),
        beta.to(v.dtype).contiguous(),
        *_get_masks(v.device),
        cu_seqlens.to(torch.int32).contiguous(),
        state,
        num_chunks * num_value_heads,
        True,
    )
    return out * (q.shape[-1] ** -0.5), final_state.view_as(state).transpose(-1, -2)
