# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache

import torch
from vllm.logger import init_logger
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend import envs
from vllm_ascend.utils import enable_custom_op

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


@cache
def _get_mega_chunk_gdn_op():
    if not enable_custom_op() or not hasattr(torch.ops._C_ascend, "npu_mega_chunk_gdn"):
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_MEGA_CHUNK_GDN=1 requires vLLM Ascend custom operators built for Ascend910B."
        )
    logger.info("Using MegaChunkGdn for supported GDN prefill batches.")
    return torch.ops._C_ascend.npu_mega_chunk_gdn


@cache
def _get_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ones = torch.ones((_CHUNK_SIZE, _CHUNK_SIZE), device=device, dtype=torch.float32)
    return ones.tril(-1), ones.tril(), -torch.eye(_CHUNK_SIZE, device=device, dtype=torch.bfloat16)


def log_mega_chunk_gdn_comparison(
    layer_name: str,
    output: torch.Tensor,
    final_state: torch.Tensor,
    reference_output: torch.Tensor,
    reference_state: torch.Tensor,
) -> None:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    metrics = []
    for actual, reference in ((output, reference_output), (final_state, reference_state)):
        reference = reference.float()
        delta = actual.float() - reference
        metrics.extend((delta.abs().amax(), delta.square().sum(), reference.square().sum()))
    out_max, out_error, out_norm, state_max, state_error, state_norm = torch.stack(metrics).cpu().tolist()
    logger.warning(
        "MegaChunkGdn compare rank=%s layer=%s output_max_abs=%.6g output_rel_l2=%.6g "
        "state_max_abs=%.6g state_rel_l2=%.6g",
        rank,
        layer_name,
        out_max,
        (out_error / max(out_norm, 1e-12)) ** 0.5,
        state_max,
        (state_error / max(state_norm, 1e-12)) ** 0.5,
    )


def mega_chunk_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...],
    *,
    layer_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    op = _get_mega_chunk_gdn_op()
    num_value_heads = v.shape[2]
    num_chunks = sum(
        (end - start + _CHUNK_SIZE - 1) // _CHUNK_SIZE for start, end in zip(cu_seqlens_host, cu_seqlens_host[1:])
    )
    num_matrices = num_chunks * num_value_heads
    stream = None
    if envs.VLLM_ASCEND_MEGA_CHUNK_GDN_DEBUG:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        stream = torch.npu.current_stream(v.device)
        tag = f"MegaChunkGdn rank={rank} device={v.device} layer={layer_name}"
        logger.warning(
            "%s phase=wait_prior q=%s v=%s state=%s host_starts=%s num_matrices=%d",
            tag,
            tuple(q.shape),
            tuple(v.shape),
            tuple(initial_state.shape),
            cu_seqlens_host,
            num_matrices,
        )
        stream.synchronize()
        device_starts = tuple(cu_seqlens.cpu().tolist())
        logger.warning("%s phase=prepare_inputs device_starts=%s", tag, device_starts)
        if device_starts != cu_seqlens_host:
            raise RuntimeError(f"{tag} sequence boundaries differ: host={cu_seqlens_host}, device={device_starts}")

    # The cache uses [V, K] while the xLLM kernel consumes and returns [K, V].
    state = initial_state.transpose(-1, -2).to(v.dtype).contiguous()
    args = (
        l2norm_fwd(q.contiguous()).contiguous(),
        l2norm_fwd(k.contiguous()).contiguous(),
        v.contiguous(),
        g.to(torch.float32).contiguous(),
        beta.to(v.dtype).contiguous(),
        *_get_masks(v.device),
        cu_seqlens.to(torch.int32).contiguous(),
        state,
        num_matrices,
    )
    if stream is not None:
        stream.synchronize()
        logger.warning("%s phase=launch", tag)
    out, final_state = op(*args)
    if stream is not None:
        logger.warning("%s phase=wait_kernel", tag)
        stream.synchronize()
        logger.warning("%s phase=complete", tag)
    return out * (q.shape[-1] ** -0.5), final_state.transpose(-1, -2)
