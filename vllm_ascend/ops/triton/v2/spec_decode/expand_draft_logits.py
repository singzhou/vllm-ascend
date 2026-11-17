# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expand reduced-vocabulary proposals using contiguous NPU writes."""

import torch
from vllm.triton_utils import tl, triton

_OUTPUT_BLOCK_SIZE = 4096


def make_draft_vocab_block_starts(target_ids: torch.Tensor, target_vocab_size: int) -> torch.Tensor | None:
    """Plan source windows once at load time; unordered maps use the generic path."""
    ids_cpu = target_ids.to(device="cpu", dtype=torch.int64)
    if ids_cpu.numel() > 1 and not bool(torch.all(ids_cpu[1:] > ids_cpu[:-1])):
        return None
    boundaries = torch.arange(0, target_vocab_size, _OUTPUT_BLOCK_SIZE, dtype=torch.int64)
    return torch.searchsorted(ids_cpu, boundaries).to(device=target_ids.device, dtype=torch.int32)


def make_target_to_draft_map(target_ids: torch.Tensor, target_vocab_size: int) -> torch.Tensor:
    """Build the fixed inverse map once when the reduced draft is loaded."""
    inverse = torch.full((target_vocab_size,), -1, dtype=torch.int32, device=target_ids.device)
    inverse.index_copy_(0, target_ids, torch.arange(target_ids.numel(), dtype=torch.int32, device=target_ids.device))
    return inverse


@triton.jit
def _expand_draft_logits_kernel(
    output_ptr,
    logits_ptr,
    inverse_ptr,
    output_row_stride,
    output_col_stride,
    logits_row_stride,
    DRAFT_VOCAB_SIZE: tl.constexpr,
    TARGET_VOCAB_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    indices = tl.load(inverse_ptr + cols, cols < TARGET_VOCAB_SIZE, other=-1)
    lanes = tl.arange(0, CHUNK_SIZE)
    values = tl.full((BLOCK_SIZE,), 0, logits_ptr.dtype.element_ty)
    # Gather within UB, rather than issuing a scalar GM load/store for every
    # vocabulary entry. Chunking keeps all supported dtypes within NPU UB.
    for chunk in range(triton.cdiv(DRAFT_VOCAB_SIZE, CHUNK_SIZE)):
        offsets = chunk * CHUNK_SIZE + lanes
        source = tl.load(
            logits_ptr + row * logits_row_stride + offsets,
            offsets < DRAFT_VOCAB_SIZE,
            other=0,
        )
        local_indices = tl.minimum(tl.maximum(indices - chunk * CHUNK_SIZE, 0), CHUNK_SIZE - 1)
        gathered = tl.gather(source, local_indices, axis=0)
        in_chunk = (indices >= chunk * CHUNK_SIZE) & (indices < (chunk + 1) * CHUNK_SIZE)
        values = tl.where(in_chunk, gathered, values)
    tl.store(
        output_ptr + row * output_row_stride + cols * output_col_stride,
        tl.where(indices >= 0, values, float("-inf")),
        cols < TARGET_VOCAB_SIZE,
    )


@triton.jit
def _expand_draft_logits_window_kernel(
    output_ptr,
    logits_ptr,
    inverse_ptr,
    starts_ptr,
    output_row_stride,
    output_col_stride,
    logits_row_stride,
    DRAFT_VOCAB_SIZE: tl.constexpr,
    TARGET_VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    lanes = tl.arange(0, BLOCK_SIZE)
    cols = block * BLOCK_SIZE + lanes
    indices = tl.load(inverse_ptr + cols, cols < TARGET_VOCAB_SIZE, other=-1)
    start = tl.load(starts_ptr + block)
    offsets = start + lanes
    source = tl.load(logits_ptr + row * logits_row_stride + offsets, offsets < DRAFT_VOCAB_SIZE, other=0)
    # Strictly increasing target ids put every live source index in this
    # consecutive window of at most BLOCK_SIZE entries.
    local_indices = tl.minimum(tl.maximum(indices - start, 0), BLOCK_SIZE - 1)
    values = tl.gather(source, local_indices, axis=0)
    tl.store(
        output_ptr + row * output_row_stride + cols * output_col_stride,
        tl.where(indices >= 0, values, float("-inf")),
        cols < TARGET_VOCAB_SIZE,
    )


def expand_draft_logits_(
    output: torch.Tensor,
    inverse: torch.Tensor,
    logits: torch.Tensor,
    block_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write target-space logits, assigning ``-inf`` to every unmapped token."""
    num_rows, draft_vocab_size = logits.shape
    target_vocab_size = output.shape[1]
    if num_rows and draft_vocab_size:
        # Markov logits are contiguous in the hot path. Normalize other
        # callers before the UB gather to keep strided-load tiling bounded.
        if logits.stride(1) != 1:
            logits = logits.contiguous()
        block_size = _OUTPUT_BLOCK_SIZE
        if block_starts is not None:
            _expand_draft_logits_window_kernel[(num_rows, triton.cdiv(target_vocab_size, block_size))](
                output,
                logits,
                inverse,
                block_starts,
                output.stride(0),
                output.stride(1),
                logits.stride(0),
                DRAFT_VOCAB_SIZE=draft_vocab_size,
                TARGET_VOCAB_SIZE=target_vocab_size,
                BLOCK_SIZE=block_size,
            )
            return output
        _expand_draft_logits_kernel[(num_rows, triton.cdiv(target_vocab_size, block_size))](
            output,
            logits,
            inverse,
            output.stride(0),
            output.stride(1),
            logits.stride(0),
            DRAFT_VOCAB_SIZE=draft_vocab_size,
            TARGET_VOCAB_SIZE=target_vocab_size,
            CHUNK_SIZE=min(16384, triton.next_power_of_2(draft_vocab_size)),
            BLOCK_SIZE=block_size,
        )
    return output
