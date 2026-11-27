# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from __future__ import annotations

import math

import torch
from vllm.triton_utils import tl, triton

STATE_IO_BLOCK_SIZE = 1024
TRANSPOSE_STATE_DIM = 128
TRANSPOSE_STATE_TILE = 64


@triton.jit
def _gather_ssm_states_kernel(
    state_ptr,
    indices_ptr,
    has_initial_state_ptr,
    output_ptr,
    stride_state_batch: tl.int64,
    stride_indices,
    stride_has_initial_state,
    row_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size

    has_initial_state = tl.load(has_initial_state_ptr + batch_idx * stride_has_initial_state).to(tl.int1)
    state_idx = tl.load(
        indices_ptr + batch_idx * stride_indices,
        mask=has_initial_state,
        other=0,
    ).to(tl.int64)

    values = tl.load(
        state_ptr + state_idx * stride_state_batch + offsets,
        mask=mask & has_initial_state,
        other=0.0,
    )
    tl.store(output_ptr + batch_idx * row_size + offsets, values, mask=mask)


@triton.jit
def _scatter_ssm_states_kernel(
    state_ptr,
    indices_ptr,
    source_ptr,
    valid_rows_ptr,
    stride_state_batch: tl.int64,
    stride_indices,
    stride_source_batch: tl.int64,
    row_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_VALID_ROWS: tl.constexpr,
):
    block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    if HAS_VALID_ROWS:
        if not tl.load(valid_rows_ptr + batch_idx):
            return
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size

    state_idx = tl.load(indices_ptr + batch_idx * stride_indices).to(tl.int64)
    values = tl.load(
        source_ptr + batch_idx * stride_source_batch + offsets,
        mask=mask,
    )
    tl.store(
        state_ptr + state_idx * stride_state_batch + offsets,
        values,
        mask=mask,
    )


@triton.jit
def _copy_transposed_ssm_states_kernel(
    cache,
    packed,
    indices,
    valid_rows,
    stride_cache_batch: tl.int64,
    stride_packed_batch: tl.int64,
    stride_indices,
    stride_valid_rows,
    DIM: tl.constexpr,
    TILE: tl.constexpr,
    GATHER: tl.constexpr,
    HAS_VALID_ROWS: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    tiles_per_dim = DIM // TILE
    head_offset = tile // (tiles_per_dim * tiles_per_dim) * DIM * DIM
    m = tile // tiles_per_dim % tiles_per_dim * TILE + tl.arange(0, TILE)
    n = tile % tiles_per_dim * TILE + tl.arange(0, TILE)
    valid = True
    if HAS_VALID_ROWS:
        valid = tl.load(valid_rows + row * stride_valid_rows).to(tl.int1)
    index = tl.load(indices + row * stride_indices, mask=valid, other=0).to(tl.int64)
    cache_base = cache + index * stride_cache_batch + head_offset
    packed_base = packed + row * stride_packed_batch + head_offset
    input_offsets = m[:, None] * DIM + n[None, :]
    output_offsets = n[:, None] * DIM + m[None, :]
    if GATHER:
        values = tl.load(cache_base + input_offsets, mask=valid, other=0)
        tl.store(packed_base + output_offsets, tl.trans(values))
    else:
        values = tl.load(packed_base + input_offsets, mask=valid, other=0)
        tl.store(cache_base + output_offsets, tl.trans(values), mask=valid)


def _copy_transposed_ssm_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    packed: torch.Tensor,
    valid_rows: torch.Tensor | None,
    *,
    gather: bool,
) -> None:
    if state.ndim != 4 or state.shape[-2:] != (TRANSPOSE_STATE_DIM, TRANSPOSE_STATE_DIM):
        raise ValueError("Transposed state copies require [rows, heads, 128, 128] state tensors")
    tiles_per_head = (TRANSPOSE_STATE_DIM // TRANSPOSE_STATE_TILE) ** 2
    grid = (state.shape[1] * tiles_per_head, indices.numel())
    _copy_transposed_ssm_states_kernel[grid](
        state,
        packed,
        indices,
        valid_rows,
        state.stride(0),
        packed.stride(0),
        indices.stride(0),
        valid_rows.stride(0) if valid_rows is not None else 0,
        DIM=TRANSPOSE_STATE_DIM,
        TILE=TRANSPOSE_STATE_TILE,
        GATHER=gather,
        HAS_VALID_ROWS=valid_rows is not None,
        num_warps=8,
    )


def _validate_state_and_indices(
    state: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    if state.ndim < 2:
        raise ValueError(f"state must have at least 2 dimensions, got {state.ndim}")
    if state.shape[0] == 0:
        raise ValueError("state must contain at least one row")
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape={tuple(indices.shape)}")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"indices must be int32 or int64, got {indices.dtype}")
    if indices.device != state.device:
        raise ValueError("indices and state must be on the same device")
    row_size = math.prod(state.shape[1:])
    if state.stride(0) < row_size:
        raise ValueError("state stride(0) must be at least the flattened row size")
    if not state[0].is_contiguous():
        raise ValueError("each state row must be contiguous")


def _validate_gather_inputs(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> None:
    _validate_state_and_indices(state, indices)
    if has_initial_state.ndim != 1 or has_initial_state.shape != indices.shape:
        raise ValueError("has_initial_state must be 1D and match indices")
    if has_initial_state.dtype != torch.bool:
        raise TypeError("has_initial_state must have bool dtype")
    if has_initial_state.device != state.device:
        raise ValueError("has_initial_state and state must be on the same device")


def _validate_scatter_inputs(
    state: torch.Tensor,
    indices: torch.Tensor,
    source: torch.Tensor,
) -> None:
    _validate_state_and_indices(state, indices)
    expected_shape = (indices.numel(), *state.shape[1:])
    if source.shape != expected_shape:
        raise ValueError(f"source shape must be {expected_shape}, got {tuple(source.shape)}")
    if source.device != state.device:
        raise ValueError("source and state must be on the same device")
    if source.numel() > 0 and not source[0].is_contiguous():
        raise ValueError("each source row must be contiguous")


def gather_ssm_states(
    state: torch.Tensor,
    indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
    transpose_last_two: bool = False,
) -> torch.Tensor:
    """Gather selected rows from a state cache with a padded batch stride.

    Unlike ``state[indices]`` or ``index_select``, this kernel reads only the
    requested rows. ``has_initial_state=False`` produces a zero row without
    reading the state cache.

    ``state`` has shape ``[num_state_rows, *inner_dims]``. The inner
    dimensions must be contiguous; only ``state.stride(0)`` may include
    physical page padding. ``indices`` select physical state rows and may be
    unsorted or non-contiguous. Callers must provide valid row IDs. The
    optional ``output_dtype`` only changes the gathered output dtype.
    ``transpose_last_two`` also transposes each 128 by 128 head matrix.
    """
    _validate_gather_inputs(state, indices, has_initial_state)

    output = torch.empty(
        (indices.numel(), *state.shape[1:]),
        dtype=state.dtype if output_dtype is None else output_dtype,
        device=state.device,
    )
    if indices.numel() == 0:
        return output
    if transpose_last_two:
        _copy_transposed_ssm_states(state, indices, output, has_initial_state, gather=True)
        return output

    row_size = math.prod(state.shape[1:])
    grid = (triton.cdiv(row_size, STATE_IO_BLOCK_SIZE), indices.numel())
    _gather_ssm_states_kernel[grid](
        state,
        indices,
        has_initial_state,
        output,
        state.stride(0),
        indices.stride(0),
        has_initial_state.stride(0),
        row_size=row_size,
        BLOCK_SIZE=STATE_IO_BLOCK_SIZE,
        num_warps=8,
    )
    return output


def scatter_ssm_states_(
    state: torch.Tensor,
    indices: torch.Tensor,
    source: torch.Tensor,
    *,
    valid_rows: torch.Tensor | None = None,
    transpose_last_two: bool = False,
) -> None:
    """Write dense state rows into selected slots of a padded state cache.

    The state layout follows :func:`gather_ssm_states`: inner dimensions are
    contiguous and padding is represented by ``state.stride(0)``. ``indices``
    select physical rows and may be unsorted or non-contiguous. ``source``
    must have the logical state shape and contiguous inner dimensions.
    Optional ``valid_rows`` skips inactive rows without reading their indices.
    ``transpose_last_two`` also transposes each 128 by 128 source head matrix.
    """
    _validate_scatter_inputs(state, indices, source)
    if valid_rows is not None:
        _validate_gather_inputs(state, indices, valid_rows)
        if not valid_rows.is_contiguous():
            raise ValueError("valid_rows must be contiguous")
    if indices.numel() == 0:
        return
    if transpose_last_two:
        _copy_transposed_ssm_states(state, indices, source, valid_rows, gather=False)
        return

    row_size = math.prod(state.shape[1:])
    grid = (triton.cdiv(row_size, STATE_IO_BLOCK_SIZE), indices.numel())
    _scatter_ssm_states_kernel[grid](
        state,
        indices,
        source,
        valid_rows,
        state.stride(0),
        indices.stride(0),
        source.stride(0),
        row_size=row_size,
        BLOCK_SIZE=STATE_IO_BLOCK_SIZE,
        HAS_VALID_ROWS=valid_rows is not None,
        num_warps=8,
    )
