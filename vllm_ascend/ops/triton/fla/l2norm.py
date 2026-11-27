# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fla/ops/l2norm.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["eps", "M", "NUM_CHUNKS"])
def l2norm_fwd_kernel2_loop(X, Y, eps, M, N: tl.constexpr, MBLOCK: tl.constexpr, NUM_CHUNKS):
    base_row = tl.program_id(0) * (NUM_CHUNKS * MBLOCK)
    rindex = tl.arange(0, N)[None, :]

    for chunk in range(NUM_CHUNKS):
        row_idx = base_row + chunk * MBLOCK + tl.arange(0, MBLOCK)[:, None]
        xmask = row_idx < M

        xs = tl.load(X + (rindex + N * row_idx), mask=xmask, other=0.0).to(tl.float32)
        square = xs * xs
        square_sum = tl.sum(square, 1)[:, None]
        rsqrt = tl.rsqrt(square_sum + eps)

        tl.store(Y + (rindex + N * row_idx), xs * rsqrt, xmask)


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError(f"l2norm_fwd: This layer doesn't support feature dim >= 64KB, got {D}.")

    MBLOCK = 69
    # M, N = x.shape
    num_core = get_vectorcore_num()
    main_bs = triton.cdiv(T, num_core)
    num_sub_blocks = triton.cdiv(main_bs, MBLOCK)
    grid = (num_core,)
    l2norm_fwd_kernel2_loop[grid](
        X=x,
        Y=y,
        eps=eps,
        M=T,
        N=D,
        MBLOCK=MBLOCK,
        NUM_CHUNKS=num_sub_blocks,
    )

    return y.view(x_shape_og)


@triton.jit(do_not_specialize=["num_tokens", "stride_head", "num_chunks"])
def _l2norm_head_first_kernel(
    x,
    y,
    num_tokens,
    stride_head,
    num_chunks,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    head = tl.program_id(1)
    rows = tl.program_id(0) * num_chunks * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, HEAD_DIM)[None, :]
    for chunk in range(num_chunks):
        token = rows + chunk * BLOCK_ROWS
        values = tl.load(x + head * stride_head + token * HEAD_DIM + cols, token < num_tokens, other=0).to(tl.float32)
        normalized = values * tl.rsqrt(tl.sum(values * values, 1) + 1.0e-6)[:, None]
        tl.store(y + token * (NUM_HEADS * HEAD_DIM) + head * HEAD_DIM + cols, normalized, token < num_tokens)


def l2norm_fwd_from_head_first(x: torch.Tensor) -> torch.Tensor:
    """Normalize a [1, T, H, D] view of [H, T, D] into packed token-major rows."""
    _, num_tokens, num_heads, head_dim = x.shape
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    block_rows = 32
    grid_tokens = min(triton.cdiv(num_tokens, block_rows), max(1, get_vectorcore_num() // num_heads))
    _l2norm_head_first_kernel[(grid_tokens, num_heads)](
        x,
        output,
        num_tokens,
        x.stride(2),
        triton.cdiv(num_tokens, grid_tokens * block_rows),
        num_heads,
        head_dim,
        block_rows,
    )
    return output
