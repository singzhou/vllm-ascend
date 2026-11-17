# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduced-vocab DSpark scatter and probabilistic sampling regressions."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.v2.sample.categorical_sample import categorical_sample
from vllm_ascend.ops.triton.v2.spec_decode.expand_draft_logits import (
    expand_draft_logits_,
    make_draft_vocab_block_starts,
    make_target_to_draft_map,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("num_rows", [1, 2, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_scatter_matches_index_copy_and_preserves_unmapped_columns(num_rows, dtype, strided, ordered):
    """Cover the profiled 64K -> 248320 mapping, tails, and strided inputs."""
    torch.manual_seed(17)
    draft_vocab = 64000 if num_rows == 2 else 1025
    target_vocab = 248320 if num_rows == 2 else 4097
    target_ids_cpu = torch.randperm(target_vocab)[:draft_vocab]
    if ordered:
        target_ids_cpu = target_ids_cpu.sort().values
    target_ids = target_ids_cpu.to("npu")
    inverse = make_target_to_draft_map(target_ids, target_vocab)
    starts = make_draft_vocab_block_starts(target_ids, target_vocab)
    assert (starts is not None) == ordered
    output = torch.full((num_rows + 1, target_vocab), -float("inf"), dtype=dtype, device="npu")
    # The last row must remain untouched when a padded buffer is sliced.
    for _ in range(2):
        host = torch.randn(num_rows, draft_vocab * (2 if strided else 1), dtype=dtype)
        source = host.to("npu")
        if strided:
            host = host[:, ::2]
            source = source[:, ::2]
        expected = torch.full((num_rows + 1, target_vocab), -float("inf"), dtype=dtype)
        expected[:num_rows].index_copy_(1, target_ids_cpu, host)
        result = expand_draft_logits_(output[:num_rows], inverse, source, starts)
        assert result.data_ptr() == output.data_ptr()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("idx_mapping", [[2, 0], [2, -1]])
def test_scatter_graph_replay_preserves_sampled_ids_and_cached_distribution(idx_mapping, ordered):
    """Changing logits on replay must exactly match the former dense sampler."""
    init_device_properties_triton()
    torch.manual_seed(19)
    rows, draft_vocab, target_vocab, steps = 2, 64000, 248320, 7
    target_ids_cpu = torch.randperm(target_vocab)[:draft_vocab]
    if ordered:
        target_ids_cpu = target_ids_cpu.sort().values
    target_ids = target_ids_cpu.to("npu")
    logits = torch.randn(rows, draft_vocab, dtype=torch.bfloat16, device="npu")
    dense = torch.full((rows, target_vocab), -float("inf"), dtype=logits.dtype, device="npu")
    cache = torch.full((rows + 1, steps, target_vocab), -float("inf"), dtype=logits.dtype, device="npu")
    idx_map = torch.tensor(idx_mapping, dtype=torch.int32, device="npu")
    temperature = torch.tensor([0.7, 1.0, 0.0], device="npu")
    seeds = torch.tensor([17, 19, 23], dtype=torch.int64, device="npu")
    pos = torch.tensor([31, 43], dtype=torch.int64, device="npu")
    col = torch.tensor(3, dtype=torch.int32, device="npu")
    speculator = SimpleNamespace(
        draft_logits=cache,
        _d2t_scatter_index=target_ids,
        _target_to_draft_map=make_target_to_draft_map(target_ids, target_vocab),
        _draft_vocab_block_starts=make_draft_vocab_block_starts(target_ids, target_vocab),
        _draft_scatter_buf=dense,
        _step_cols=torch.arange(steps, dtype=torch.int32, device="npu"),
        _maybe_predict_acceptance=lambda *args: None,
        temperature=temperature,
        seeds=seeds,
        use_fp64_gumbel=False,
        draft_watermarker=None,
    )

    def sample(values, cache_buffer):
        return categorical_sample(
            values,
            idx_map,
            temperature,
            seeds,
            pos,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=cache_buffer,
            logits_cache_col=col,
        )

    def run():
        return AscendDSparkSpeculator._sample_logits(speculator, logits, idx_map, pos + 1, 3)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = run()
    for _ in range(2):
        logits.normal_()
        graph.replay()
        expected_dense = torch.full_like(dense, -float("inf"))
        expected_dense.index_copy_(1, target_ids, logits)
        expected_cache = torch.full_like(cache, -float("inf"))
        expected = sample(expected_dense, expected_cache)
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)


@pytest.mark.parametrize(
    "source_dtype,output_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.float32),
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "row_stride", "col_stride"])
@pytest.mark.parametrize("dense", [False, True])
def test_window_dtype_conversion_empty_blocks_and_layouts(source_dtype, output_dtype, layout, dense):
    """Match upstream conversion with full windows, empty blocks and masked tails."""
    draft_vocab, target_vocab = (8193, 8193) if dense else (32, 16385)
    ids_cpu = torch.arange(draft_vocab) + (0 if dense else 8192)
    ids = ids_cpu.to("npu")
    inverse = make_target_to_draft_map(ids, target_vocab)
    starts = make_draft_vocab_block_starts(ids, target_vocab)
    storage = torch.randn(2, 3, draft_vocab * 2, dtype=source_dtype, device="npu")
    if layout == "row_stride":
        logits = storage[:, 1, :draft_vocab]
    elif layout == "col_stride":
        logits = storage[:, 1, ::2]
    else:
        logits = storage[:, 1, :draft_vocab].contiguous()
    # Exercise exact zeros, infinities and rounding on both conversion directions.
    logits[:, 0] = float("-inf")
    logits[:, 1] = float("inf")
    logits[:, 2] = -0.0
    output = torch.full((2, target_vocab), 17, dtype=output_dtype, device="npu")
    expected = torch.full((2, target_vocab), -float("inf"), dtype=output_dtype)
    expected.index_copy_(1, ids_cpu, logits.cpu().to(output_dtype))
    expand_draft_logits_(output, inverse, logits, starts)
    actual_cpu = output.cpu()
    torch.testing.assert_close(actual_cpu, expected, rtol=0, atol=0)
    bits = torch.int32 if output_dtype == torch.float32 else torch.int16
    assert torch.equal(actual_cpu.view(bits), expected.view(bits))
    # Also compare the existing general path for the same cross-dtype inputs.
    expand_draft_logits_(output, inverse, logits)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [32, 64])
def test_window_large_batch_matches_general_path(rows):
    torch.manual_seed(23)
    ids = torch.randperm(248320)[:64000].sort().values.to("npu")
    inverse = make_target_to_draft_map(ids, 248320)
    starts = make_draft_vocab_block_starts(ids, 248320)
    logits = torch.randn(rows, 7, 64000, dtype=torch.bfloat16, device="npu")[:, 3]
    expected = torch.full((rows, 248320), -float("inf"), dtype=logits.dtype, device="npu")
    expected.index_copy_(1, ids, logits)
    output = torch.empty_like(expected)
    expand_draft_logits_(output, inverse, logits, starts)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
