# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""How AscendDSparkSpeculator wires the draft context window in.

These build no NPU metadata; they check that the windowed buffers reach the
upstream metadata build and that a draft without the config is untouched.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.spec_decode.dspark.context_window import (
    DraftContextWindow,
    DraftContextWindowConfig,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator

MAX_NUM_REQS = 4
NUM_COLS = 16
KERNEL_BLOCK_SIZE = 128


def paged_group():
    return SimpleNamespace(kv_cache_spec=MagicMock(spec=AttentionSpec))


def make_speculator(
    context_window_config=None,
    *,
    use_dcp=False,
    cp_size=1,
    num_groups=1,
    draft_group_ids=(0,),
    kv_cache_groups=None,
):
    speculator = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    speculator.device = torch.device("cpu")
    speculator.num_query_per_req = 4
    speculator.draft_kv_cache_group_ids = list(draft_group_ids)
    speculator.kv_cache_config = SimpleNamespace(
        kv_cache_groups=kv_cache_groups if kv_cache_groups is not None else [paged_group() for _ in range(num_groups)]
    )
    speculator.use_dcp = use_dcp
    speculator.draft_max_seq_len = 100000
    speculator._context_window_config = context_window_config
    speculator._context_window = None
    speculator._context_window_depth = 0
    speculator.input_buffers = SimpleNamespace(seq_lens=torch.zeros(MAX_NUM_REQS, dtype=torch.int32))
    speculator.block_tables = SimpleNamespace(
        cp_size=cp_size,
        kernel_block_sizes=[KERNEL_BLOCK_SIZE] * num_groups,
        input_block_tables=[
            torch.arange(MAX_NUM_REQS * NUM_COLS, dtype=torch.int32).reshape(MAX_NUM_REQS, NUM_COLS)
            for _ in range(num_groups)
        ],
    )
    return speculator


class TestWithoutTheConfig:
    def test_init_leaves_the_speculator_alone(self):
        speculator = make_speculator()
        speculator._init_context_window()
        assert speculator._context_window is None

    def test_scope_is_a_no_op(self):
        speculator = make_speculator()
        speculator._init_context_window()
        seq_lens = speculator.input_buffers.seq_lens
        block_tables = speculator.block_tables.input_block_tables
        with speculator.draft_attn_input_scope():
            assert speculator.input_buffers.seq_lens is seq_lens
            assert speculator.block_tables.input_block_tables is block_tables
            assert speculator.draft_max_seq_len == 100000

    def test_the_upper_bound_is_passed_through_unchanged(self):
        speculator = make_speculator()
        speculator._init_context_window()
        bound = torch.tensor([5000, 10], dtype=torch.int32)
        assert speculator._windowed_seq_lens_cpu_upper_bound(bound) is bound


class TestWithTheConfig:
    def test_init_builds_the_window(self):
        speculator = make_speculator(DraftContextWindowConfig(256, 128))
        speculator._init_context_window()
        window = speculator._context_window
        assert isinstance(window, DraftContextWindow)
        assert window.block_length == speculator.num_query_per_req

    @pytest.mark.parametrize("use_dcp, cp_size", [(True, 1), (False, 2)])
    def test_context_parallelism_is_rejected(self, use_dcp, cp_size):
        speculator = make_speculator(DraftContextWindowConfig(256, 128), use_dcp=use_dcp, cp_size=cp_size)
        with pytest.raises(ValueError, match="context parallelism"):
            speculator._init_context_window()

    def test_only_paged_groups_are_windowed(self):
        # A recurrent group's block table addresses state slots: windowing it
        # would rebase those onto nothing.
        speculator = make_speculator(
            DraftContextWindowConfig(256, 128),
            num_groups=2,
            draft_group_ids=(0, 1),
            kv_cache_groups=[paged_group(), SimpleNamespace(kv_cache_spec=object())],
        )
        speculator._init_context_window()
        assert speculator._context_window.group_ids == (0,)
        tables = speculator._context_window.windowed_input_block_tables
        assert tables[1] is speculator.block_tables.input_block_tables[1]

    def test_a_draft_without_a_paged_group_is_rejected(self):
        speculator = make_speculator(
            DraftContextWindowConfig(256, 128),
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())],
        )
        with pytest.raises(ValueError, match="no paged draft KV cache group"):
            speculator._init_context_window()

    def test_scope_swaps_the_inputs_and_caps_the_batch_maximum(self):
        speculator = make_speculator(DraftContextWindowConfig(256, 128))
        speculator._init_context_window()
        speculator.input_buffers.seq_lens.copy_(torch.tensor([1000, 0, 0, 0], dtype=torch.int32))
        original_seq_lens = speculator.input_buffers.seq_lens

        with speculator.draft_attn_input_scope():
            window = speculator._context_window
            assert speculator.input_buffers.seq_lens is window.windowed_seq_lens
            assert speculator.block_tables.input_block_tables is window.windowed_input_block_tables
            assert speculator.draft_max_seq_len == window.max_windowed_seq_len

        assert speculator.input_buffers.seq_lens is original_seq_lens

    def test_a_nested_scope_does_not_recompute_the_window(self):
        speculator = make_speculator(DraftContextWindowConfig(256, 128))
        speculator._init_context_window()
        speculator.input_buffers.seq_lens.copy_(torch.tensor([1000, 0, 0, 0], dtype=torch.int32))

        with speculator.draft_attn_input_scope():
            windowed = speculator.input_buffers.seq_lens.clone()
            # A nested build must not re-window the already windowed lengths.
            with speculator.draft_attn_input_scope():
                torch.testing.assert_close(speculator.input_buffers.seq_lens, windowed)
            torch.testing.assert_close(speculator.input_buffers.seq_lens, windowed)

    def test_build_attn_metadata_runs_inside_the_scope_with_a_capped_bound(self):
        speculator = make_speculator(DraftContextWindowConfig(256, 128))
        speculator._init_context_window()
        speculator.input_buffers.seq_lens.copy_(torch.tensor([1000, 500, 0, 0], dtype=torch.int32))
        window = speculator._context_window
        seen = {}

        def record(**kwargs):
            seen["bound"] = kwargs["seq_lens_cpu_upper_bound"].tolist()
            seen["seq_lens"] = speculator.input_buffers.seq_lens
            seen["block_tables"] = speculator.block_tables.input_block_tables
            return None

        speculator._build_windowed_attn_metadata = record
        bound = torch.tensor([1000, 500, 0, 0], dtype=torch.int32)
        speculator._build_attn_metadata(
            num_reqs=2,
            batch_desc=BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=8, num_reqs=2),
            query_start_loc_np=None,
            seq_lens_cpu_upper_bound=bound,
            step=speculator.num_query_per_req,
        )

        assert seen["seq_lens"] is window.windowed_seq_lens
        assert seen["block_tables"] is window.windowed_input_block_tables
        # W + align = 384 caps the context part; the builder adds the block on top.
        assert seen["bound"] == [384, 384, 0, 0]
        assert bound.tolist() == [1000, 500, 0, 0]
