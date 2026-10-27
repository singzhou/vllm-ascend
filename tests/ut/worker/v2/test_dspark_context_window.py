# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""CPU tests for the DSpark draft context window.

The golden table below is the contract with the SpecForge training side: both
must place the window start on exactly the same token.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.dspark.context_window import (
    DraftContextWindow,
    DraftContextWindowConfig,
    read_context_window_config,
    window_start,
)

# (context_window, align, anchor, expected start)
GOLDEN_WINDOW_STARTS = [
    (4096, 128, 0, 0),
    (4096, 128, 100, 0),
    (4096, 128, 4096, 0),
    (4096, 128, 4097, 0),
    (4096, 128, 4223, 0),
    (4096, 128, 4224, 128),
    (4096, 128, 5000, 896),
    (4096, 128, 8192, 4096),
    (4096, 128, 10000, 5888),
    (128, 128, 128, 0),
    (128, 128, 255, 0),
    (128, 128, 256, 128),
    (128, 128, 300, 128),
    (4096, 1, 5000, 904),
]


@pytest.mark.parametrize("context_window, align, anchor, expected", GOLDEN_WINDOW_STARTS)
def test_window_start_golden_table(context_window, align, anchor, expected):
    anchors = torch.tensor([anchor], dtype=torch.int32)
    starts = window_start(anchors, context_window, align)
    assert starts.tolist() == [expected]
    assert starts.dtype == torch.int32


def test_window_start_matches_the_scalar_formula():
    anchors = torch.arange(0, 9000, 37, dtype=torch.int32)
    starts = window_start(anchors, 4096, 128)
    expected = [max(int(a) - 4096, 0) // 128 * 128 for a in anchors]
    assert starts.tolist() == expected


def test_window_start_writes_into_the_out_buffer():
    anchors = torch.tensor([5000, 100], dtype=torch.int32)
    out = torch.zeros(2, dtype=torch.int32)
    starts = window_start(anchors, 4096, 128, out=out)
    assert starts is out
    assert out.tolist() == [896, 0]
    # The anchors must survive: apply() reuses them across groups.
    assert anchors.tolist() == [5000, 100]


class TestReadContextWindowConfig:
    def test_absent_config_disables_the_feature(self):
        assert read_context_window_config(SimpleNamespace()) is None
        assert read_context_window_config(SimpleNamespace(dflash_config=None)) is None
        assert read_context_window_config(SimpleNamespace(dflash_config={"causal": True})) is None

    def test_mapping_and_object_sections(self):
        expected = DraftContextWindowConfig(4096, 128)
        mapping = SimpleNamespace(dflash_config={"context_window": 4096, "context_window_align": 128})
        nested = SimpleNamespace(dflash_config=SimpleNamespace(context_window=4096, context_window_align=128))
        assert read_context_window_config(mapping) == expected
        assert read_context_window_config(nested) == expected

    @pytest.mark.parametrize(
        "section",
        [
            {"context_window": 4096},
            {"context_window_align": 128},
            {"context_window": 0, "context_window_align": 128},
            {"context_window": 4096, "context_window_align": 0},
            {"context_window": True, "context_window_align": 128},
        ],
    )
    def test_incomplete_or_invalid_config_is_rejected(self, section):
        with pytest.raises(ValueError):
            read_context_window_config(SimpleNamespace(dflash_config=section))


def make_window(
    *,
    context_window=256,
    align=128,
    kernel_block_size=128,
    block_length=4,
    max_num_reqs=4,
    num_cols=16,
    num_groups=1,
    draft_group_ids=(0,),
):
    full_block_tables = [
        torch.arange(max_num_reqs * num_cols, dtype=torch.int32).reshape(max_num_reqs, num_cols)
        for _ in range(num_groups)
    ]
    window = DraftContextWindow(
        DraftContextWindowConfig(context_window, align),
        group_ids=draft_group_ids,
        kernel_block_sizes=[kernel_block_size] * num_groups,
        full_block_tables=full_block_tables,
        block_length=block_length,
        device=torch.device("cpu"),
    )
    return window, full_block_tables


class TestDraftContextWindowInit:
    def test_align_must_be_a_multiple_of_the_kernel_block_size(self):
        with pytest.raises(ValueError, match="kernel block size"):
            make_window(align=100, kernel_block_size=128)

    def test_clone_covers_the_widest_possible_window(self):
        window, _ = make_window(context_window=256, align=128, kernel_block_size=128, block_length=4)
        # W + align + B = 388 tokens -> 4 blocks of 128, plus one spare column.
        assert window.max_windowed_seq_len == 388
        assert window.block_table_clones[0].shape == (4, 5)

    def test_clone_never_exceeds_the_full_table(self):
        window, _ = make_window(context_window=4096, align=128, num_cols=4)
        assert window.block_table_clones[0].shape == (4, 4)

    def test_only_the_draft_groups_are_replaced(self):
        window, full = make_window(num_groups=3, draft_group_ids=(1,))
        tables = window.windowed_input_block_tables
        assert tables[0] is full[0]
        assert tables[1] is window.block_table_clones[1]
        assert tables[2] is full[2]


class TestDraftContextWindowApply:
    def test_lengths_and_block_tables_are_rebased_on_the_window(self):
        window, full = make_window(context_window=256, align=128, kernel_block_size=128, block_length=4)
        # anchor = seq_len - B: 1000-4=996 -> start 640 -> column 5
        #          500-4=496   -> start 128 -> column 1
        #          100-4=96    -> start 0   -> column 0
        seq_lens = torch.tensor([1000, 500, 100, 0], dtype=torch.int32)
        window.apply(seq_lens)

        assert window.windowed_seq_lens.tolist() == [1000 - 640, 500 - 128, 100, 0]
        clone = window.block_table_clones[0]
        torch.testing.assert_close(clone[0], full[0][0, 5:10])
        torch.testing.assert_close(clone[1], full[0][1, 1:6])
        torch.testing.assert_close(clone[2], full[0][2, 0:5])

    def test_padded_requests_stay_empty(self):
        window, full = make_window(context_window=256, block_length=4)
        window.apply(torch.tensor([5000, 0, 0, 0], dtype=torch.int32))
        # A zero KV length is a padded slot: anchor clamps to 0, so it reads
        # the head of its own (unused) row and reports no KV at all.
        assert window.windowed_seq_lens.tolist()[1:] == [0, 0, 0]
        torch.testing.assert_close(window.block_table_clones[0][1], full[0][1, 0:5])

    def test_columns_past_the_full_table_are_blanked(self):
        # A start block near the end of the table: the clone's last columns
        # have no source column and must read the null block instead of
        # wrapping into the next request's row.
        window, full = make_window(context_window=256, align=128, kernel_block_size=128, num_cols=8, block_length=4)
        seq_lens = torch.tensor([900, 0, 0, 0], dtype=torch.int32)
        window.apply(seq_lens)
        # anchor 896 -> start 640 -> column 5, so only columns 5..7 exist.
        clone = window.block_table_clones[0]
        torch.testing.assert_close(clone[0][:3], full[0][0, 5:8])
        assert clone[0][3:].tolist() == [0, 0]

    def test_apply_is_idempotent_and_writes_in_place(self):
        window, _ = make_window(context_window=256, block_length=4)
        seq_lens = torch.tensor([1000, 500, 100, 0], dtype=torch.int32)
        lengths_buffer = window.windowed_seq_lens
        clone = window.block_table_clones[0]
        window.apply(seq_lens)
        first = lengths_buffer.clone(), clone.clone()
        window.apply(seq_lens)
        assert window.windowed_seq_lens is lengths_buffer
        assert window.block_table_clones[0] is clone
        torch.testing.assert_close(lengths_buffer, first[0])
        torch.testing.assert_close(clone, first[1])

    def test_a_shrinking_batch_leaves_no_stale_length(self):
        window, _ = make_window(context_window=256, block_length=4)
        window.apply(torch.tensor([1000, 1000, 1000, 1000], dtype=torch.int32))
        window.apply(torch.tensor([1000, 0, 0, 0], dtype=torch.int32))
        assert window.windowed_seq_lens.tolist()[1:] == [0, 0, 0]

    def test_host_upper_bound_is_capped_at_the_window(self):
        window, _ = make_window(context_window=256, align=128, block_length=4)
        bound = torch.tensor([10, 300, 5000], dtype=torch.int32)
        capped = window.windowed_seq_lens_cpu_upper_bound(bound)
        # Capped at W + align; the draft block's own B positions are added on
        # top of this by the upstream builder.
        assert capped.tolist() == [10, 300, 384]
        # The caller's buffer belongs to the input batch: never written to.
        assert bound.tolist() == [10, 300, 5000]


class TestSwapIn:
    def test_swap_in_restores_the_original_inputs(self):
        window, full = make_window(context_window=256, block_length=4)
        seq_lens = torch.tensor([1000, 0, 0, 0], dtype=torch.int32)
        input_buffers = SimpleNamespace(seq_lens=seq_lens)
        block_tables = SimpleNamespace(input_block_tables=full)

        with window.swap_in(input_buffers, block_tables):
            assert input_buffers.seq_lens is window.windowed_seq_lens
            assert block_tables.input_block_tables is window.windowed_input_block_tables
            assert input_buffers.seq_lens[0].item() == 1000 - 640

        assert input_buffers.seq_lens is seq_lens
        assert block_tables.input_block_tables is full

    def test_swap_in_restores_after_an_exception(self):
        window, full = make_window()
        input_buffers = SimpleNamespace(seq_lens=torch.zeros(4, dtype=torch.int32))
        block_tables = SimpleNamespace(input_block_tables=full)
        with pytest.raises(RuntimeError), window.swap_in(input_buffers, block_tables):
            raise RuntimeError("build failed")
        assert block_tables.input_block_tables is full
