#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm_ascend.attention.fa4_v1 as fa4_module
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
)
from vllm_ascend.attention.fa4_v1 import (
    AscendFA4Backend,
    AscendFA4Impl,
    AscendFA4MetadataBuilder,
    AscendFA4TargetMetadataBuilder,
    AscendFA4WindowMetadataBuilder,
)


@pytest.mark.parametrize("head_size", [128, 256])
def test_paged_forward_and_metadata_sharing(head_size):
    impl = AscendFA4Impl.__new__(AscendFA4Impl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 4, 2, head_size
    impl.scale = 0.17
    impl.use_bnsd_kv_cache = False
    impl.sliding_window = None
    impl.key_cache = impl.value_cache = None
    kv_cache = torch.randn(2, 16, 16, 2, head_size)
    query = torch.randn(10, 8, head_size)[:, :4]
    output = torch.full_like(query, -7)
    # A sliced table exercises packed row stride; extra rows are not requests.
    metadata = SimpleNamespace(
        causal=False,
        num_actual_tokens=8,
        seq_lens=torch.tensor([19, 0]),
        block_tables=torch.zeros(3, 8, dtype=torch.int32)[:, :4],
    )
    fa4 = MagicMock()
    fa4.flash_attn_varlen_func.return_value = torch.ones(8, 4, head_size)
    context = SimpleNamespace()
    with (
        patch.object(fa4_module, "_load_fa4", return_value=fa4),
        patch.object(fa4_module, "get_forward_context", return_value=context),
    ):
        for _ in range(2):
            assert impl.forward_fused_infer_attention(query, None, None, metadata, output, kv_cache) is output
        fa4.get_scheduler_metadata.assert_called_once()
        args = fa4.get_scheduler_metadata.call_args.kwargs
        assert args["max_seqlen_k"] == 64  # table capacity, NOT max(seq_lens)
        assert args["max_seqlen_q"] == 4
        assert args["headdim"] == head_size
        assert args["nheads"] == 4
        assert args["nheads_kv"] == 2
        assert args["window_size_left"] == args["window_size_right"] == -1
        legacy_keys = {
            "batch_size",
            "num_heads_q",
            "num_heads_kv",
            "cache_seqlens",
            "qkv_dtype",
            "page_size",
            "softmax_scale",
            "window_size",
        }
        assert not (legacy_keys & args.keys())
        assert args["seqused_k"].dtype == torch.int32
        assert args["seqused_k"].tolist() == [19, 1]
        assert args["cu_seqlens_q"].tolist() == [0, 4, 8]
        call = fa4.flash_attn_varlen_func.call_args
        assert call.args[0].is_contiguous()
        assert call.args[1].shape == (16, 16, 2, head_size)
        assert call.args[1].data_ptr() == kv_cache[0].data_ptr()
        assert call.kwargs["page_table"].shape == (2, 4)
        assert call.kwargs["page_table"].is_contiguous()
        assert call.kwargs["scheduler_metadata"] is fa4.get_scheduler_metadata.return_value
        assert call.kwargs["seqused_k"] is args["seqused_k"]
        assert torch.all(output[:8] == 1)
        assert torch.all(output[8:] == -7)
        # Different scale and table capacity must never reuse old tiling.
        impl.scale = 0.25
        impl.forward_fused_infer_attention(query, None, None, metadata, output)
        metadata.block_tables = torch.zeros(2, 8, dtype=torch.int32)
        impl.forward_fused_infer_attention(query, None, None, metadata, output)
        assert fa4.get_scheduler_metadata.call_count == 3
        # New forward: same pointer but changed device lengths needs fresh metadata.
        context = SimpleNamespace()
        metadata.seq_lens[0] = 7
        with patch.object(fa4_module, "get_forward_context", return_value=context):
            impl.forward_fused_infer_attention(query, None, None, metadata, output)
        assert fa4.get_scheduler_metadata.call_args.kwargs["seqused_k"].tolist() == [7, 1]


@pytest.mark.parametrize("num_tokens,lens", [(0, []), (0, [1]), (7, [2, 3])])
def test_rejects_nonuniform_or_empty_queries(num_tokens, lens):
    with pytest.raises(RuntimeError, match="uniform query batch"):
        fa4_module._build_fa4_seq_tensors(num_tokens, torch.tensor(lens))


def test_causal_forward_uses_base_backend():
    impl = AscendFA4Impl.__new__(AscendFA4Impl)
    with patch.object(AscendAttentionBackendImpl, "forward_fused_infer_attention", return_value="ordinary") as base:
        assert impl.forward_fused_infer_attention("q", "k", "v", SimpleNamespace(causal=True), "out") == "ordinary"
        base.assert_called_once()


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("builder_cls", [AscendFA4MetadataBuilder, AscendFA4WindowMetadataBuilder])
def test_builder_lengths_and_causal_fallback(causal, builder_cls):
    builder = builder_cls.__new__(builder_cls)
    common = SimpleNamespace(causal=causal, query_start_loc=torch.tensor([0, 4, 8]))
    lengths = torch.tensor([19, 23])
    table = torch.zeros(2, 4, dtype=torch.int32)
    with patch.object(AscendAttentionMetadataBuilder, "_build_fia_seq_inputs", return_value="ordinary") as base:
        result = builder._build_fia_seq_inputs(common, 2, common.query_start_loc, lengths, table)
        if causal and builder_cls is AscendFA4MetadataBuilder:
            assert result == "ordinary"
            base.assert_called_once()
        else:
            assert result[1:3] == (None, None)
            assert result[3] is lengths
            assert result[4] is table
            base.assert_not_called()


def test_builder_requires_parallel_drafting_and_loads_fa4():
    with (
        patch.object(AscendAttentionMetadataBuilder, "__init__", return_value=None),
        patch.object(fa4_module, "_load_fa4") as load,
    ):
        config = SimpleNamespace(speculative_config=None)
        with pytest.raises(RuntimeError, match="without parallel drafting"):
            AscendFA4MetadataBuilder(None, ["draft"], config, "cpu")
        load.assert_not_called()
        config.speculative_config = SimpleNamespace(parallel_drafting=True)
        AscendFA4MetadataBuilder(None, ["draft"], config, "cpu")
        load.assert_called_once()


def test_target_builder_uses_fa4_causal_mask():
    builder = AscendFA4TargetMetadataBuilder.__new__(AscendFA4TargetMetadataBuilder)
    builder.pcp_enabled = False
    builder.kv_cache_spec = None
    builder.model_config = SimpleNamespace(runner_type="generate")
    builder.attn_mask_builder = MagicMock()
    offsets = torch.tensor([0, 2], dtype=torch.int32)
    common = SimpleNamespace(
        causal=True,
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=offsets,
        query_start_loc_cpu=offsets,
        seq_lens=torch.tensor([129], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0, 1]], dtype=torch.int32),
        slot_mapping=torch.arange(2),
        attn_state=None,
    )

    with patch.object(builder, "_split_decodes_and_prefills", return_value=(1, 0, 2, 0)):
        metadata = builder.build(0, common)

    builder.attn_mask_builder.get_attention_mask.assert_not_called()
    assert metadata.attn_mask is None
    assert metadata.seq_lens is common.seq_lens


def test_virtual_page_table_is_shared_within_one_forward_only():
    first_layer = torch.empty(4, 2, 16, 2, 256)
    second_layer = torch.empty_like(first_layer)
    different_stride = torch.empty(4, 3, 16, 2, 256)
    block_table = torch.tensor([[3, 1]], dtype=torch.int32)
    context = SimpleNamespace()

    first_key, _, first_table = fa4_module._fa4_cache_views(
        first_layer[:, 0], first_layer[:, 1], block_table, forward_context=context
    )
    second_key, _, second_table = fa4_module._fa4_cache_views(
        second_layer[:, 0], second_layer[:, 1], block_table[:], forward_context=context
    )
    _, other_value, other_table = fa4_module._fa4_cache_views(
        different_stride[:, 0], different_stride[:, 1], block_table, forward_context=context
    )
    assert first_key.data_ptr() == first_layer.data_ptr()
    assert second_key.data_ptr() == second_layer.data_ptr()
    assert other_value.data_ptr() == different_stride[:, 1].data_ptr()
    assert first_table is second_table
    torch.testing.assert_close(first_table, block_table * 2)
    torch.testing.assert_close(other_table, block_table * 3)

    block_table[0, 0] = 0
    _, _, next_table = fa4_module._fa4_cache_views(
        first_layer[:, 0], first_layer[:, 1], block_table, forward_context=SimpleNamespace()
    )
    assert next_table is not first_table
    torch.testing.assert_close(next_table, block_table * 2)


@pytest.mark.parametrize(
    "enabled,non_causal,window,sink,expected",
    [
        (False, True, False, False, False),
        (True, True, False, False, True),
        (True, False, False, False, False),
        (True, True, True, False, False),
        (True, True, False, True, False),
    ],
)
def test_selection(enabled, non_causal, window, sink, expected):
    with patch.object(fa4_module, "_FA4_ENABLED", enabled):
        assert (
            fa4_module.fa4_selected(
                SimpleNamespace(
                    use_non_causal=non_causal,
                    has_sliding_window=window,
                    has_sink=sink,
                )
            )
            is expected
        )


def test_loader_errors():
    with (
        patch.object(fa4_module.importlib, "import_module", side_effect=ImportError),
        pytest.raises(RuntimeError, match="requires flash-attention-npu"),
    ):
        fa4_module._load_fa4()
    with (
        patch.object(fa4_module.importlib, "import_module", return_value=SimpleNamespace()),
        pytest.raises(RuntimeError, match="must provide"),
    ):
        fa4_module._load_fa4()


def test_backend_wiring_preserves_shared_cache_layout():
    assert AscendFA4Backend.get_impl_cls() is AscendFA4Impl
    assert AscendFA4Backend.get_builder_cls() is AscendFA4MetadataBuilder
    assert AscendFA4Backend.get_name() == AscendAttentionBackend.get_name()
    assert AscendFA4Backend.get_kv_cache_shape(2, 16, 4, 256) == AscendAttentionBackend.get_kv_cache_shape(
        2, 16, 4, 256
    )


@pytest.mark.parametrize("failure", ["missing_cache", "layout", "table"])
def test_invalid_cache_inputs(failure):
    impl = AscendFA4Impl.__new__(AscendFA4Impl)
    impl.key_cache = impl.value_cache = None if failure == "missing_cache" else torch.empty(2, 16, 2, 256)
    impl.use_bnsd_kv_cache = failure == "layout"
    impl.sliding_window = None
    metadata = SimpleNamespace(
        causal=False, num_actual_tokens=8, seq_lens=torch.ones(2), block_tables=torch.zeros(1, 4)
    )
    with patch.object(fa4_module, "_load_fa4"), pytest.raises(RuntimeError):
        impl.forward_fused_infer_attention(None, None, None, metadata, None)
