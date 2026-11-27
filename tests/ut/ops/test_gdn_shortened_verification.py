# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the real GDN metadata path for shrinking verify queries.

Run with ``pytest --noconftest tests/ut/ops/test_gdn_shortened_verification.py``.
Only vLLM runtime containers and cache lookup are substituted.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ut.helpers.dcut_reference import recurrent_reference

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/gdn_attn_builder.py"
_MAX_QUERY = 8


@pytest.fixture
def metadata_runtime():
    tree = ast.parse(_SOURCE.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_has_shortened_spec_queries", "_build_actual_seq_lengths")
    ]
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendGDNAttentionMetadataBuilder"
    )
    names = {
        "_compute_shared_batch_plan",
        "_derive_group_state_indices",
        "_attach_spec_decode_metadata",
        "_try_fused_spec_metadata",
        "_can_pad_spec_decode",
        "build",
    }
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    harness = ast.ClassDef(name="_Builder", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions, harness],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "GDNAttentionMetadata": SimpleNamespace,
        "GDNSpecDecodeMetadata": SimpleNamespace,
        "GDNSpecCausalConv1dMetadata": lambda query_start_loc, cache_indices, num_accepted_tokens: SimpleNamespace(
            query_start_loc=query_start_loc, cache_indices=cache_indices, num_accepted_tokens=num_accepted_tokens
        ),
        "_GDNSharedBatchPlan": SimpleNamespace,
        "_STATE_INDEX_NON_SPEC": "non_spec",
        "_STATE_INDEX_SPEC_ONLY": "spec_only",
        "_STATE_INDEX_SPEC_MIXED": "spec_mixed",
        "mamba_get_block_table_tensor": lambda table, *args: table,
    }
    exec(compile(ast.fix_missing_locations(module), str(_SOURCE), "exec"), namespace)
    return namespace


def _build(runtime, lengths, accepted):
    builder = runtime["_Builder"]()
    builder.num_spec = _MAX_QUERY - 1
    builder.use_spec_decode = builder._USE_COMMON_KERNEL_METADATA = True
    builder.use_full_cuda_graph = False
    builder.ragged_spec_decode = False
    builder.kv_cache_spec = object()
    builder.spec_sequence_masks_cpu = torch.empty(len(lengths), dtype=torch.bool)
    builder.spec_sequence_masks = torch.empty(len(lengths), dtype=torch.bool)
    builder._copy_sequence_indices_to_device = lambda mask, count: (
        mask.nonzero().flatten(),
        (~mask).nonzero().flatten(),
    )
    builder._get_gdn_local_metadata = lambda metadata, *args: metadata
    builder._get_shared_batch_plan = lambda metadata, counts, drafts, cache: builder._compute_shared_batch_plan(
        metadata, counts, drafts
    )
    builder._attach_non_spec_prefill_metadata = lambda *args: None
    builder._attach_non_spec_decode_metadata = lambda *args: None
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="all"))
    query_start = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32)
    table = torch.arange(1, len(lengths) * _MAX_QUERY + 1, dtype=torch.int32).reshape(len(lengths), _MAX_QUERY)
    metadata = SimpleNamespace(
        num_reqs=len(lengths),
        num_actual_tokens=sum(lengths),
        query_start_loc=query_start,
        query_start_loc_cpu=query_start,
        block_table_tensor=table,
        seq_lens=torch.tensor(lengths),
    )
    draft_counts = torch.tensor([length - 1 for length in lengths], dtype=torch.int32)
    accepted = torch.tensor(accepted, dtype=torch.int32)
    result = builder.build(0, metadata, accepted, draft_counts)
    return result, table


@pytest.mark.parametrize("lengths", [[5], [1], [8, 5, 1], [8, 5, 0]])
def test_short_queries_preserve_previous_accepted_slot_and_full_table(metadata_runtime, lengths):
    result, table = _build(metadata_runtime, lengths, [8] * len(lengths))
    live = torch.tensor(lengths) > 0
    assert result.use_dcut_varlen
    torch.testing.assert_close(result.spec_state_indices_tensor, table[live])
    torch.testing.assert_close(result.num_accepted_tokens, torch.full((int(live.sum()),), 8, dtype=torch.int32))
    conv = result.spec_decode_metadata.spec_causal_conv1d
    assert conv.cache_indices.shape[1] == _MAX_QUERY
    torch.testing.assert_close(conv.num_accepted_tokens, result.num_accepted_tokens)
    assert result.spec_decode_metadata.actual_seq_lengths.tolist() == [0, *[length for length in lengths if length]]


def test_full_queries_keep_the_existing_backend(metadata_runtime):
    result, table = _build(metadata_runtime, [8, 8], [8, 3])
    assert not result.use_dcut_varlen
    torch.testing.assert_close(result.spec_state_indices_tensor, table)
    assert result.num_accepted_tokens.tolist() == [8, 3]


@pytest.mark.parametrize("length,full_graph,expected", [(5, True, True), (8, True, False), (5, False, None)])
def test_fused_metadata_retains_short_query_routing(metadata_runtime, length, full_graph, expected):
    builder = metadata_runtime["_Builder"]()
    builder.num_spec = 7
    builder.use_full_cuda_graph = full_graph
    builder.use_spec_decode = builder._USE_COMMON_KERNEL_METADATA = True
    builder.ragged_spec_decode = False
    builder.decode_cudagraph_max_bs = 16
    builder._SPEC_GRAPH_PAD_SLOT_ID = -1
    builder.kv_cache_spec = SimpleNamespace(num_speculative_blocks=7, block_size=128)
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="align"))
    builder.spec_state_indices_tensor = torch.empty((1, 8), dtype=torch.int32)
    builder.spec_query_start_loc = torch.empty(2, dtype=torch.int32)
    builder.num_accepted_tokens = torch.empty(1, dtype=torch.int32)
    builder.spec_sequence_masks = torch.empty(1, dtype=torch.bool)
    builder.spec_actual_seq_lengths = torch.empty(2, dtype=torch.int32)
    builder.spec_token_indx = torch.empty(length, dtype=torch.int32)
    builder.non_spec_token_indx = torch.empty(0, dtype=torch.int32)
    device = SimpleNamespace(type="npu")

    def device_view(values):
        return SimpleNamespace(value=torch.tensor(values, dtype=torch.int32), dtype=torch.int32, device=device)

    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=length,
        query_start_loc_cpu=torch.tensor([0, length], dtype=torch.int32),
        query_start_loc=device_view([0, length]),
        seq_lens=device_view([length]),
        block_table_tensor=device_view([list(range(1, 9))]),
    )
    calls = []

    def fused(table, seq_lens, query, accepted, outputs, live, num_tokens, block_size, padding):
        states_out, query_out, accepted_out, masks_out, lengths_out, tokens_out = outputs
        states_out.copy_(table.value)
        query_out.copy_(query.value)
        accepted_out.copy_(accepted.value)
        masks_out.fill_(True)
        lengths_out.copy_(query.value)
        tokens_out.copy_(torch.arange(num_tokens))
        calls.append(live)

    metadata_runtime["fused_gdn_spec_metadata"] = fused
    result = builder._try_fused_spec_metadata(common, device_view([8]), torch.tensor([length - 1]), 1, {})
    if expected is None:
        assert result is None and not calls
    else:
        assert calls == [1]
        assert result.use_dcut_varlen is expected
        assert result.spec_state_indices_tensor.tolist() == [list(range(1, 9))]
        assert result.num_accepted_tokens.tolist() == [8]
        assert result.spec_decode_metadata.actual_seq_lengths.tolist() == [0, length]


@pytest.mark.parametrize(
    "lengths,drafts,expected",
    [
        ([8, 8], [7, 7], False),
        ([8, 0], [7, -1], False),
        ([0], [0], False),
        ([1], [-1], False),
        ([5], [-1], False),  # A short prefill is not a speculative query.
        ([1], [0], True),
        ([5], [4], True),
        ([8, 5], [7, 4], True),
    ],
)
def test_shortening_gate_uses_real_cpu_query_lengths(metadata_runtime, lengths, drafts, expected):
    query_start = torch.tensor([0, *np.cumsum(lengths)], dtype=torch.int32)
    assert metadata_runtime["_has_shortened_spec_queries"](query_start, torch.tensor(drafts), 7) is expected
    assert not metadata_runtime["_has_shortened_spec_queries"](query_start, None, 7)


@pytest.mark.parametrize("current_length", [5, 1])
def test_dcut_reference_reads_previous_slot_eight_after_query_shrinks(metadata_runtime, current_length):
    metadata, table = _build(metadata_runtime, [current_length], [8])
    state = np.arange(9, dtype=float).reshape(9, 1, 1, 1)
    output, updated = recurrent_reference(
        np.ones((current_length, 1, 1)),
        np.zeros((current_length, 1, 1)),
        np.zeros((current_length, 1, 1)),
        state,
        np.zeros((current_length, 1)),
        np.zeros((current_length, 1)),
        metadata.spec_decode_metadata.spec_causal_conv1d.query_start_loc.numpy(),
        metadata.spec_state_indices_tensor.numpy(),
        metadata.num_accepted_tokens.numpy(),
        scale=1.0,
    )
    # beta=g=0: the exact output exposes the selected initial-state value.
    np.testing.assert_array_equal(output, np.full_like(output, 8.0))
    np.testing.assert_array_equal(updated[table[0, :current_length].numpy()], 8.0)
    np.testing.assert_array_equal(state[:, 0, 0, 0], np.arange(9))
