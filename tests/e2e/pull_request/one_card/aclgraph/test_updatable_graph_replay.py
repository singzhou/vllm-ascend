# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

import vllm_ascend.attention.attention_v1 as attn_module
from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState, AttentionType
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


def make_replay_case(num_layers, factor):
    """Use the target manager with real FIA update tasks and shared KV pages."""
    blocks, block_size, head_size, num_heads = 6, 128, 256, 6
    page = block_size * head_size
    raw = torch.randn(blocks * factor * page, dtype=torch.bfloat16, device="npu")
    key = raw.as_strided((blocks, block_size, 1, head_size), (factor * page, head_size, head_size, 1))
    value = key if factor == 1 else raw.as_strided(key.shape, key.stride(), storage_offset=page)
    query = torch.randn(8, num_heads, head_size, dtype=key.dtype, device="npu")
    outputs = [torch.empty_like(query) for _ in range(num_layers)]
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.SpecDecoding,
        actual_seq_lengths_q=[4, 8],
        seq_lens_list=[137, 191],
        block_tables=torch.tensor([[0, 3], [5, 1]], dtype=torch.int32, device="npu"),
        num_actual_tokens=8,
        causal=False,
        attn_mask=None,
    )
    impls = []
    for index in range(num_layers):
        impl = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
        impl.num_heads, impl.num_kv_heads, impl.head_size = num_heads, 1, head_size
        impl.scale = head_size**-0.5
        impl.attn_type = AttentionType.DECODER
        impl.use_bnsd_kv_cache = False
        impl.key_cache, impl.value_cache = key, value
        impl.sliding_window = impl.sinks = None
        impl.enable_c8_quant = impl._use_max_workspace_for_fia_graph = False
        impl._layer_name = f"layer{index}"
        impls.append(impl)
        impl.full_graph_fia(query, query, query, metadata, outputs[index])
    torch.npu.synchronize()
    graph = UpdatableGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream()):
        for impl, output in zip(impls, outputs):
            impl.full_graph_fia(query, query, query, metadata, output)
    torch.npu.synchronize()
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=8, num_reqs=2)
    manager = ModelAclGraphManager.__new__(ModelAclGraphManager)
    manager.device = query.device
    manager.update_stream = torch.npu.Stream()
    manager._update_inputs_ready = torch.npu.Event()
    manager.graphs = {desc: graph}
    manager.is_last_pp_rank = True
    manager.hidden_states = outputs[-1]
    manager.use_aux_hidden_state_outputs = False
    return manager, desc, metadata, query, key, value, outputs


@pytest.mark.parametrize("num_layers", [1, 16])
@pytest.mark.parametrize("factor", [1, 2])
def test_target_replay_dynamic_parameters_and_reused_event(num_layers, factor, monkeypatch):
    monkeypatch.setattr(attn_module, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False))
    torch.manual_seed(42)
    manager, desc, metadata, query, key, value, outputs = make_replay_case(num_layers, factor)
    replay_streams = [torch.npu.Stream(), torch.npu.Stream()]
    producer_stream = torch.npu.Stream()
    producer_stream.wait_stream(torch.npu.current_stream())
    snapshots = []
    references = []
    # No host synchronize between replays: exercise event re-recording with
    # work outstanding, changing producer streams, tables and host lengths.
    for step in range(24):
        stream = replay_streams[step % 2]
        if step:
            producer_stream.wait_stream(replay_streams[(step - 1) % 2])
        pages = [[0, 3], [5, 1]] if step % 2 else [[2, 4], [1, 0]]
        metadata.actual_seq_lengths_q = [3 if step % 2 else 5, 8]
        metadata.seq_lens_list = [137 + step, 191 - step]
        with torch.npu.stream(producer_stream):
            query.add_(0.01)
            key[0].add_(0.03125)
            metadata.block_tables = torch.tensor(pages, dtype=torch.int32, device="npu")
        with torch.npu.stream(stream):
            stream.wait_stream(producer_stream)
            result = manager._updatable_graph_replay(
                desc, {impl.provider.layer_name: metadata for impl in manager.graphs[desc].tasks}
            )
            assert result.shape == outputs[-1].shape
            snapshots.append([output.clone() for output in outputs])
            reference, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key.contiguous().flatten(2),
                value=value.contiguous().flatten(2),
                block_table=metadata.block_tables,
                num_heads=6,
                num_key_value_heads=1,
                input_layout="TND",
                block_size=128,
                actual_seq_lengths=metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=metadata.seq_lens_list,
                scale=256**-0.5,
                sparse_mode=0,
                pre_tokens=2147483647,
                next_tokens=2147483647,
            )
            references.append(reference)
    torch.npu.synchronize()
    for snapshot, reference in zip(snapshots, references):
        for output in snapshot:
            torch.testing.assert_close(output, reference, rtol=0.01, atol=0.01)
