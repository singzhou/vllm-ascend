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
"""Run on Ascend910 with the flash-attention-npu FA4 wheel installed."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend.attention.fa4_v1 import AscendFA4Impl, AscendFA4TargetImpl


@pytest.mark.parametrize("head_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fa4_draft_eager_and_graph_replay(head_size, dtype):
    if "Ascend910" not in torch.npu.get_device_name():
        pytest.skip("FA4 AICPU metadata requires Ascend910")
    batch, q_len, heads, kv_heads, page_size, pages_per_req = 4, 4, 8, 2, 128, 16
    torch.manual_seed(0)
    query = torch.randn(batch * q_len, heads, head_size, device="npu", dtype=dtype)
    key = torch.randn(batch * pages_per_req, page_size, kv_heads, head_size, device="npu", dtype=dtype)
    value = torch.randn_like(key)
    table = torch.arange(batch * pages_per_req, device="npu", dtype=torch.int32).view(batch, pages_per_req)
    seq_lens = torch.empty(batch, device="npu", dtype=torch.int32)
    metadata = SimpleNamespace(causal=False, num_actual_tokens=batch * q_len, seq_lens=seq_lens, block_tables=table)
    impl = AscendFA4Impl.__new__(AscendFA4Impl)
    impl.key_cache, impl.value_cache = key, value
    impl.num_heads, impl.num_kv_heads, impl.head_size = heads, kv_heads, head_size
    impl.scale = head_size**-0.5
    impl.use_bnsd_kv_cache = False

    def run():
        output = torch.empty_like(query)
        # One fresh context per forward, as in the model runner. Metadata must
        # be captured INSIDE the graph and re-executed after seq_lens changes.
        with patch("vllm_ascend.attention.fa4_v1.get_forward_context", return_value=SimpleNamespace()):
            return impl.forward_fused_infer_attention(query, None, None, metadata, output)

    q_cpu, k_cpu, v_cpu = query.cpu().float(), key.cpu().float(), value.cpu().float()
    length_sets = ([900, 512, 700, 0], [2048, 1500, 64, 777])
    eager = []
    for lengths in length_sets:
        seq_lens.copy_(torch.tensor(lengths, device="npu", dtype=torch.int32))
        actual = run().clone()
        refs = []
        for i, length in enumerate(lengths):
            length = max(length, 1)
            start = i * pages_per_req
            k = k_cpu[start : start + pages_per_req].flatten(0, 1)[:length].repeat_interleave(heads // kv_heads, dim=1)
            v = v_cpu[start : start + pages_per_req].flatten(0, 1)[:length].repeat_interleave(heads // kv_heads, dim=1)
            q = q_cpu[i * q_len : (i + 1) * q_len]
            refs.append(
                F.scaled_dot_product_attention(
                    q.transpose(0, 1),
                    k.transpose(0, 1),
                    v.transpose(0, 1),
                    scale=impl.scale,
                ).transpose(0, 1)
            )
        torch.testing.assert_close(actual.cpu().float(), torch.cat(refs), atol=0.02, rtol=0.02)
        eager.append(actual)
    assert not torch.allclose(eager[0].float(), eager[1].float(), atol=1e-3)

    seq_lens.copy_(torch.tensor(length_sets[0], device="npu", dtype=torch.int32))
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = run()
    for index in (1, 0):
        seq_lens.copy_(torch.tensor(length_sets[index], device="npu", dtype=torch.int32))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(graph_output, eager[index], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "max_query_len,pages_per_req,stride_factor",
    [(4, 2, 1), (2051, 20, 3)],
    ids=["short-contiguous", "long-strided"],
)
@torch.inference_mode()
def test_fa4_causal_ragged_replay_after_rejection(max_query_len, pages_per_req, stride_factor):
    if "Ascend910" not in torch.npu.get_device_name():
        pytest.skip("FA4 AICPU metadata requires Ascend910")
    torch.manual_seed(0)
    heads, kv_heads, head_size, page_size = 4, 2, 256, 128
    query = torch.randn(2 * max_query_len, heads, head_size, device="npu", dtype=torch.bfloat16)
    num_pages = 2 * pages_per_req
    if stride_factor == 1:
        key = torch.randn(num_pages, page_size, kv_heads, head_size, device=query.device, dtype=query.dtype)
        value = torch.randn_like(key)
    else:
        # Match a hybrid cache that stores K, V and unused padding in each page.
        physical_pages = torch.full(
            (num_pages, stride_factor, page_size, kv_heads, head_size),
            float("nan"),
            device=query.device,
            dtype=query.dtype,
        )
        key, value = physical_pages[:, 0], physical_pages[:, 1]
        key.copy_(torch.randn_like(key))
        value.copy_(torch.randn_like(value))
    offsets = torch.empty(3, device="npu", dtype=torch.int32)
    lengths = torch.empty(2, device="npu", dtype=torch.int32)
    table = torch.empty(2, pages_per_req, device="npu", dtype=torch.int32)
    metadata = SimpleNamespace(
        causal=True,
        num_actual_tokens=query.shape[0],
        seq_lens=lengths,
        block_tables=table,
        query_start_loc=offsets,
        max_query_len=max_query_len,
    )
    impl = AscendFA4TargetImpl.__new__(AscendFA4TargetImpl)
    impl.key_cache, impl.value_cache = key, value
    impl.num_heads, impl.num_kv_heads, impl.head_size = heads, kv_heads, head_size
    impl.scale = head_size**-0.5
    impl.use_bnsd_kv_cache = False

    def set_batch(query_offsets, kv_lengths, page_ids):
        for buffer, data in ((offsets, query_offsets), (lengths, kv_lengths), (table, page_ids)):
            buffer.copy_(torch.tensor(data, device=query.device, dtype=torch.int32))

    def run(capturing=False):
        with (
            patch("vllm_ascend.attention.fa4_v1.get_forward_context", return_value=SimpleNamespace()),
            patch("vllm_ascend.attention.fa4_v1._EXTRA_CTX", SimpleNamespace(capturing=capturing)),
        ):
            return impl.forward_impl(query, None, None, None, metadata, torch.empty_like(query))

    first_pages = list(range(pages_per_req))
    second_pages = list(range(pages_per_req, num_pages))
    capacity = pages_per_req * page_size
    cases = (
        ([0, max_query_len, 2 * max_query_len], [capacity - 116, capacity - 1], [first_pages, second_pages]),
        ([0, max_query_len - 1, max_query_len - 1], [capacity - 129, 0], [first_pages[::-1], [0] * pages_per_req]),
        ([0, 1, max_query_len + 1], [capacity - 127, capacity - 166], [second_pages[::-1], first_pages[::-1]]),
    )
    q_cpu, k_cpu, v_cpu = query.cpu().float(), key.cpu().float(), value.cpu().float()

    def expected_output(query_offsets, kv_lengths, page_ids):
        expected = torch.zeros_like(q_cpu)
        for row, (start, end) in enumerate(zip(query_offsets[:-1], query_offsets[1:])):
            if start == end:
                continue
            length = kv_lengths[row]
            keys = k_cpu[page_ids[row]].flatten(0, 1)[:length].repeat_interleave(heads // kv_heads, dim=1)
            values = v_cpu[page_ids[row]].flatten(0, 1)[:length].repeat_interleave(heads // kv_heads, dim=1)
            q = q_cpu[start:end].transpose(0, 1)
            mask = torch.ones(end - start, length, dtype=torch.bool).tril(length - (end - start))
            expected[start:end] = F.scaled_dot_product_attention(
                q, keys.transpose(0, 1), values.transpose(0, 1), attn_mask=mask, scale=impl.scale
            ).transpose(0, 1)
        return expected

    set_batch(*cases[0])
    torch.testing.assert_close(run().cpu().float(), expected_output(*cases[0]), atol=0.02, rtol=0.02)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = run(capturing=True)

    for case in (*cases[1:], cases[0]):
        set_batch(*case)
        graph.replay()
        torch.testing.assert_close(graph_output.cpu().float(), expected_output(*case), atol=0.02, rtol=0.02)
