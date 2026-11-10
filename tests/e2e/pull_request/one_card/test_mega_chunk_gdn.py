# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend import envs
from vllm_ascend.ops.mega_chunk_gdn import mega_chunk_gdn_prefill, supports_mega_chunk_gdn


@pytest.mark.skipif(not envs.VLLM_ASCEND_ENABLE_MEGA_CHUNK_GDN, reason="requires the optional xLLM operator package")
def test_mega_chunk_gdn_ragged_prefill_with_initial_state():
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    # Both rows have a partial chunk. The second row resumes a cached prefix.
    starts = (0, 65, 195)
    tokens, key_heads, value_heads, dim = starts[-1], 2, 4, 128
    rng = torch.Generator().manual_seed(42)
    q = torch.randn(1, tokens, key_heads, dim, generator=rng).bfloat16()
    k = torch.randn(1, tokens, key_heads, dim, generator=rng).bfloat16()
    v = (torch.randn(1, tokens, value_heads, dim, generator=rng) * 0.25).bfloat16()
    g = -torch.rand(1, tokens, value_heads, generator=rng) * 0.1
    beta = torch.rand(1, tokens, value_heads, generator=rng).bfloat16()
    initial_state = torch.randn(2, value_heads, dim, dim, generator=rng) * 0.01
    initial_state[0].zero_()
    cu_seqlens = torch.tensor(starts, device=device, dtype=torch.int32)

    q_npu, k_npu, v_npu, g_npu, beta_npu, state_npu = (x.to(device) for x in (q, k, v, g, beta, initial_state))
    assert supports_mega_chunk_gdn(q_npu, v_npu, starts)
    assert not supports_mega_chunk_gdn(q_npu, v_npu, (0, 0, tokens))
    actual, final = mega_chunk_gdn_prefill(q_npu, k_npu, v_npu, g_npu, beta_npu, state_npu, cu_seqlens, starts)

    # Independent token-by-token FP32 reference with the same BF16 boundaries.
    def normalize(x):
        x = x.float()
        return (x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).bfloat16().float()

    q = normalize(q).repeat_interleave(value_heads // key_heads, dim=2)
    k = normalize(k).repeat_interleave(value_heads // key_heads, dim=2)
    expected = torch.empty_like(v, dtype=torch.float32)
    expected_final = initial_state.bfloat16().float()
    for row, (start, end) in enumerate(zip(starts, starts[1:])):
        state = expected_final[row]
        for token in range(start, end):
            key = k[0, token, :, None, :]
            state.mul_(g[0, token, :, None, None].exp())
            delta = (v[0, token].float() - (state * key).sum(-1)) * beta[0, token, :, None].float()
            state.add_(delta.unsqueeze(-1) * key)
            expected[0, token] = (state * q[0, token, :, None, :]).sum(-1) * dim**-0.5

    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.001, rtol=0.025)
    torch.testing.assert_close(final.float().cpu(), expected_final, atol=0.003, rtol=0.025)
