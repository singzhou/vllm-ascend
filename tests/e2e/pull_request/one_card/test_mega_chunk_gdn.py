# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import recurrent_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend import envs
from vllm_ascend.ops.mega_chunk_gdn import mega_chunk_gdn_prefill, supports_mega_chunk_gdn
from vllm_ascend.ops.triton.mamba.state_index import scatter_ssm_states_


def _normalize(x):
    x = x.float()
    return (x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).bfloat16().float()


def _recurrent_step(state, query, key, value, g, beta):
    key = key.unsqueeze(-2)
    state.mul_(g[:, None, None].exp())
    delta = (value.float() - (state * key).sum(-1)) * beta[:, None].float()
    state.add_(delta.unsqueeze(-1) * key)
    return (state * query.unsqueeze(-2)).sum(-1) * query.shape[-1] ** -0.5


@pytest.mark.skipif(not envs.VLLM_ASCEND_ENABLE_MEGA_CHUNK_GDN, reason="requires the MegaChunkGdn Ascend910B build")
def test_mega_chunk_gdn_ragged_prefill_and_decode():
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
    heads_per_key = value_heads // key_heads
    q = _normalize(q).repeat_interleave(heads_per_key, dim=2)
    k = _normalize(k).repeat_interleave(heads_per_key, dim=2)
    expected = torch.empty_like(v, dtype=torch.float32)
    expected_final = initial_state.bfloat16().float()
    for row, (start, end) in enumerate(zip(starts, starts[1:])):
        state = expected_final[row]
        for token in range(start, end):
            expected[0, token] = _recurrent_step(
                state, q[0, token], k[0, token], v[0, token], g[0, token], beta[0, token]
            )

    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.001, rtol=0.025)
    torch.testing.assert_close(final.float().cpu(), expected_final, atol=0.003, rtol=0.025)

    # Write the prefill result into real cache slots, then use the ordinary decode operator.
    slots = torch.tensor([3, 1], dtype=torch.int32, device=device)
    cache = torch.zeros((4, value_heads, dim, dim), dtype=torch.float32, device=device)
    scatter_ssm_states_(cache, slots, final.float().contiguous())
    dq = torch.randn(2, key_heads, dim, generator=rng).bfloat16()
    dk = torch.randn(2, key_heads, dim, generator=rng).bfloat16()
    dv = (torch.randn(2, value_heads, dim, generator=rng) * 0.25).bfloat16()
    dg = -torch.rand(2, value_heads, generator=rng) * 0.1
    dbeta = torch.rand(2, value_heads, generator=rng).bfloat16()
    decoded = recurrent_gated_delta_rule(
        l2norm_fwd(dq.to(device)),
        l2norm_fwd(dk.to(device)),
        dv.to(device),
        cache,
        g=dg.to(device),
        beta=dbeta.to(device),
        scale=dim**-0.5,
        actual_seq_lengths=torch.tensor([0, 1, 1], dtype=torch.int32, device=device),
        ssm_state_indices=slots,
    )
    dq = _normalize(dq).repeat_interleave(heads_per_key, dim=1)
    dk = _normalize(dk).repeat_interleave(heads_per_key, dim=1)
    expected_state = expected_final.bfloat16().float()
    expected_decode = torch.stack(
        [_recurrent_step(expected_state[i], dq[i], dk[i], dv[i], dg[i], dbeta[i]) for i in range(2)]
    )
    torch.testing.assert_close(decoded.float().cpu(), expected_decode, atol=0.001, rtol=0.025)
    cache_cpu = cache.cpu()
    torch.testing.assert_close(cache_cpu[[3, 1]], expected_state, atol=0.003, rtol=0.025)
    assert not cache_cpu[[0, 2]].any()
