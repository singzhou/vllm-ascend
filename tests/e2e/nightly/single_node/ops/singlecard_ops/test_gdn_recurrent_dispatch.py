# SPDX-License-Identifier: Apache-2.0
"""CPU-reference and graph gates for fixed-length FLA/legacy GDN dispatch."""

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import recurrent_gated_delta_rule

from tests.ut.helpers.dcut_reference import recurrent_reference
from vllm_ascend.ops.gdn_recurrent import legacy_recurrent_gated_delta_rule, select_gdn_recurrent_backend
from vllm_ascend.utils import enable_custom_op


@pytest.fixture(scope="module", autouse=True)
def initialize_custom_operators():
    assert enable_custom_op()
    assert hasattr(torch.ops._C_ascend, "npu_recurrent_gated_delta_rule")
    torch.npu.set_compile_mode(jit_compile=False)


@pytest.mark.parametrize("width", [2, 8, 9, 16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_fixed_recurrent_multiround_cpu_reference(width, state_dtype, strided, graph_mode):
    torch.manual_seed(41)
    rows, nk, nv, dim, padding = 2, 1, 2, 128, 3
    tokens = rows * width
    slots = tokens + 2
    table = np.arange(1, tokens + 1, dtype=np.int32)[::-1].copy().reshape(rows, width)
    state = torch.randn(slots, nv, dim, dim, dtype=state_dtype) * 0.1
    if strided:
        backing = torch.full((slots, nv * dim * dim + 32), 17, dtype=state_dtype, device="npu")
        device_state = backing[:, : nv * dim * dim].view(slots, nv, dim, dim)
    else:
        device_state = torch.empty_like(state, device="npu")
    device_state.copy_(state)
    q = torch.nn.functional.normalize(torch.randn(tokens + padding, nk, dim), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(tokens + padding, nk, dim), dim=-1).bfloat16()
    v = torch.randn(tokens + padding, nv, dim, dtype=torch.bfloat16) * 0.2
    beta = torch.rand(tokens + padding, nv, dtype=torch.bfloat16)
    g = -torch.rand(tokens + padding, nv, dtype=torch.float32) * 0.1
    device_inputs = [x.npu() for x in (q, k, v)]
    indices = torch.tensor(np.r_[table.reshape(-1), [-1] * padding], dtype=torch.int32, device="npu")
    accepted = torch.tensor([1, 1, 1], dtype=torch.int32, device="npu")
    kwargs = dict(
        beta=beta.npu(),
        g=g.npu(),
        scale=dim**-0.5,
        actual_seq_lengths=torch.tensor([0, width, width, 0], dtype=torch.int32, device="npu"),
        ssm_state_indices=indices,
        num_accepted_tokens=accepted,
    )
    backend = select_gdn_recurrent_backend(width - 1, False)
    op = legacy_recurrent_gated_delta_rule if backend == "legacy" else recurrent_gated_delta_rule

    def run():
        return op(*device_inputs, device_state, **kwargs)

    if graph_mode:
        run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run()
        device_state.copy_(state)

    for counts in ([1, 1], [width, max(1, width // 2)], [1, width]):
        accepted.copy_(torch.tensor([*counts, 1], dtype=torch.int32))
        expected, next_state = recurrent_reference(
            q[:tokens].float().numpy(),
            k[:tokens].float().numpy(),
            v[:tokens].float().numpy(),
            state.float().numpy(),
            beta[:tokens].float().numpy(),
            g[:tokens].numpy(),
            np.array([0, width, tokens]),
            table,
            np.array(counts),
            scale=dim**-0.5,
        )
        if graph_mode:
            graph.replay()
        else:
            output = run()
        torch.npu.synchronize()
        torch.testing.assert_close(
            output[:tokens].cpu().float(), torch.from_numpy(expected).float(), rtol=2e-2, atol=2e-3
        )
        torch.testing.assert_close(
            device_state.cpu().float(), torch.from_numpy(next_state).float(), rtol=2e-2, atol=2e-3
        )
        if strided:
            assert torch.all(backing[:, -32:].cpu() == 17)
        # Reference state is rounded to the cache dtype between rounds.
        state = torch.from_numpy(next_state).to(state_dtype)
