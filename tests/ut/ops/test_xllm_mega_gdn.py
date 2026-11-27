# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend import envs
from vllm_ascend.ops import xllm_mega_gdn


def test_opt_in_env(monkeypatch):
    flag = "VLLM_ASCEND_ENABLE_XLLM_MEGA_GDN"
    monkeypatch.delenv(flag, raising=False)
    assert not envs.VLLM_ASCEND_ENABLE_XLLM_MEGA_GDN
    monkeypatch.setenv(flag, "1")
    assert envs.VLLM_ASCEND_ENABLE_XLLM_MEGA_GDN
    monkeypatch.setenv(flag, "true")
    with pytest.raises(ValueError, match=flag):
        _ = envs.VLLM_ASCEND_ENABLE_XLLM_MEGA_GDN


def test_packed_metadata_counts_partial_chunks_per_sequence():
    offsets = (0, 63, 192, 449)
    cpu_offsets = torch.tensor(offsets, dtype=torch.int64)
    kernel_offsets = torch.tensor(offsets, dtype=torch.int32)
    plan = xllm_mega_gdn.XllmMegaGdnMetadata.build(cpu_offsets, kernel_offsets)
    assert plan.num_chunks == 6
    assert plan.num_tokens == 449
    assert plan.cu_seqlens is kernel_offsets
    assert plan.cu_seqlens.tolist() == list(offsets)


def test_empty_segments_compact_to_active_cache_rows():
    offsets = torch.tensor([0, 0, 63, 63, 192, 192], dtype=torch.int32)
    plan = xllm_mega_gdn.XllmMegaGdnMetadata.build(offsets, offsets)
    assert plan.cu_seqlens.tolist() == [0, 63, 192]
    assert plan.active_rows.tolist() == [1, 3]
    assert plan.num_chunks == 3


def test_adapter_normalizes_qk_and_preserves_state_contract(monkeypatch):
    dtype = torch.bfloat16
    q, k = [torch.arange(256, dtype=dtype).reshape(1, 2, 1, 128) for _ in range(2)]
    v = torch.ones(1, 2, 2, 128, dtype=dtype)
    gate = torch.zeros(1, 2, 2, dtype=dtype)
    beta = torch.ones(1, 2, 2, dtype=torch.float32)
    initial_state = torch.zeros(1, 2, 128, 128, dtype=dtype)
    initial_state[:, :, 0, 1] = 3
    plan = xllm_mega_gdn.XllmMegaGdnMetadata(torch.tensor([0, 2], dtype=torch.int32), 1, 2)

    def fused(q_arg, k_arg, v_arg, g_arg, beta_arg, lower, full, minus_eye, cu, state, matrices, has_state):
        torch.testing.assert_close(q_arg, q + 1)
        torch.testing.assert_close(k_arg, k + 1)
        assert v_arg is v
        assert g_arg.dtype == torch.float32 and beta_arg.dtype == dtype
        assert lower[0, 0] == 0 and full[0, 0] == 1 and minus_eye[0, 0] == -1
        assert cu is plan.cu_seqlens
        assert state.is_contiguous() and state.dtype == dtype and state[0, 0, 1, 0] == 3
        assert matrices == 2 and has_state
        outputs = [None] * 12
        outputs[0] = torch.full_like(v, 8)
        outputs[-1] = state.reshape(2, 128, 128) + 1
        return outputs

    monkeypatch.setitem(sys.modules, "custom_ops_lib", SimpleNamespace(mega_chunk_gdn=fused))
    monkeypatch.setattr(xllm_mega_gdn, "l2norm_fwd", Mock(side_effect=lambda tensor: tensor + 1))
    adapter = xllm_mega_gdn.XllmMegaGdnPrefill(q.device, dtype)
    out, state = adapter(q, k, v, gate, beta, initial_state, plan)
    torch.testing.assert_close(out, torch.full_like(v, 8 * 128**-0.5))
    torch.testing.assert_close(state, initial_state + 1)
    assert state.is_contiguous()


def test_missing_opt_in_dependency_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "custom_ops_lib", None)
    with pytest.raises(RuntimeError, match="requires the xLLM MegaChunkGdn"):
        xllm_mega_gdn.XllmMegaGdnPrefill(torch.device("cpu"), torch.bfloat16)


def test_adapter_issues_one_op_without_device_readback(monkeypatch):
    dtype = torch.bfloat16
    q = torch.ones(1, 2, 1, 128, dtype=dtype)
    g = torch.zeros(1, 2, 1)
    initial_state = torch.zeros(1, 1, 128, 128, dtype=dtype)
    plan = xllm_mega_gdn.XllmMegaGdnMetadata(torch.tensor([0, 2], dtype=torch.int32), 1, 2)
    outputs = [None] * 12
    outputs[0] = torch.full_like(q, float("nan"))
    outputs[-1] = initial_state
    fused = Mock(return_value=outputs)
    monkeypatch.setitem(sys.modules, "custom_ops_lib", SimpleNamespace(mega_chunk_gdn=fused))
    monkeypatch.setattr(xllm_mega_gdn, "l2norm_fwd", lambda tensor: tensor)

    adapter = xllm_mega_gdn.XllmMegaGdnPrefill(q.device, dtype)
    with monkeypatch.context() as no_readback:
        no_readback.setattr(torch.Tensor, "item", Mock(side_effect=AssertionError("device readback")))
        out, state = adapter(q, q, q, g, g, initial_state, plan)
    fused.assert_called_once()
    assert out is outputs[0]
    assert state.is_contiguous()
