# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import patch

import pytest
import torch
import torch_npu

from vllm_ascend.worker.v2.sample.apply_top_k_top_p import apply_top_k_top_p_npu


@pytest.mark.parametrize("use_top_k,use_top_p", [(True, False), (False, True), (True, True)])
def test_apply_top_k_top_p_dispatches_to_cann(use_top_k, use_top_p):
    logits = torch.randn(2, 4)
    k = torch.tensor([2, 4], dtype=torch.int32) if use_top_k else None
    p = torch.tensor([0.9, 1.0]) if use_top_p else None
    filtered_logits = torch.zeros_like(logits)

    with patch.object(torch_npu, "npu_top_k_top_p", create=True, return_value=filtered_logits) as op:
        assert apply_top_k_top_p_npu(logits, k, p) is filtered_logits

    op.assert_called_once_with(logits, k=k, p=p)


def test_apply_top_k_top_p_skips_unfiltered_logits():
    logits = torch.randn(2, 4)

    with patch.object(torch_npu, "npu_top_k_top_p", create=True) as op:
        assert apply_top_k_top_p_npu(logits, None, None) is logits

    op.assert_not_called()


def test_apply_top_k_top_p_propagates_cann_errors():
    logits = torch.randn(2, 4)
    k = torch.tensor([2, 4], dtype=torch.int32)

    with (
        patch.object(torch_npu, "npu_top_k_top_p", create=True, side_effect=RuntimeError("CANN operator failed")),
        pytest.raises(RuntimeError, match="CANN operator failed"),
    ):
        apply_top_k_top_p_npu(logits, k, None)
