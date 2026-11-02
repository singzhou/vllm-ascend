# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
from vllm.v1.outputs import LogprobsTensors

from vllm_ascend.worker.v2.spec_decode import rejection_sampler as rejection


@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits"])
@pytest.mark.parametrize("adaptive,chunk_limit,num_topk", [(False, 2, 3), (False, 8, -1), (True, 8, 0)])
def test_mixed_decision_verification(monkeypatch, mode, adaptive, chunk_limit, num_topk):
    # A decision with 128 labels shares the batch with a draft and a binary read.
    # Request-state order is unrelated to logit row order.
    state = SimpleNamespace(max_num_token_ids=Mock(return_value=128))
    verifier = object.__new__(rejection.AscendRejectionSampler)
    verifier.sampler = SimpleNamespace(logprob_token_ids_state=state, logprobs_mode=mode)
    verifier.enable_adaptive_verification = adaptive
    cu = torch.tensor([0, 1, 3, 4])
    rows = torch.tensor([2, 0, 0, 1])
    batch = SimpleNamespace(
        num_reqs=3,
        idx_mapping=torch.tensor([2, 0, 1]),
        idx_mapping_np=np.array([2, 0, 1]),
        expanded_idx_mapping=rows,
        expanded_local_pos=torch.tensor([0, 0, 1, 0]),
        cu_num_logits=cu,
        cu_num_logits_np=np.array([0, 3, 6, 9]) if adaptive else cu.numpy(),
    )
    logits = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256) / 100

    def verify(raw, draft_logits, draft_sampled, pos, boundaries, *args):
        counts = boundaries.diff()
        sampled = torch.full((len(counts), 2), 255)
        return -raw, sampled, counts

    monkeypatch.setattr(verifier, "_verify", verify)
    flatten = MagicMock()

    def flatten_rows(flat, sampled, stride, counts, boundaries, **kwargs):
        for row, count in enumerate(counts):
            flat[boundaries[row] : boundaries[row] + count] = sampled[row, :count]

    flatten.__getitem__.return_value = flatten_rows
    monkeypatch.setattr(rejection, "_flatten_sampled_kernel", flatten)

    observed_rows = []

    def gather(
        scores,
        topk,
        sampled,
        boundaries,
        *,
        logprob_token_ids_state,
        expanded_idx_mapping,
        max_per_req_token_ids,
        logits_mode,
    ):
        assert logprob_token_ids_state is state
        assert topk == max(num_topk, 0) and max_per_req_token_ids == 128
        assert logits_mode == mode.endswith("logits")
        observed_rows.extend(expanded_idx_mapping.tolist())
        assert sampled.tolist() == [255] * len(scores)
        # Retain one diagnostic column to check the chunk assembly and score source.
        return LogprobsTensors(
            sampled[:, None], scores[:, :1], sampled, None if adaptive else boundaries, boundaries if adaptive else None
        )

    monkeypatch.setattr(rejection, "compute_topk_scores", gather)
    sampled, counts, result = verifier._verify_in_chunks(logits, batch, None, rows, rows, chunk_limit, num_topk)
    assert observed_rows == rows.tolist()
    assert counts.tolist() == [1, 2, 1] and sampled.shape == (3, 2)
    torch.testing.assert_close(result.logprobs, (-logits if mode.startswith("processed") else logits)[:, :1])
    if adaptive:
        assert result.cu_num_generated_tokens is None
        torch.testing.assert_close(result.cu_num_generated_tokens_tensor, cu)
    else:
        assert result.cu_num_generated_tokens == cu.tolist()


def test_no_custom_ids_preserves_upstream_verification(monkeypatch):
    verifier = object.__new__(rejection.AscendRejectionSampler)
    verifier.sampler = SimpleNamespace(logprob_token_ids_state=SimpleNamespace(max_num_token_ids=lambda indices: 0))
    original = Mock(return_value=object())
    monkeypatch.setattr(rejection.RejectionSampler, "_verify_in_chunks", original)
    args = (None, SimpleNamespace(idx_mapping_np=np.array([1])), None, None, None, 1, -1)
    assert verifier._verify_in_chunks(*args) is original.return_value
    original.assert_called_once_with(*args)
