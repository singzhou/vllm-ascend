# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.gdn_recurrent import legacy_recurrent_gated_delta_rule, select_gdn_recurrent_backend


@pytest.mark.parametrize("num_spec,expected", [(0, "fla"), (1, "fla"), (7, "fla"), (8, "legacy"), (15, "legacy")])
def test_fixed_length_backend(num_spec, expected):
    assert select_gdn_recurrent_backend(num_spec, False) == expected


@pytest.mark.parametrize("num_spec", [1, 7, 8, 15])
def test_dynamic_verification_always_uses_dcut(num_spec):
    assert select_gdn_recurrent_backend(num_spec, True) == "dcut"


@pytest.mark.parametrize("num_spec", [-1, 16, 31])
@pytest.mark.parametrize("use_dcut", [False, True])
def test_unsupported_length_rejected(num_spec, use_dcut):
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        select_gdn_recurrent_backend(num_spec, use_dcut)


@pytest.mark.parametrize("strided", [False, True])
def test_legacy_passes_original_state_and_indices_without_copies(strided):
    backing = torch.zeros(20, 8)
    state = backing[:, 1:5].view(20, 1, 2, 2) if strided else torch.zeros(20, 1, 2, 2)
    qkv = torch.zeros(9, 1, 2)
    indices = torch.arange(1, 10, dtype=torch.int32)
    lengths = torch.tensor([0, 9], dtype=torch.int32)

    def kernel(query, key, value, actual_state, **kwargs):
        assert actual_state is state
        assert kwargs["ssm_state_indices"] is indices
        assert kwargs["actual_seq_lengths"] is lengths
        actual_state[1:10].fill_(3)
        return value

    with patch("torch.ops._C_ascend.npu_recurrent_gated_delta_rule", side_effect=kernel, create=True):
        result = legacy_recurrent_gated_delta_rule(
            qkv, qkv, qkv, state, ssm_state_indices=indices, actual_seq_lengths=lengths
        )
    assert result is qkv
    assert torch.all(state[1:10] == 3)
    assert torch.all(state[10:] == 0)
