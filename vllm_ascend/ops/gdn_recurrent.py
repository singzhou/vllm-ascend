# SPDX-License-Identifier: Apache-2.0
"""Select GDN verification kernels without changing their operator contracts."""

import torch

FLA_GDN_MAX_VERIFY_TOKENS = 8
LEGACY_GDN_MAX_VERIFY_TOKENS = 16


def select_gdn_recurrent_backend(num_speculative_tokens: int, use_dcut: bool) -> str:
    """Select from the configured maximum, never the current device query lengths."""
    verify_tokens = num_speculative_tokens + 1
    if not 1 <= verify_tokens <= LEGACY_GDN_MAX_VERIFY_TOKENS:
        raise ValueError("Ascend GDN requires 0 <= num_speculative_tokens <= 15")
    if use_dcut:
        return "dcut"
    if verify_tokens > FLA_GDN_MAX_VERIFY_TOKENS:
        return "legacy"
    return "fla"


def legacy_recurrent_gated_delta_rule(query, key, value, state, *, ssm_state_indices, actual_seq_lengths, **kwargs):
    """Pass the original cache view and indices to the stride-aware legacy kernel."""
    return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        state,
        ssm_state_indices=ssm_state_indices,
        actual_seq_lengths=actual_seq_lengths,
        **kwargs,
    )
