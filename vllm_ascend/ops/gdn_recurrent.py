# SPDX-License-Identifier: Apache-2.0
"""Select GDN verification kernels without changing their operator contracts."""

import torch

from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_

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
    """Call the unchanged legacy kernel, compacting only active cache rows if strided.

    The original kernel addresses dense state rows. Hybrid caches can have page
    padding between rows, so gather the token slots into dense scratch, remap
    indices, then publish only live rows back to the original cache. All lengths
    and validity masks stay on device, including during graph capture/replay.
    """
    if state.is_contiguous():
        return torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
            query,
            key,
            value,
            state,
            ssm_state_indices=ssm_state_indices,
            actual_seq_lengths=actual_seq_lengths,
            **kwargs,
        )
    token_slots = torch.arange(query.shape[0], dtype=torch.int32, device=query.device)
    valid_rows = (token_slots < actual_seq_lengths[1:].sum()) & (ssm_state_indices >= 0)
    compact_state = gather_ssm_states(state, ssm_state_indices, valid_rows)
    output = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query,
        key,
        value,
        compact_state,
        ssm_state_indices=token_slots,
        actual_seq_lengths=actual_seq_lengths,
        **kwargs,
    )
    scatter_ssm_states_(state, ssm_state_indices, compact_state, valid_rows=valid_rows)
    return output
