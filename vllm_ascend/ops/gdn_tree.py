# SPDX-License-Identifier: Apache-2.0
"""Eager GDN snapshots using the existing one-token recurrent kernel."""

import torch
import torch.nn.functional as F

from vllm_ascend.worker.v2.spec_decode.tree.layout import gather_cache_rows, scatter_cache_rows


def tree_conv(x, weights, bias, state, pages, steps, activate, *, dim_first):
    # Each node keeps its final W-1 taps in its own physical state page.
    taps = weights.shape[0] - 1
    weights = weights.float()
    output = torch.empty_like(x)
    for step in steps:
        parents = pages[step.rows, step.parents, None]
        destinations = pages[step.rows, step.nodes, None]
        snapshots = gather_cache_rows(state, parents)
        history = (snapshots.transpose(1, 2) if dim_first else snapshots)[:, :taps]
        window = torch.cat((history, x.index_select(0, step.tokens).unsqueeze(1)), dim=1)
        convolved = (window.float() * weights).sum(dim=1)
        if bias is not None:
            convolved += bias.float()
        if activate:
            convolved = F.silu(convolved)
        history.copy_(window[:, 1:])
        scatter_cache_rows(state, destinations, snapshots)
        output.index_copy_(0, step.tokens, convolved.to(x.dtype))
    return output


def tree_recurrent(query, key, value, g, beta, state, pages, steps, recurrent_op):
    output = torch.empty_like(value)
    for depth, step in enumerate(steps):
        destinations = pages[step.rows, step.nodes].contiguous()
        if depth:
            parents = pages[step.rows, step.parents, None]
            scatter_cache_rows(state, destinations.unsqueeze(1), gather_cache_rows(state, parents))
        index = step.tokens
        out = recurrent_op(
            query.index_select(1, index).squeeze(0),
            key.index_select(1, index).squeeze(0),
            value.index_select(1, index).squeeze(0),
            state,
            g=g.index_select(1, index).squeeze(0),
            beta=beta.index_select(1, index).squeeze(0),
            scale=key.shape[-1] ** -0.5,
            actual_seq_lengths=step.lengths,
            ssm_state_indices=destinations,
        )
        output.index_copy_(1, index, out.unsqueeze(0))
    return output
