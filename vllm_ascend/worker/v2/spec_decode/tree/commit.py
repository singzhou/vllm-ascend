# SPDX-License-Identifier: Apache-2.0
"""Greedy tree walk and accepted-path commitment before request advancement."""

from dataclasses import dataclass

import torch

from vllm_ascend.worker.v2.spec_decode.tree.layout import copy_paged_kv, gather_cache_rows, scatter_cache_rows


@dataclass(frozen=True)
class CacheGroup:
    group_id: int
    recurrent: bool
    caches: tuple[torch.Tensor, ...]


def tree_argmax(logits, input_batch):
    """Read only the target argmax IDs, retaining each packed row's boundaries."""
    batch = input_batch.tree_batch
    indices = []
    for row in batch.tree_rows:
        start, end = input_batch.cu_num_logits_np[row.row : row.row + 2]
        if end - start != row.width:
            raise RuntimeError("Tree query rows and target logits must have the same width")
        indices.extend(range(start, end))
    if not indices:
        return {}
    target_tokens = logits.argmax(dim=-1).index_select(0, torch.tensor(indices, device=logits.device)).tolist()
    results = {}
    offset = 0
    for row in batch.tree_rows:
        results[row.row] = row.walk(target_tokens[offset : offset + row.width])
        offset += row.width
    return results


def commit_tree_batch(input_batch, sampler_output, verified, cache_groups, hidden_states, aux_hidden_states):
    batch = input_batch.tree_batch
    chain_rows = [row.row for row in batch.rows if row.row not in verified]
    chain_counts = dict(zip(chain_rows, sampler_output.num_sampled[chain_rows].tolist())) if chain_rows else {}
    src_tokens, dst_tokens = [], []
    state_rows, state_nodes = [], []
    for row in batch.rows:
        if row.row in verified:
            tokens, path = verified[row.row]
            sampler_output.sampled_token_ids[row.row].fill_(-1)
            sampler_output.sampled_token_ids[row.row, : len(tokens)] = sampler_output.sampled_token_ids.new_tensor(
                tokens
            )
            sampler_output.num_sampled[row.row] = len(tokens)
            sampler_output.num_rejected[row.row] = row.width - len(tokens)
            if path:
                src_tokens.extend(row.start + node for node in path)
                dst_tokens.extend(range(row.start + 1, row.start + 1 + len(path)))
            endpoint = path[-1] if path else 0
        else:
            endpoint = max(chain_counts[row.row] - 1, 0)
        if endpoint:
            state_rows.append(row.row)
            state_nodes.append(endpoint)

    source = torch.tensor(src_tokens, dtype=torch.long, device=hidden_states.device)
    destination = torch.tensor(dst_tokens, dtype=torch.long, device=hidden_states.device)
    request_rows = torch.tensor(state_rows, dtype=torch.long, device=hidden_states.device)
    endpoints = torch.tensor(state_nodes, dtype=torch.long, device=hidden_states.device)
    for group in cache_groups:
        if group.recurrent and state_rows:
            table = batch.block_tables[group.group_id]
            source_pages = table[request_rows, endpoints, None]
            destination_pages = table[request_rows, 0, None]
            for cache in group.caches:
                scatter_cache_rows(cache, destination_pages, gather_cache_rows(cache, source_pages))
        elif not group.recurrent and src_tokens:
            slots = batch.slot_mappings[group.group_id]
            source_slots = slots[source].long()
            destination_slots = slots[destination].long()
            for cache in group.caches:
                copy_paged_kv(cache, source_slots, destination_slots)

    if src_tokens:
        # DFlash/DSpark consume the left-aligned committed hidden rows. Their
        # context KV code assigns physical slots from these linear positions.
        for hidden in (hidden_states, *(aux_hidden_states or ())):
            hidden.index_copy_(0, destination, hidden.index_select(0, source))
