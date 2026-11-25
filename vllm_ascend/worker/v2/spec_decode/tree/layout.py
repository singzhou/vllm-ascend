# SPDX-License-Identifier: Apache-2.0
"""Packed query geometry and accepted-path transport for Qwen tree verification."""

from dataclasses import dataclass

import torch

from vllm_ascend.worker.v2.spec_decode.tree.draft import DraftNode


@dataclass(frozen=True)
class VerifyRow:
    row: int
    start: int
    parents: tuple[int, ...]
    prefix: int
    tree: tuple[DraftNode, ...] | None

    @property
    def width(self):
        return len(self.parents)

    @property
    def depths(self):
        depths = [0]
        for parent in self.parents[1:]:
            depths.append(depths[parent] + 1)
        return depths

    def walk(self, target_ids):
        """Return generated tokens (including the bonus) and accepted node IDs."""
        children = {(node.parent, node.token): i for i, node in enumerate(self.tree, 1)}
        tokens, path = [], []
        node = 0
        while True:
            token = target_ids[node]
            tokens.append(token)
            node = children.get((node, token))
            if node is None:
                return tokens, path
            path.append(node)


@dataclass(frozen=True)
class GDNStep:
    tokens: torch.Tensor
    rows: torch.Tensor
    nodes: torch.Tensor
    parents: torch.Tensor
    lengths: torch.Tensor


class TreeBatch:
    def __init__(self, rows, positions):
        self.rows = rows
        self.tree_rows = tuple(row for row in rows if row.tree is not None)
        self.slot_positions = positions
        self.block_tables = None
        self.slot_mappings = None
        self.verified = {}
        self.aux_hidden_states = None
        self._gdn_steps = None
        self._attention_masks = {}

    @classmethod
    def from_schedule(cls, batch, proposals, scheduled_drafts):
        rows = []
        for row, req_id in enumerate(batch.req_ids):
            proposal = proposals.pop(req_id, None)
            scheduled = scheduled_drafts.get(req_id)
            start, end = (int(offset) for offset in batch.query_start_loc_np[row : row + 2])
            if not scheduled or end - start != len(scheduled) + 1:
                continue
            tree = proposal[: len(scheduled)] if proposal is not None else None
            if tree is not None and (
                len(tree) != len(scheduled)
                or any(token != -1 and token != node.token for token, node in zip(scheduled, tree))
            ):
                # A flat scheduler rewrite cannot safely verify a branched
                # draft. Structured-output requests never enter tree drafting.
                raise RuntimeError("Scheduled draft tokens no longer match the published tree")
            parents = (0, *(node.parent for node in tree)) if tree is not None else (0, *range(len(scheduled)))
            rows.append(VerifyRow(row, start, parents, int(batch.num_computed_tokens_np[row]), tree))
        return cls(rows, batch.positions) if rows else None

    def rope_positions(self):
        positions = self.slot_positions.clone()
        for row in self.tree_rows:
            positions[row.start : row.start + row.width] = positions[row.start] + positions.new_tensor(row.depths)
        return positions

    def attention_mask(self, row, kv_len):
        cache_key = (row.row, kv_len)
        if cache_key not in self._attention_masks:
            # FIA uses True for masked entries. Match the upstream BSND
            # contract: the cached prefix and precisely the node's ancestors.
            visible = torch.eye(row.width, dtype=torch.bool)
            for node, parent in enumerate(row.parents[1:], 1):
                visible[node] |= visible[parent]
            mask = torch.ones((1, row.width, kv_len), dtype=torch.bool, device=self.slot_positions.device)
            mask[:, :, : row.prefix] = False
            mask[:, :, row.prefix : row.prefix + row.width] = (~visible).to(mask.device)
            self._attention_masks[cache_key] = mask
        return self._attention_masks[cache_key]

    def gdn_steps(self):
        """GDN packs spec rows in batch order. Group independent nodes by depth."""
        if self._gdn_steps is None:
            groups = []
            offset = 0
            for i, row in enumerate(self.rows):
                for node, (parent, depth) in enumerate(zip(row.parents, row.depths)):
                    while len(groups) <= depth:
                        groups.append([])
                    groups[depth].append((offset + node, i, node, parent))
                offset += row.width
            self._gdn_steps = []
            device = self.slot_positions.device
            for group in groups:
                columns = torch.tensor(group, dtype=torch.long, device=device).unbind(1)
                # The recurrent kernel expects an initial zero followed by one
                # length per sequence, with independent one-token sequences.
                lengths = torch.ones(len(group) + 1, dtype=torch.int32, device=device)
                lengths[0] = 0
                self._gdn_steps.append(GDNStep(*columns, lengths))
        return self._gdn_steps


def gather_cache_rows(cache, indices):
    if cache.device.type == "npu":
        from vllm_ascend.ops.triton.spec_decode.tree_kv_transport import gather_tree_cache_rows

        values = cache.new_empty((indices.shape[0], *cache.shape[indices.shape[1] :]))
        gather_tree_cache_rows(cache, indices, values)
        return values
    return cache[tuple(indices.unbind(dim=1))]


def scatter_cache_rows(cache, indices, values):
    if cache.device.type == "npu":
        from vllm_ascend.ops.triton.spec_decode.tree_kv_transport import scatter_tree_cache_rows_

        scatter_tree_cache_rows_(cache, indices, values)
    else:
        cache[tuple(indices.unbind(dim=1))] = values


def copy_paged_kv(cache, source, destination):
    """Gather every source before writing any destination, even for overlap."""
    block_size = cache.shape[1]
    src = torch.stack((source // block_size, source % block_size), dim=1)
    dst = torch.stack((destination // block_size, destination % block_size), dim=1)
    scatter_cache_rows(cache, dst, gather_cache_rows(cache, src))
