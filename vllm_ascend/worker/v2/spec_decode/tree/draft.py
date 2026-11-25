# SPDX-License-Identifier: Apache-2.0
"""Eager parent-conditioned drafting for the MRV2 tree verifier."""

from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True)
class DraftNode:
    token: int
    parent: int  # Node 0 is the anchor. Draft nodes are numbered from 1.
    score: float


def build_tree(batch_size, depth, topk, score_children):
    """Expand a beam at each depth, then keep an ancestor-closed scored prefix."""
    pools = [[DraftNode(-1, 0, 0.0)] for _ in range(batch_size)]
    frontier = [[0] for _ in pools]
    for step in range(depth):
        token_rows, score_rows = score_children(step, pools, frontier)
        next_frontier = []
        for nodes, parents, tokens, scores in zip(pools, frontier, token_rows, score_rows):
            start = len(nodes)
            for parent, child_tokens, child_scores in zip(parents, tokens, scores):
                for token, score in zip(child_tokens, child_scores):
                    score = min(score, 0.0) if isfinite(score) else -float("inf")
                    nodes.append(DraftNode(token, parent, nodes[parent].score + score))
            ranked = sorted(range(start, len(nodes)), key=lambda i: -nodes[i].score)
            next_frontier.append(ranked[:topk])
        frontier = next_frontier

    trees = []
    for nodes in pools:
        # Child log scores cannot exceed the parent score. Stable ordering
        # resolves ties in favor of the earlier (ancestor) node.
        selected = sorted(range(1, len(nodes)), key=lambda i: -nodes[i].score)[:depth]
        remap = {old: new for new, old in enumerate([0, *selected])}
        trees.append(tuple(DraftNode(nodes[i].token, remap[nodes[i].parent], nodes[i].score) for i in selected))
    return trees


def build_dspark_trees(model, base_logits, hidden, anchors, draft_topk, use_confidence, topk):
    """Score each branch with the Markov head conditioned on its immediate parent."""
    batch, depth, vocab = base_logits.shape
    support = base_logits.topk(min(draft_topk, vocab), dim=-1) if draft_topk is not None else None
    child_count = min(topk, vocab if support is None else support.indices.shape[-1])

    def score_children(step, pools, frontier):
        width = len(frontier[0])
        parent_tokens = (
            anchors
            if step == 0
            else anchors.new_tensor(
                [nodes[parent].token for nodes, parents in zip(pools, frontier) for parent in parents]
            )
        )
        embed = model.markov_embed(parent_tokens)
        if support is None:
            logits = base_logits[:, step].repeat_interleave(width, dim=0) + model.markov_bias(embed)
        else:
            indices = support.indices[:, step].repeat_interleave(width, dim=0)
            values = support.values[:, step].repeat_interleave(width, dim=0)
            scratch = base_logits.new_full((batch * width, vocab), -torch.inf)
            logits = model.apply_markov_bias_gathered(embed, scratch, values, indices).gather(1, indices)
        values, tokens = logits.float().topk(child_count, dim=-1)
        if support is not None:
            tokens = indices.gather(1, tokens)
        if use_confidence:
            features = hidden[:, step].repeat_interleave(width, dim=0)
            confidence = model.compute_confidence(features, embed).float().reshape(-1, 1)
            confidence = torch.nan_to_num(confidence, nan=1.0, posinf=1.0, neginf=1.0).clamp(0, 1)
            scores = confidence.log() + values - values[:, :1]
        else:
            scores = values - logits.float().logsumexp(dim=-1, keepdim=True)
        return (
            model.map_draft_to_target(tokens).reshape(batch, width, child_count).tolist(),
            scores.reshape(batch, width, child_count).tolist(),
        )

    return build_tree(batch, depth, topk, score_children)


def build_dflash2_trees(candidate_ids, selector_scores, topk):
    """Use the selector's parent-slot rows and normalize before pruning children."""
    batch, depth, candidates = candidate_ids.shape
    topk = min(topk, candidates)
    candidate_lists = candidate_ids.tolist()
    values, child_slots = selector_scores.float().log_softmax(dim=-1).topk(topk, dim=-1)
    value_rows, slot_rows = values.tolist(), child_slots.tolist()

    def score_children(step, pools, frontier):
        tokens, scores = [], []
        for row, parents in enumerate(frontier):
            parent_slots = (
                [0]
                if step == 0
                else [candidate_lists[row][step - 1].index(pools[row][parent].token) for parent in parents]
            )
            tokens.append(
                [
                    [candidate_lists[row][step][child] for child in slot_rows[row][step][parent]]
                    for parent in parent_slots
                ]
            )
            scores.append([value_rows[row][step][parent] for parent in parent_slots])
        return tokens, scores

    return build_tree(batch, depth, topk, score_children)


def supports_tree_sampling(params):
    """The raw target argmax is valid for requests without logits processors."""
    return (
        params is not None
        and params.temperature == 0
        and params.presence_penalty == 0
        and params.frequency_penalty == 0
        and params.repetition_penalty == 1
        and params.min_tokens == 0
        and params.logprobs is None
        and params.structured_outputs is None
        and not params.logit_bias
        and not params.allowed_token_ids
        and not params.bad_words
        and getattr(params, "thinking_token_budget", None) is None
        and not getattr(params, "logits_processors", None)
    )


class TreeDraftMixin:
    """Own proposal dispatch and publish geometry alongside fixed-width draft tokens."""

    def init_tree_drafting(self, topk):
        self.tree_topk = topk
        self.tree_proposals = {}
        self._tree_draft_rows = ()
        self._draft_trees = None

    def propose(self, input_batch, *args, **kwargs):
        rows = getattr(input_batch, "tree_draft_rows", ()) if getattr(self, "tree_topk", 0) else ()
        if not rows:
            return self._propose_draft(input_batch, *args, **kwargs)

        self._tree_draft_rows = rows
        try:
            tokens = self._propose_draft(input_batch, *args, **kwargs)
            if self._draft_trees is None or len(self._draft_trees) != len(rows):
                raise RuntimeError(f"{type(self).__name__} did not produce a draft tree for every eligible row")
            draft_ids = tokens.new_tensor([[node.token for node in tree] for tree in self._draft_trees])
            tokens = tokens.clone()
            tokens[list(rows)] = draft_ids
            self.tree_proposals.update(zip((input_batch.req_ids[row] for row in rows), self._draft_trees))
            return tokens
        finally:
            self._tree_draft_rows = ()
            self._draft_trees = None

    def _propose_draft(self, input_batch, *args, **kwargs):
        """Run the drafter; subclasses can override this hook to stage metadata."""
        return super().propose(input_batch, *args, **kwargs)
