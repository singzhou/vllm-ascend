# SPDX-License-Identifier: Apache-2.0
"""Greedy DSpark tree drafting and paired acceptance measurement."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite

import torch

TREE_NODE_BUDGET = 15
TREE_FRONTIER_WIDTH = 4
TREE_CHILDREN_PER_PARENT = 3
ROOT = -1


@dataclass(frozen=True)
class DraftNode:
    token: int
    parent: int
    score: float


def prune_tree(nodes: list[DraftNode], budget: int) -> list[DraftNode]:
    # Scores are non-increasing along a path. Parents are generated first, so
    # stable tie breaking keeps every prefix of the result ancestor-closed.
    ranked = sorted(range(len(nodes)), key=lambda i: -nodes[i].score)[:budget]
    remap = {old: new for new, old in enumerate(ranked)}
    remap[ROOT] = ROOT
    return [DraftNode(nodes[i].token, remap[nodes[i].parent], nodes[i].score) for i in ranked]


def build_draft_trees(
    model,
    base_logits: torch.Tensor,
    head_hidden: torch.Tensor,
    anchors: torch.Tensor,
    draft_topk: int | None,
    use_confidence: bool,
) -> list[list[DraftNode]]:
    """Expand each parent's Markov row using the chain's candidate support."""
    batch_size, depth, vocab_size = base_logits.shape
    base_topk = base_logits.topk(draft_topk, dim=-1) if draft_topk is not None else None
    trees: list[list[DraftNode]] = [[] for _ in range(batch_size)]
    parents = [[ROOT] for _ in range(batch_size)]
    previous_tokens = anchors
    for step in range(depth):
        width = len(parents[0])
        embed = model.markov_embed(previous_tokens)
        if base_topk is None:
            base = base_logits[:, step].repeat_interleave(width, dim=0)
            logits = base + model.markov_bias(embed)
        else:
            values = base_topk.values[:, step].repeat_interleave(width, dim=0)
            indices = base_topk.indices[:, step].repeat_interleave(width, dim=0)
            scratch = base_logits.new_full((batch_size * width, vocab_size), -torch.inf)
            corrected = model.apply_markov_bias_gathered(embed, scratch, values, indices)
            logits = corrected.gather(1, indices)
        values, ids = logits.topk(min(TREE_CHILDREN_PER_PARENT, logits.shape[-1]), dim=-1)
        if base_topk is not None:
            ids = indices.gather(1, ids)
        values = values.float()
        if use_confidence:
            hidden = head_hidden[:, step].repeat_interleave(width, dim=0)
            confidence = model.compute_confidence(hidden, embed).float().reshape(-1, 1)
            # Match adaptive verification's fallback for non-finite confidence.
            confidence = torch.nan_to_num(confidence, nan=1.0, posinf=1.0, neginf=1.0)
            # q(child)/q(best) is exp(logit(child)-logit(best)).
            scores = confidence * (values - values[:, :1]).exp()
        else:
            scores = (values - logits.float().logsumexp(dim=-1, keepdim=True)).exp()
        token_rows = model.map_draft_to_target(ids).view(batch_size, width, -1).tolist()
        score_rows = scores.view(batch_size, width, -1).tolist()

        next_parents = []
        next_tokens = []
        for nodes, parent_ids, token_row, score_row in zip(trees, parents, token_rows, score_rows):
            start = len(nodes)
            for parent, tokens, local_scores in zip(parent_ids, token_row, score_row):
                parent_score = 1.0 if parent == ROOT else nodes[parent].score
                nodes.extend(
                    DraftNode(token, parent, parent_score * score if isfinite(score) else 0.0)
                    for token, score in zip(tokens, local_scores)
                )
            frontier = sorted(range(start, len(nodes)), key=lambda i: -nodes[i].score)[:TREE_FRONTIER_WIDTH]
            next_parents.append(frontier)
            next_tokens.extend(nodes[i].token for i in frontier)
        parents = next_parents
        previous_tokens = anchors.new_tensor(next_tokens)

    return [prune_tree(nodes, TREE_NODE_BUDGET) for nodes in trees]


def accepted_chain_tokens(chain: list[int], reference: list[int]) -> int:
    for depth, (draft, target) in enumerate(zip(chain, reference)):
        if draft != target:
            return depth
    return min(len(chain), len(reference))


def accepted_tree_tokens(tree: list[DraftNode], reference: list[int]) -> int:
    children = {(node.parent, node.token): i for i, node in enumerate(tree)}
    parent = ROOT
    for depth, token in enumerate(reference):
        child = children.get((parent, token))
        if child is None:
            return depth
        parent = child
    return len(reference)


@dataclass
class ProposalComparison:
    chain: list[int]
    tree: list[DraftNode]
    reference: list[int] = field(default_factory=list)
    observed_chain_acceptance: int = 0


@dataclass
class RequestComparisons:
    proposed: ProposalComparison | None = None
    active: list[ProposalComparison] = field(default_factory=list)
    unconfirmed: list[int] = field(default_factory=list)


class TreeAcceptanceStats:
    """Compare both trees with the actual linear decode at matched prefixes."""

    def __init__(self, depth: int):
        self.depth = depth
        self.requests: dict[str, RequestComparisons] = {}
        self.histograms: tuple[Counter[int], ...] = (Counter(), Counter(), Counter())
        self.dropped = 0
        self.mismatched = 0
        self.paired = 0

    def propose(self, req_id: str, chain: list[int], tree: list[DraftNode]) -> None:
        state = self.requests.setdefault(req_id, RequestComparisons())
        state.proposed = ProposalComparison(chain, tree)

    def observe(self, req_id: str, sampled: list[int], scheduled_drafts: int) -> None:
        state = self.requests.get(req_id)
        if state is None or not sampled:
            return

        # Seeing another step confirms the previous output was committed by the
        # synchronous scheduler. A terminal step may contain tokens after EOS
        # or the generation limit, so its output is never used as reference.
        pending = []
        for trial in state.active:
            trial.reference.extend(state.unconfirmed)
            if len(trial.reference) < self.depth:
                pending.append(trial)
                continue
            reference = trial.reference[: self.depth]
            chain = accepted_chain_tokens(trial.chain, reference)
            if chain != trial.observed_chain_acceptance:
                self.mismatched += 1
                continue
            lengths = (
                chain,
                accepted_tree_tokens(trial.tree[: self.depth], reference),
                accepted_tree_tokens(trial.tree, reference),
            )
            for histogram, length in zip(self.histograms, lengths):
                histogram[length] += 1
            self.paired += 1

        if state.proposed is not None:
            if scheduled_drafts == self.depth:
                state.proposed.observed_chain_acceptance = len(sampled) - 1
                pending.append(state.proposed)
            else:
                self.dropped += 1
        state.proposed = None
        state.active = pending
        state.unconfirmed = sampled

    def discard(self, req_ids: Iterable[str]) -> bool:
        discarded = False
        for req_id in req_ids:
            state = self.requests.pop(req_id, None)
            if state is not None:
                self.dropped += len(state.active) + (state.proposed is not None)
                discarded = True
        return discarded

    def summary(self) -> dict:
        result = {"paired": self.paired, "dropped": self.dropped, "chain_mismatch": self.mismatched}
        for budget, name, histogram in zip(
            (self.depth, self.depth, TREE_NODE_BUDGET), ("chain", "tree_equal", "tree_wide"), self.histograms
        ):
            counts = [histogram[i] for i in range(self.depth + 1)]
            mean = sum(i * count for i, count in enumerate(counts)) / self.paired if self.paired else None
            result[name] = {"budget": budget, "accepted_histogram": counts, "mean_accepted": mean}
        return result
