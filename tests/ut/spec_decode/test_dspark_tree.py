# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.spec_decode.dspark_tree import (
    ROOT,
    DraftNode,
    TreeAcceptanceStats,
    accepted_tree_tokens,
    build_draft_trees,
    prune_tree,
)


class MarkovModel:
    def markov_embed(self, ids):
        return ids

    def markov_bias(self, previous):
        bias = torch.zeros(len(previous), 4)
        bias[previous == 11, 2] = 10
        return bias

    def apply_markov_bias_gathered(self, previous, logits, values, indices):
        return logits.scatter_(1, indices, values + self.markov_bias(previous).gather(1, indices))

    def map_draft_to_target(self, ids):
        return ids + 10

    def compute_confidence(self, hidden, previous):
        return hidden[:, :1]


@pytest.mark.parametrize("use_confidence", [False, True])
@pytest.mark.parametrize(("topk", "accepted"), [(None, 2), (2, 1)])
def test_markov_branches_use_their_own_parent_and_candidate_support(topk, accepted, use_confidence):
    logits = torch.tensor([[[2.0, 1.9, -5.0, -20.0], [3.0, 2.0, -2.0, -20.0]]])
    hidden = torch.full((1, 2, 1), 0.9)
    tree = build_draft_trees(MarkovModel(), logits, hidden, torch.tensor([10]), topk, use_confidence)[0]
    assert accepted_tree_tokens(tree, [11, 12]) == accepted
    assert accepted_tree_tokens(tree, [10, 10]) == 2
    assert accepted_tree_tokens(tree, [13, 13]) == 0


def test_pruning_keeps_tied_ancestors_and_remaps_reordered_parents():
    original = [
        DraftNode(10, ROOT, 0.2),
        DraftNode(11, ROOT, 0.9),
        DraftNode(12, 0, 0.2),
        DraftNode(13, 1, 0.9),
    ]
    assert prune_tree(original, 1) == [DraftNode(11, ROOT, 0.9)]
    assert prune_tree(original, 2) == [DraftNode(11, ROOT, 0.9), DraftNode(13, 0, 0.9)]
    assert accepted_tree_tokens(prune_tree(original, 3), [10, 12]) == 1
    assert accepted_tree_tokens(prune_tree(original, 4), [10, 12]) == 2


def test_nonfinite_confidence_uses_relative_markov_scores():
    logits = torch.tensor([[[2.0, 1.9, -5.0, -20.0]]])
    hidden = torch.full((1, 1, 1), torch.nan)
    tree = build_draft_trees(MarkovModel(), logits, hidden, torch.tensor([10]), None, True)[0]
    assert tree[0].score == 1.0
    assert 0.0 < tree[2].score < tree[1].score < tree[0].score


def test_paired_acceptance_joins_confirmed_steps_and_excludes_terminal_tail():
    stats = TreeAcceptanceStats(depth=3)
    tree = [
        DraftNode(1, ROOT, 0.9),
        DraftNode(8, ROOT, 0.8),
        DraftNode(2, 0, 0.7),
        DraftNode(9, 1, 0.6),
    ]
    stats.propose("a", [1, 2, 3], tree)
    stats.observe("a", [8], scheduled_drafts=3)
    stats.observe("a", [9, 4, 5], scheduled_drafts=3)
    assert stats.summary()["paired"] == 0
    stats.observe("a", [7], scheduled_drafts=3)
    summary = stats.summary()
    assert summary["paired"] == 1
    assert summary["chain"]["mean_accepted"] == 0.0
    assert summary["tree_equal"]["mean_accepted"] == 1.0
    assert summary["tree_wide"]["mean_accepted"] == 2.0

    stats.propose("b", [1, 2, 3], tree)
    stats.observe("b", [1, 2, 3, 4], scheduled_drafts=3)
    stats.discard(["b"])
    assert stats.summary()["paired"] == 1
    assert stats.summary()["dropped"] == 1


def test_paired_acceptance_excludes_unscheduled_and_mismatched_chains():
    stats = TreeAcceptanceStats(depth=2)
    stats.propose("a", [1, 2], [])
    stats.observe("a", [1], scheduled_drafts=0)
    stats.propose("a", [1, 2], [])
    stats.observe("a", [1], scheduled_drafts=2)
    stats.observe("a", [2], scheduled_drafts=0)
    stats.observe("a", [3], scheduled_drafts=0)
    assert stats.summary()["paired"] == 0
    assert stats.summary()["dropped"] == 1
    assert stats.summary()["chain_mismatch"] == 1
