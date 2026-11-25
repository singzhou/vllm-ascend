# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.gdn_tree import tree_conv, tree_recurrent
from vllm_ascend.worker.v2.spec_decode.tree.commit import CacheGroup, commit_tree_batch, tree_argmax
from vllm_ascend.worker.v2.spec_decode.tree.draft import DraftNode, build_dflash2_trees, build_dspark_trees
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeBatch, VerifyRow


class MarkovDraft:
    def markov_embed(self, previous):
        return previous

    def markov_bias(self, previous):
        return torch.nn.functional.one_hot(previous % 3, 3) * 3.0

    def apply_markov_bias_gathered(self, previous, scratch, values, indices):
        return scratch.scatter_(1, indices, values + self.markov_bias(previous).gather(1, indices))

    def map_draft_to_target(self, tokens):
        return tokens + 3

    def compute_confidence(self, hidden, previous):
        return torch.full_like(previous, 0.2, dtype=torch.float)


@pytest.mark.parametrize("draft_topk", [None, 2])
def test_dspark_tree_uses_parent_conditioned_reduced_vocabulary(draft_topk):
    logits = torch.tensor([[[1.0, 0.9, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])
    (tree,) = build_dspark_trees(MarkovDraft(), logits, torch.zeros(1, 3, 2), torch.tensor([4]), draft_topk, True, 2)
    assert [(node.token, node.parent) for node in tree] == [(4, 0), (4, 1), (3, 0)]


def test_dflash2_tree_uses_selector_parent_slot():
    candidates = torch.tensor([[[7, 8], [9, 10], [11, 12]]])
    scores = torch.tensor([[[[1.0, 0.9], [0, 0]], [[0, 0], [0, 5]], [[0, 0], [0, 5]]]])
    (tree,) = build_dflash2_trees(candidates, scores, 2)
    assert [(node.token, node.parent) for node in tree] == [(7, 0), (8, 0), (10, 2)]


def _verify_row(row=0, start=0, prefix=5):
    tree = (DraftNode(10, 0, 0), DraftNode(11, 0, -1), DraftNode(12, 2, -2))
    return VerifyRow(row, start, (0, 0, 0, 2), prefix, tree)


def test_schedule_truncation_keeps_parent_ids_and_separates_rope_from_kv():
    row = _verify_row()
    batch = SimpleNamespace(
        req_ids=["first", "prefill"],
        query_start_loc_np=[0, 3, 8],
        num_computed_tokens_np=[5, 0],
        positions=torch.tensor([5, 6, 7, 0, 1, 2, 3, 4]),
    )
    proposals = {"first": row.tree, "prefill": row.tree}
    layout = TreeBatch.from_schedule(batch, proposals, {"first": [-1, -1]})
    assert proposals == {}
    assert layout.rope_positions().tolist() == [5, 6, 6, 0, 1, 2, 3, 4]
    assert layout.slot_positions.tolist() == [5, 6, 7, 0, 1, 2, 3, 4]


def test_tree_mask_includes_prefix_and_ancestors_only():
    row = _verify_row()
    layout = TreeBatch([row], torch.arange(5, 9))
    mask = layout.attention_mask(row, 12)[0]
    assert not mask[:, :5].any()
    assert mask[:, 9:].all()
    assert (~mask[:, 5:9]).tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, False, True, False],
        [True, False, True, True],
    ]


@pytest.mark.parametrize("dim_first", [False, True])
def test_gdn_conv_and_recurrent_load_parent_snapshot(dim_first):
    row = _verify_row()
    layout = TreeBatch([row], torch.arange(4))
    pages = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
    state = torch.zeros(6, 3, 1)
    state[1, 0, 0] = 4
    if dim_first:
        state = state.transpose(1, 2).contiguous()
    x = torch.tensor([[1.0], [2.0], [3.0], [5.0]])
    conv = tree_conv(
        x, torch.tensor([[10.0], [1.0]]), None, state, pages, layout.gdn_steps(), False, dim_first=dim_first
    )
    assert conv.flatten().tolist() == [41, 12, 13, 35]

    def recurrent(_query, _key, value, state, *, ssm_state_indices, actual_seq_lengths, **_kwargs):
        assert actual_seq_lengths.tolist() == [0] + [1] * value.shape[0]
        new_state = state[ssm_state_indices] + value
        state[ssm_state_indices] = new_state
        return new_state

    recurrent_state = torch.tensor([[0.0], [10.0], [0.0], [0.0], [0.0], [0.0]])
    x = x.unsqueeze(0)
    result = tree_recurrent(x, x, x, x, x, recurrent_state, pages, layout.gdn_steps(), recurrent)
    assert result.flatten().tolist() == [11, 13, 14, 19]


def test_verify_commits_off_chain_kv_aux_hidden_and_gdn_state_in_mixed_batch():
    row = _verify_row(row=1, start=3)
    chain = VerifyRow(0, 0, (0, 0, 1), 2, None)
    batch = TreeBatch([chain, row], torch.arange(7))
    batch.block_tables = (torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]]),)
    batch.slot_mappings = torch.tensor([[2, 3, 4, 7, 8, 9, 10], [2, 3, 4, 7, 8, 9, 10]])
    input_batch = SimpleNamespace(tree_batch=batch, cu_num_logits_np=[0, 3, 7])
    logits = torch.nn.functional.one_hot(torch.tensor([1, 2, 3, 11, 15, 12, 13]), num_classes=16).float()
    verified = tree_argmax(logits, input_batch)
    assert verified == {1: ([11, 12, 13], [2, 3])}
    output = SimpleNamespace(
        sampled_token_ids=torch.tensor([[1, 2, -1, -1], [11, -1, -1, -1]]),
        num_sampled=torch.tensor([2, 1]),
        num_rejected=torch.tensor([1, 3]),
    )
    state = torch.arange(8).reshape(8, 1)
    kv = torch.arange(16).reshape(4, 4, 1)
    hidden = torch.arange(7).reshape(7, 1)
    aux = torch.arange(100, 107).reshape(7, 1)
    commit_tree_batch(
        input_batch, output, verified, (CacheGroup(0, True, (state,)), CacheGroup(1, False, (kv,))), hidden, [aux]
    )
    assert output.sampled_token_ids.tolist() == [[1, 2, -1, -1], [11, 12, 13, -1]]
    assert output.num_sampled.tolist() == [2, 3]
    assert output.num_rejected.tolist() == [1, 1]
    assert state.flatten().tolist() == [0, 2, 2, 3, 7, 5, 6, 7]
    assert kv.flatten().tolist()[7:11] == [7, 9, 10, 10]
    assert hidden.flatten().tolist() == [0, 1, 2, 3, 5, 6, 6]
    assert aux.flatten().tolist() == [100, 101, 102, 103, 105, 106, 106]


def test_tree_root_rejection_returns_target_bonus_without_a_draft():
    assert _verify_row().walk([14, 11, 12, 13]) == ([14], [])


def test_tree_schedule_rejects_changed_token_ids():
    row = _verify_row()
    batch = SimpleNamespace(
        req_ids=["first"],
        query_start_loc_np=[0, 3],
        num_computed_tokens_np=[5],
        positions=torch.arange(5, 8),
    )
    with pytest.raises(RuntimeError, match="no longer match"):
        TreeBatch.from_schedule(batch, {"first": row.tree}, {"first": [10, 12]})
