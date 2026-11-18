# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for dummy rounds, dynamic K and padded draft vocabularies."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import math
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "_dspark_probability_harness", Path(__file__).with_name("test_dspark_probabilistic.py")
)
_HARNESS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HARNESS)


@pytest.fixture
def proposer_class():
    return _HARNESS.proposer_class.__wrapped__()


def _load_method(path, class_name, name, namespace):
    tree = ast.parse((_ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), path, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("metadata", [None, _HARNESS._metadata([0.7])])
def test_single_request_dummy_never_inherits_two_request_metadata(proposer_class, metadata):
    proposer = _HARNESS._proposer(proposer_class, num_reqs=1)
    proposer.runner = SimpleNamespace(input_batch=SimpleNamespace(sampling_metadata=_HARNESS._metadata([0.7, 1.3])))
    tokens = _HARNESS._run(proposer, metadata)
    assert tokens.shape == (1, 3)
    if metadata is None:
        assert proposer._last_draft_probs is None
        assert proposer.sampler_inputs == []
    else:
        assert proposer._last_draft_probs.shape == (1, 3, 10)


def test_dynamic_k_returns_only_current_tokens_and_probability_cache(proposer_class, monkeypatch):
    proposer = _HARNESS._proposer(proposer_class)
    metadata = _HARNESS._metadata([0.7, 1.3])
    proposer._dspark_draft_buffer.fill_(-456)
    # Stacking per-step full-vocabulary q would retain a second B*K*V cache.
    monkeypatch.setattr(torch, "stack", lambda *args, **kwargs: pytest.fail("q must write into its final cache"))
    for current_k in (1, 3, 2):
        proposer.num_speculative_tokens = current_k
        proposer.input_ids = torch.zeros(2 * current_k, dtype=torch.long)
        proposer.model.logits = _HARNESS._BASE_LOGITS.repeat(2, current_k, 1)
        tokens = _HARNESS._run(proposer, metadata)
        assert tokens.shape == (2, current_k)
        assert proposer._last_draft_probs.shape == (2, current_k, 10)
        assert not torch.any(tokens == -456)
        torch.testing.assert_close(proposer._last_draft_probs.sum(-1), torch.ones(2, current_k))


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_propose_updates_query_width_and_clears_q_before_k_zero(proposer_class, sample_from_anchor):
    proposer = _HARNESS._proposer(proposer_class)
    proposer.sample_from_anchor = sample_from_anchor
    proposer.hidden_size = 1
    proposer.model.combine_hidden_states = lambda states: states
    proposer.dynamic_spec = SimpleNamespace(num_verify_tokens=None)
    method = _load_method(
        "vllm_ascend/spec_decode/llm_base_proposer.py",
        "AscendSpecDecodeBaseProposer",
        "_propose",
        {
            "torch": torch,
            "BreakableACLGraphWrapper": type("UnusedWrapper", (), {}),
            "_HIDDEN_STATE_DRAFTER_TYPES": (object,),
            "_HiddenStateDrafter": object,
            "cast": lambda kind, value: value,
        },
    )

    class PreparedInputs(Exception):
        pass

    def prepare(**kwargs):
        assert proposer.num_query_per_req == proposer.num_speculative_tokens + (not sample_from_anchor)
        raise PreparedInputs

    proposer.set_inputs_first_pass = prepare
    propose = MethodType(method, proposer)
    for current_k in (1, 0, 3):
        proposer._last_draft_probs = torch.ones(2, 3, 10)
        proposer.dynamic_spec.num_verify_tokens = torch.tensor([3])
        arguments = dict(
            num_speculative_tokens=current_k,
            target_token_ids=torch.zeros(2, dtype=torch.long),
            target_positions=torch.arange(2),
            target_hidden_states=torch.zeros(2, 1),
            next_token_ids=torch.ones(2, dtype=torch.long),
            token_indices_to_sample=torch.arange(2),
            common_attn_metadata=SimpleNamespace(batch_size=lambda: 2),
            target_model_batch_desc=None,
            sampling_metadata=_HARNESS._metadata([0.7, 1.3]),
        )
        if current_k:
            with pytest.raises(PreparedInputs):
                propose(**arguments)
        else:
            assert propose(**arguments).shape == (2, 0)
            assert proposer.dynamic_spec.num_verify_tokens is None
        assert proposer._last_draft_probs is None
        assert proposer.num_query_per_req == current_k + (not sample_from_anchor)


@pytest.mark.parametrize("enabled,temperature", [(True, 1.0), (True, 0.0), (False, 1.0)])
def test_unmapped_padded_logits_cannot_sample_outside_target_vocabulary(proposer_class, enabled, temperature):
    proposer = _HARNESS._proposer(proposer_class, num_reqs=1, mapped=False, enabled=enabled)
    proposer.vllm_config.model_config.get_vocab_size = lambda: 3
    proposer.model.logits[:, :, -1] = 100
    metadata = _HARNESS._metadata([temperature])
    tokens, probabilities = proposer.compute_draft_token_ids(torch.zeros(1, 1), metadata)
    assert tokens.item() < 3
    if enabled and temperature:
        torch.testing.assert_close(probabilities, _HARNESS._BASE_LOGITS[:3].softmax(-1)[None])
    else:
        assert probabilities is None or probabilities.shape[-1] == 3


@pytest.mark.parametrize("fails", [False, True])
def test_zero_k_dummy_runs_backbone_and_keeps_state_even_on_error(fails):
    mode = SimpleNamespace(NONE="none")
    dummy_run = _load_method(
        "vllm_ascend/spec_decode/dspark_proposer.py",
        "AscendDSparkProposer",
        "dummy_run",
        {"torch": torch, "CUDAGraphMode": mode, "set_ascend_forward_context": lambda *args, **kwargs: nullcontext()},
    )
    calls = []

    class Draft:
        def precompute_and_store_context_kv(self, states, positions):
            calls.append(("context", len(states)))

        def __call__(self, input_ids, **kwargs):
            calls.append(("query", len(input_ids)))
            if fails:
                raise RuntimeError("model failed")

    proposer = SimpleNamespace(
        _last_draft_probs=torch.ones(1, 3, 10),
        num_speculative_tokens=0,
        num_query_per_req=0,
        sample_from_anchor=True,
        max_query_tokens=6,
        runner=SimpleNamespace(
            _sync_metadata_across_dp=lambda count, **kwargs: (count, None, None), dynamic_eplb=False
        ),
        use_cuda_graph=False,
        _context_positions_buffer=torch.arange(6),
        hidden_states=torch.zeros(6, 1),
        token_indices_to_sample=torch.zeros(6, dtype=torch.long),
        _clear_dummy_slot_mappings=lambda: None,
        vllm_config=None,
        model=Draft(),
        input_ids=torch.zeros(6, dtype=torch.long),
        _get_positions=torch.arange,
        _runnable=lambda **kwargs: pytest.fail("zero K cannot run the proposal loop"),
    )
    if fails:
        with pytest.raises(RuntimeError, match="model failed"):
            dummy_run(proposer, num_tokens=1, num_reqs=1)
    else:
        dummy_run(proposer, num_tokens=1, num_reqs=1)
    assert calls == [("context", 1), ("query", 1)]
    assert proposer._last_draft_probs is None
    assert proposer.num_speculative_tokens == proposer.num_query_per_req == 0


def test_confidence_scheduler_uses_current_k_after_shrink_and_restore():
    path = _ROOT / "vllm_ascend/spec_decode/utils.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DynamicSpecScheduler")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    scheduler = namespace["DynamicSpecScheduler"](
        method="dspark", method_params={}, max_batch_size=2, num_speculative_tokens=3, device="cpu"
    )
    model = SimpleNamespace(
        markov_embed=lambda tokens: tokens.float().unsqueeze(-1),
        compute_confidence=lambda hidden, embeddings: torch.full((len(hidden),), 0.8),
    )
    for current_k in (1, 3, 2):
        keep_lens = scheduler.update(
            model=model,
            last_hidden_states=torch.zeros(2 * current_k, 1),
            draft_token_ids=torch.ones(2, current_k + 1, dtype=torch.long),
            num_reqs=2,
        )
        assert keep_lens.shape == (2,)
        assert torch.all(keep_lens >= 1) and torch.all(keep_lens <= current_k)
        assert scheduler.num_speculative_tokens == current_k


@pytest.mark.parametrize("sample_from_anchor", [False, True])
def test_query_kernel_respects_current_k_and_leaves_tail_sentinels_on_npu(sample_from_anchor):
    """Even an oversized query block must not write beyond current B*K indices."""
    try:
        importlib.import_module("torch_npu")
    except ImportError as error:
        pytest.skip(f"NPU integration dependency unavailable: {error}")
    if not torch.npu.is_available():
        pytest.skip("NPU integration requires an available device")
    kernel = importlib.import_module(
        "vllm_ascend.ops.triton.spec_decode.utils"
    ).copy_and_expand_dflash_and_dspark_inputs_kernel
    device = "npu:0"
    num_reqs, num_queries = 2, 4
    block_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
    query_positions = torch.empty(num_reqs * num_queries, dtype=torch.int32, device=device)
    for current_k in (1, 3, 1):
        sentinel = -777
        indices = torch.full((num_reqs * 3 + 4,), sentinel, dtype=torch.int32, device=device)
        kernel[(1,)](
            next_token_ids_ptr=torch.tensor([11, 22], dtype=torch.int64, device=device),
            target_positions_ptr=torch.tensor([0, 1], dtype=torch.int32, device=device),
            context_slot_mapping_ptr=torch.tensor([0, 16], dtype=torch.int64, device=device),
            out_input_ids_ptr=torch.empty(num_reqs * num_queries, dtype=torch.int64, device=device),
            out_context_positions_ptr=torch.empty(num_reqs, dtype=torch.int32, device=device),
            out_query_positions_ptr=query_positions,
            out_context_slot_mapping_ptr=torch.empty(num_reqs, dtype=torch.int64, device=device),
            out_query_slot_mapping_ptr=torch.empty(num_reqs * num_queries, dtype=torch.int64, device=device),
            out_token_indices_ptr=indices[: num_reqs * current_k],
            block_table_ptr=block_table,
            block_table_stride=block_table.stride(0),
            query_start_loc_ptr=torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
            seq_lens_ptr=torch.tensor([1, 1], dtype=torch.int32, device=device),
            num_rejected_tokens_ptr=None,
            parallel_drafting_token_id=0,
            block_size=16,
            num_query_per_req=num_queries,
            num_speculative_tokens=current_k,
            total_input_tokens=num_reqs,
            batch_size=num_reqs,
            HAS_NUM_REJECTED=False,
            SAMPLE_FROM_ANCHOR=sample_from_anchor,
        )
        actual = indices.cpu()
        offset = int(not sample_from_anchor)
        expected = torch.arange(current_k).repeat(num_reqs) + torch.arange(num_reqs).repeat_interleave(current_k) * 4
        torch.testing.assert_close(actual[: num_reqs * current_k], (expected + offset).to(torch.int32))
        assert torch.all(actual[num_reqs * current_k :] == sentinel)
