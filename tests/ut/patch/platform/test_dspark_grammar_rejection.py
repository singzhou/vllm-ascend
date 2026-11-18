# SPDX-License-Identifier: Apache-2.0
"""Run with --noconftest; also exercise vLLM's manager when its source is available."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_PATCH = Path(__file__).resolve().parents[4] / "vllm_ascend/patch/platform/patch_dspark_structured_output.py"


class _Grammar:
    def __init__(self):
        self.history = []

    def allowed(self, history=None):
        history = self.history if history is None else history
        return ({0, 1}, {1}, {2}, set())[min(len(history), 3)]

    def validate_tokens(self, tokens):
        history = self.history.copy()
        validated = []
        for token in tokens:
            if token not in self.allowed(history):
                break
            validated.append(token)
            history.append(token)
        return validated

    def is_terminated(self):
        return len(self.history) == 3

    def accept_tokens(self, request_id, tokens):
        for token in tokens:
            assert token in self.allowed(), "mask generation must not advance through the rejected proposal"
            self.history.append(token)
        return True

    def rollback(self, count):
        del self.history[-count:]

    def fill_bitmask(self, bitmask, index):
        bitmask[index, 0] = sum(1 << token for token in self.allowed())


def _validate_tokens(self, request, tokens):
    if not request.use_structured_output:
        return tokens
    tokens = tokens[: next((i for i, token in enumerate(tokens) if token < 0), len(tokens))]
    return request.structured_output_request.grammar.validate_tokens(tokens)


def _grammar_bitmask(self, requests, request_ids, tokens_by_request):
    self.mask_input = tokens_by_request
    masks = []
    for req_id in request_ids:
        grammar = requests[req_id].structured_output_request.grammar
        advancements = 0
        padding = False
        for token in [*tokens_by_request.get(req_id, []), -1]:
            masks.append([-1 if padding or grammar.is_terminated() else sum(1 << t for t in grammar.allowed())])
            if token < 0:
                padding = True
            elif not padding:
                grammar.accept_tokens(req_id, [token])
                advancements += 1
        if advancements:
            grammar.rollback(advancements)
    return torch.tensor(masks, dtype=torch.int32).numpy()


@pytest.fixture(params=["contract", "upstream"])
def manager_class(request):
    if request.param == "contract":
        return type("Manager", (), {"validate_tokens": _validate_tokens, "grammar_bitmask": _grammar_bitmask})

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        pytest.skip("vLLM source unavailable; run with the matching vLLM on PYTHONPATH")
    source = Path(spec.origin).parent / "v1/structured_output/__init__.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StructuredOutputManager")
    names = {"validate_tokens", "grammar_bitmask", "_fill_bitmasks"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    harness = ast.ClassDef(name="Manager", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), harness],
        type_ignores=[],
    )
    namespace = {
        "TYPE_CHECKING": False,
        "strip_speculative_padding": lambda tokens: tokens[
            : next((i for i, token in enumerate(tokens) if token < 0), len(tokens))
        ],
        "logger": SimpleNamespace(error=lambda *args: pytest.fail(f"unexpected grammar failure: {args}")),
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["Manager"]


@pytest.fixture
def manager(manager_class, monkeypatch):
    for name in ("vllm", "vllm.v1", "vllm.v1.structured_output"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["vllm.v1.structured_output"].StructuredOutputManager = manager_class
    spec = importlib.util.spec_from_file_location("_dspark_grammar_patch", _PATCH)
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    manager = manager_class()
    manager.vllm_config = SimpleNamespace(
        use_v2_model_runner=False,
        speculative_config=SimpleNamespace(
            method="dspark", draft_sample_method="probabilistic", rejection_sample_method="standard"
        ),
        num_speculative_tokens=3,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        model_config=SimpleNamespace(is_diffusion=False),
    )
    manager._get_constraint_start = lambda request, tokens: 0
    manager._grammar_bitmask = None
    manager._full_mask = -1
    manager.fill_bitmask_parallel_threshold = 100
    manager.backend = SimpleNamespace(allocate_token_bitmask=lambda count: torch.empty(count, 1, dtype=torch.int32))
    return manager


def _request():
    return SimpleNamespace(use_structured_output=True, structured_output_request=SimpleNamespace(grammar=_Grammar()))


@pytest.mark.parametrize(
    "tokens,expected",
    [
        ([], []),
        ([2, 1, 0], [2]),
        ([0, 0, 1], [0, 0]),
        ([0, 1, 2], [0, 1, 2]),
        ([0, 1, 2, 0], [0, 1, 2, 0]),
        ([0, -1, -1], [0]),
        ([-1, -1], []),
        ([0, -1, -1], [0]),
        ([0, 0, -1], [0, 0]),
    ],
)
def test_keeps_first_rejected_position_and_does_not_advance_grammar(manager, tokens, expected):
    request = _request()
    original = tokens.copy()
    validated = manager.validate_tokens(request, tokens)
    assert validated == expected
    assert len(validated) <= len(tokens)
    masks = manager.grammar_bitmask({"r": request}, ["r"], {"r": validated})
    assert masks.shape[0] == len(validated) + 1
    assert int(masks[0, 0]) == 3
    if len(validated) > 1:
        assert int(masks[1, 0]) == 2
    assert request.structured_output_request.grammar.history == []
    assert tokens == original


def test_async_padding_changes_only_mask_copy(manager):
    request = _request()
    scheduled = {"r": [0, 0, -1], "unstructured": [2]}
    original = {key: value.copy() for key, value in scheduled.items()}
    masks = manager.grammar_bitmask({"r": request}, ["r"], scheduled)
    assert int(masks[1, 0]) == 2
    assert int(masks[2, 0]) == -1
    assert scheduled == original
    assert request.structured_output_request.grammar.history == []


@pytest.mark.parametrize(
    "field,value",
    [("method", "eagle3"), ("draft_sample_method", "greedy"), ("rejection_sample_method", "block")],
)
def test_other_drafting_modes_keep_upstream_validation(manager, field, value):
    setattr(manager.vllm_config.speculative_config, field, value)
    assert manager.validate_tokens(_request(), [2, 0]) == []


def test_v2_and_unstructured_requests_keep_upstream_behavior(manager):
    manager.vllm_config.use_v2_model_runner = True
    assert manager.validate_tokens(_request(), [2, 0]) == []
    manager.vllm_config.use_v2_model_runner = False
    request = _request()
    request.use_structured_output = False
    tokens = [2, 0]
    assert manager.validate_tokens(request, tokens) is tokens


def test_retained_illegal_draw_uses_residual_distribution(manager):
    request = _request()
    p = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
    q = torch.tensor([0.4, 0.1, 0.5], dtype=torch.float64)
    residual = (p - q).clamp_min(0)
    residual /= residual.sum()
    result = torch.zeros_like(p)
    truncated_result = torch.zeros_like(p)
    for token, proposal_mass in enumerate(q):
        retained = manager.validate_tokens(request, [token])
        assert retained == [token]
        masks = manager.grammar_bitmask({"r": request}, ["r"], {"r": retained})
        target = p * torch.tensor([bool(int(masks[0, 0]) & (1 << i)) for i in range(len(p))])
        target /= target.sum()
        acceptance = min(1.0, float(target[token] / q[token]))
        result[token] += proposal_mass * acceptance
        result += proposal_mass * (1 - acceptance) * residual
        if token == 2:
            truncated_result += proposal_mass * p
        else:
            truncated_result[token] += proposal_mass
    torch.testing.assert_close(result, p)
    torch.testing.assert_close(truncated_result, torch.tensor([0.65, 0.35, 0.0], dtype=torch.float64))
