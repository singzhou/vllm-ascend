# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for DSpark cache alignment and uncached draft placeholders.

Run with ``pytest --noconftest tests/ut/spec_decode/test_dspark_probability_cache.py``.
The actual runner override is compiled without importing the device runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
_VOCAB_SIZE = 4


class _BaseRunner:
    def _get_spec_decode_draft_probs(self, metadata):
        self.delegated_metadata = metadata
        return "upstream"


@pytest.fixture
def runner_class():
    tree = ast.parse(_SOURCE.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_spec_decode_draft_probs"
    )
    harness = ast.ClassDef(
        name="_Runner",
        bases=[ast.Name(id="_BaseRunner", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), harness],
        type_ignores=[],
    )
    namespace = {"torch": torch, "_BaseRunner": _BaseRunner}
    exec(compile(ast.fix_missing_locations(module), str(_SOURCE), "exec"), namespace)
    return namespace["_Runner"]


def _cache(steps=3, dtype=torch.float32):
    values = torch.arange(1, 2 * steps * _VOCAB_SIZE + 1, dtype=dtype).reshape(2, steps, _VOCAB_SIZE)
    return values / values.sum(-1, keepdim=True)


def _runner(runner_class, cache, req_ids, cached_ids=("A", "B")):
    runner = runner_class()
    runner.speculative_config = SimpleNamespace(method="dspark")
    runner._draft_probs = cache
    runner._draft_prob_req_ids = cached_ids
    runner.input_batch = SimpleNamespace(req_ids=req_ids, vocab_size=_VOCAB_SIZE)
    return runner


def _metadata(counts, tokens):
    return SimpleNamespace(num_draft_tokens=counts, draft_token_ids=torch.tensor(tokens))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_new_placeholder_request_preserves_reordered_cached_requests(runner_class, dtype):
    cache = _cache(dtype=dtype)
    runner = _runner(runner_class, cache, ["B", "new", "prefill", "A"])
    metadata = _metadata([2, 3, 0, 1], [1, 2, -1, -1, -1, 0])
    probabilities = runner._get_spec_decode_draft_probs(metadata)
    expected = torch.cat([cache[1, :2], cache.new_zeros((3, _VOCAB_SIZE)), cache[0, :1]])

    torch.testing.assert_close(probabilities, expected)
    assert probabilities.shape == (6, _VOCAB_SIZE)
    assert probabilities.dtype == cache.dtype and probabilities.device == cache.device
    assert probabilities.is_contiguous()
    # All-zero q is a forced rejection whose recovery distribution is p.
    target = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype)
    torch.testing.assert_close((target - probabilities[2]).clamp_min(0), target)


def test_actual_placeholders_do_not_use_stale_q_for_the_same_request(runner_class):
    cache = _cache()
    original = cache.clone()
    runner = _runner(runner_class, cache, ["A"])
    probabilities = runner._get_spec_decode_draft_probs(_metadata([3], [2, -1, -1]))
    torch.testing.assert_close(probabilities[0], cache[0, 0])
    assert torch.count_nonzero(probabilities[1:]) == 0
    torch.testing.assert_close(cache, original)


def test_uncached_real_candidate_is_rejected_without_discarding_other_q(runner_class):
    cache = _cache()
    runner = _runner(runner_class, cache, ["new", "B"])
    probabilities = runner._get_spec_decode_draft_probs(_metadata([1, 1], [3, 1]))
    assert probabilities[0, 3] == 0
    assert torch.count_nonzero(probabilities[0]) == 0
    torch.testing.assert_close(probabilities[1], cache[1, 0])


def test_short_cache_preserves_known_prefix_and_zero_fills_missing_steps(runner_class):
    cache = _cache(steps=2)
    runner = _runner(runner_class, cache, ["B", "A"])
    probabilities = runner._get_spec_decode_draft_probs(_metadata([1, 4], [1, 0, 1, 2, 3]))
    expected = torch.cat([cache[1, :1], cache[0], cache.new_zeros((2, _VOCAB_SIZE))])
    torch.testing.assert_close(probabilities, expected)
    assert probabilities.shape == (5, _VOCAB_SIZE)


def test_missing_cache_row_does_not_index_past_the_tensor(runner_class):
    cache = _cache()[:1]
    runner = _runner(runner_class, cache, ["B", "A"])
    probabilities = runner._get_spec_decode_draft_probs(_metadata([1, 1], [-1, 1]))
    torch.testing.assert_close(probabilities[1], cache[0, 0])
    assert torch.count_nonzero(probabilities[0]) == 0


@pytest.mark.parametrize("missing_field", ["_draft_probs", "_draft_prob_req_ids"])
def test_absent_cache_preserves_none_semantics(runner_class, missing_field):
    runner = _runner(runner_class, _cache(), ["new"])
    setattr(runner, missing_field, None)
    assert runner._get_spec_decode_draft_probs(_metadata([3], [-1, -1, -1])) is None


def test_incompatible_vocab_width_is_not_passed_to_the_verifier(runner_class):
    runner = _runner(runner_class, _cache()[..., :2], ["A"])
    assert runner._get_spec_decode_draft_probs(_metadata([1], [1])) is None


def test_zero_draft_batch_does_not_allocate_probability_rows(runner_class):
    runner = _runner(runner_class, _cache(), ["A", "new"])
    assert runner._get_spec_decode_draft_probs(_metadata([0, 0], [])) is None


@pytest.mark.parametrize("method", [None, "eagle3", "dflash"])
def test_other_drafter_modes_delegate_to_upstream(runner_class, method):
    runner = _runner(runner_class, _cache(), ["A"])
    runner.speculative_config = None if method is None else SimpleNamespace(method=method)
    metadata = _metadata([1], [2])
    assert runner._get_spec_decode_draft_probs(metadata) == "upstream"
    assert runner.delegated_metadata is metadata
