# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real DSpark sampling methods without loading the NPU runtime.

Run on CPU with ``pytest --noconftest tests/ut/spec_decode/test_dspark_probabilistic.py``.
Only the runtime sampler and model are substituted; the production methods,
including the complete merged draft loop, are compiled from their source.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/llm_base_proposer.py"
_KEPT_IDS = torch.tensor([1, 3, 6, 9])
_TARGET_VOCAB_SIZE = 10
_BLOCK_SIZE = 3
_BASE_LOGITS = torch.tensor([2.0, 0.0, 1.0, -1.0])


def _metadata(temperatures):
    temperatures = torch.tensor(temperatures, dtype=torch.float32)
    return SimpleNamespace(
        temperature=temperatures,
        all_greedy=bool((temperatures == 0).all()),
        uniforms=torch.full_like(temperatures, 0.9),
    )


def _sample(logits, metadata):
    """A fixed uniform draw makes categorical outcomes reproducible on CPU."""
    greedy = metadata.temperature == 0
    temperatures = torch.where(greedy, 1.0, metadata.temperature)
    probs = torch.softmax(logits.float() / temperatures[:, None], dim=-1)
    token_ids = (probs.cumsum(-1) < metadata.uniforms[:, None]).sum(-1)
    token_ids = torch.where(greedy, logits.argmax(-1), token_ids)
    return token_ids, probs


class _RuntimeSampler:
    def _sample_from_logits(self, logits, metadata):
        self.sampler_inputs.append(logits.clone())
        if not self._enable_probabilistic_draft_probs:
            return logits.argmax(-1), None
        if metadata.all_greedy:
            return logits.argmax(-1), logits
        return _sample(logits, metadata)


@pytest.fixture
def proposer_class():
    tree = ast.parse(_SOURCE.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSpecDecodeBaseProposer"
    )
    names = {
        "_sample_draft_from_logits",
        "_sample_dspark_draft_from_logits",
        "compute_draft_token_ids",
        "_run_merged_draft",
    }
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert names - {node.name for node in methods} <= {"_sample_dspark_draft_from_logits"}
    harness = ast.ClassDef(
        name="_Proposer",
        bases=[ast.Name(id="_RuntimeSampler", ctx=ast.Load())],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), harness],
        type_ignores=[],
    )
    namespace = {
        "torch": torch,
        "_RuntimeSampler": _RuntimeSampler,
        "_split_draft_outputs": lambda output: (output, output),
        "lmhead_tp_enable": lambda: False,
        "greedy_sample": lambda logits: logits.argmax(-1),
        "logger": SimpleNamespace(warning=lambda *args: None, warning_once=lambda *args: None),
    }
    exec(compile(ast.fix_missing_locations(module), str(_SOURCE), "exec"), namespace)
    return namespace["_Proposer"]


class _Draft:
    def __init__(self, num_reqs, mapped):
        self.logits = _BASE_LOGITS.repeat(num_reqs, _BLOCK_SIZE, 1)
        self.draft_id_to_target_id = _KEPT_IDS - torch.arange(len(_KEPT_IDS)) if mapped else None
        self.markov_inputs = []
        self.lm_head = object()

    def __call__(self, input_ids, **kwargs):
        return torch.arange(len(input_ids), dtype=torch.float32).unsqueeze(-1)

    def compute_draft_logits(self, hidden_states):
        return self.logits.flatten(0, 1)[hidden_states[:, 0].long()].clone()

    def logits_processor(self, head, hidden_states):
        return self.compute_draft_logits(hidden_states)

    def markov_embed(self, token_ids):
        self.markov_inputs.append(token_ids.clone())
        return token_ids.float().unsqueeze(-1)

    def markov_bias(self, embeddings):
        return embeddings * torch.tensor([0.03, -0.02, 0.01, 0.04])

    def map_draft_to_target(self, token_ids):
        return token_ids + self.draft_id_to_target_id[token_ids]


def _proposer(proposer_class, *, num_reqs=2, mapped=True, enabled=True):
    proposer = proposer_class()
    proposer.model = _Draft(num_reqs, mapped)
    proposer.method = "dspark"
    proposer._enable_probabilistic_draft_probs = enabled
    proposer.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_vocab_size=lambda: _TARGET_VOCAB_SIZE if mapped else len(_BASE_LOGITS))
    )
    proposer.num_speculative_tokens = _BLOCK_SIZE
    proposer.parallel_drafting = True
    proposer.dynamic_spec = None
    proposer.runner = None
    proposer._share_mtp_indices = False
    proposer._context_slot_mapping_buffers = None
    proposer._dspark_draft_buffer = torch.zeros(num_reqs, _BLOCK_SIZE + 1, dtype=torch.long)
    proposer._dspark_seed_buffer = torch.arange(num_reqs) + 1
    proposer.input_ids = torch.zeros(num_reqs * _BLOCK_SIZE, dtype=torch.long)
    proposer._get_positions = lambda count: torch.arange(count)
    proposer.build_model_inputs_first_pass = lambda *args: None
    proposer.sampler_inputs = []
    return proposer


def _run(proposer, metadata):
    count = len(proposer.input_ids)
    return proposer._run_merged_draft(
        num_input_tokens=count,
        batch_size=len(proposer._dspark_seed_buffer),
        token_indices_to_sample=torch.arange(count),
        target_positions=torch.arange(count),
        inputs_embeds=None,
        multi_steps_attn_metadata=[],
        num_tokens=count,
        sampling_metadata=metadata,
    )


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("temperatures", [[0.7, 1.3], [0.0, 1.0]])
def test_probabilistic_markov_loop_exports_true_target_space_q(proposer_class, mapped, temperatures):
    proposer = _proposer(proposer_class, mapped=mapped)
    metadata = _metadata(temperatures)
    tokens = _run(proposer, metadata)
    probabilities = proposer._last_draft_probs
    assert probabilities is not None, "probabilistic DSpark must export its actual sampling distribution"
    vocab_size = _TARGET_VOCAB_SIZE if mapped else len(_BASE_LOGITS)
    kept_ids = _KEPT_IDS if mapped else torch.arange(vocab_size)
    expected_tokens, expected_probs = [], []
    previous_ids = proposer._dspark_seed_buffer.clone()

    for step in range(_BLOCK_SIZE):
        torch.testing.assert_close(proposer.model.markov_inputs[step], previous_ids)
        corrected_logits = proposer.model.logits[:, step] + proposer.model.markov_bias(previous_ids.float()[:, None])
        target_logits = torch.full((len(tokens), vocab_size), -torch.inf)
        target_logits[:, kept_ids] = corrected_logits
        torch.testing.assert_close(proposer.sampler_inputs[step], corrected_logits)
        previous_ids, step_probs = _sample(target_logits, metadata)
        expected_tokens.append(previous_ids)
        expected_probs.append(step_probs)

    torch.testing.assert_close(tokens, torch.stack(expected_tokens, dim=1))
    torch.testing.assert_close(probabilities, torch.stack(expected_probs, dim=1))
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(len(tokens), _BLOCK_SIZE))
    assert probabilities.shape == (len(tokens), _BLOCK_SIZE, vocab_size)
    assert probabilities.dtype == torch.float32 and probabilities.is_contiguous()
    # This draw deliberately samples outside the argmax on the random row.
    assert tokens[-1, 0] != kept_ids[_BASE_LOGITS.argmax()]
    assert int(torch.count_nonzero(probabilities[-1, 0])) == len(_BASE_LOGITS)
    if mapped:
        removed_ids = [index for index in range(vocab_size) if index not in _KEPT_IDS.tolist()]
        assert torch.count_nonzero(probabilities[:, :, removed_ids]) == 0

    # A later profile round must clear the previous round's q cache.
    _run(proposer, None)
    assert proposer._last_draft_probs is None


@pytest.mark.parametrize("mapped", [False, True])
@pytest.mark.parametrize("mode", ["disabled", "all_greedy", "profile"])
def test_greedy_and_profile_paths_preserve_target_ids(proposer_class, mapped, mode):
    proposer = _proposer(proposer_class, mapped=mapped, enabled=mode != "disabled")
    metadata = None if mode == "profile" else _metadata([0.0, 0.0] if mode == "all_greedy" else [1.0, 1.0])
    tokens = _run(proposer, metadata)
    previous_ids = proposer._dspark_seed_buffer.clone()
    for step in range(_BLOCK_SIZE):
        torch.testing.assert_close(proposer.model.markov_inputs[step], previous_ids)
        logits = proposer.model.logits[:, step] + proposer.model.markov_bias(previous_ids.float()[:, None])
        previous_ids = logits.argmax(-1)
        if mapped:
            previous_ids = _KEPT_IDS[previous_ids]
        torch.testing.assert_close(tokens[:, step], previous_ids)
    assert proposer._last_draft_probs is None


def test_compute_draft_token_ids_uses_dspark_probability_mapping(proposer_class):
    proposer = _proposer(proposer_class, num_reqs=1)
    metadata = _metadata([1.0])
    tokens, probabilities = proposer.compute_draft_token_ids(torch.zeros(1, 1), metadata)
    expected = torch.zeros(1, _TARGET_VOCAB_SIZE)
    expected[:, _KEPT_IDS] = _BASE_LOGITS.softmax(-1)
    torch.testing.assert_close(probabilities, expected)
    assert tokens.tolist() == [6]


def test_lmhead_tp_padding_is_trimmed_before_mapped_sampling(proposer_class, monkeypatch):
    metadata = _metadata([0.7, 1.3])
    reference = _proposer(proposer_class)
    expected_tokens = _run(reference, metadata)

    proposer = _proposer(proposer_class)
    proposer.num_query_per_req = _BLOCK_SIZE
    padded_count = 3 * _BLOCK_SIZE
    actual_count = 2 * _BLOCK_SIZE
    proposer.input_ids = torch.zeros(padded_count, dtype=torch.long)
    namespace = proposer_class._run_merged_draft.__globals__
    monkeypatch.setitem(namespace, "lmhead_tp_enable", lambda: True)
    monkeypatch.setitem(namespace, "nn", torch.nn)
    tokens = proposer._run_merged_draft(
        num_input_tokens=padded_count,
        batch_size=2,
        token_indices_to_sample=torch.arange(actual_count),
        target_positions=torch.arange(padded_count),
        inputs_embeds=None,
        multi_steps_attn_metadata=[],
        num_tokens=padded_count,
        sampling_metadata=metadata,
    )

    torch.testing.assert_close(tokens, expected_tokens)
    torch.testing.assert_close(proposer._last_draft_probs, reference._last_draft_probs)
    assert proposer._last_draft_probs.shape == (2, _BLOCK_SIZE, _TARGET_VOCAB_SIZE)
    assert len(proposer.sampler_inputs) == _BLOCK_SIZE
    assert all(logits.shape == (2, len(_BASE_LOGITS)) for logits in proposer.sampler_inputs)


@pytest.mark.parametrize("method", ["eagle3", "dflash"])
def test_other_mapped_drafters_keep_the_existing_greedy_path(proposer_class, method):
    proposer = _proposer(proposer_class, num_reqs=1)
    proposer.method = method
    tokens, probabilities = proposer.compute_draft_token_ids(torch.zeros(1, 1), _metadata([1.0]))
    assert tokens.tolist() == [1]
    assert probabilities is None
    assert proposer.sampler_inputs == []


def test_mapped_draft_with_real_npu_sampler(proposer_class, monkeypatch):
    """Optional hardware check of the production helper and real vLLM sampler."""
    for module_name in ("torch_npu", "vllm"):
        try:
            importlib.import_module(module_name)
        except ImportError as error:
            pytest.skip(f"NPU integration dependency unavailable: {error}")
    if not torch.npu.is_available():
        pytest.skip("NPU integration requires an available device")
    upstream = importlib.import_module("vllm.v1.spec_decode.llm_base_proposer")
    monkeypatch.setattr(_RuntimeSampler, "_sample_from_logits", upstream.SpecDecodeBaseProposer._sample_from_logits)
    proposer = _proposer(proposer_class, num_reqs=1)
    proposer.use_fp64_gumbel = False
    proposer.model.draft_id_to_target_id = proposer.model.draft_id_to_target_id.to("npu:0")
    sample_count = 4096
    logits = _BASE_LOGITS.repeat(sample_count, 1).to("npu:0")
    metadata = _metadata([1.0] * sample_count)
    metadata.temperature = metadata.temperature.to("npu:0")
    metadata.all_random = True

    tokens, probabilities = proposer._sample_dspark_draft_from_logits(logits, metadata)
    expected = torch.zeros(_TARGET_VOCAB_SIZE)
    expected[_KEPT_IDS] = _BASE_LOGITS.softmax(-1)
    torch.testing.assert_close(
        probabilities,
        expected.expand(sample_count, -1).to("npu:0"),
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(sample_count).to("npu:0"))
    removed_ids = [index for index in range(_TARGET_VOCAB_SIZE) if index not in _KEPT_IDS.tolist()]
    assert torch.count_nonzero(probabilities.cpu()[:, removed_ids]) == 0
    frequencies = torch.bincount(tokens.cpu(), minlength=_TARGET_VOCAB_SIZE).float() / sample_count
    torch.testing.assert_close(frequencies, expected, atol=0.04, rtol=0)

    mixed = _metadata([0.0, 1.0])
    mixed.temperature = mixed.temperature.to("npu:0")
    mixed.all_random = False
    tokens, probabilities = proposer._sample_dspark_draft_from_logits(logits[:2], mixed)
    assert tokens[0].cpu().item() == _KEPT_IDS[_BASE_LOGITS.argmax()].item()
    torch.testing.assert_close(probabilities.cpu(), expected.expand(2, -1), atol=1e-6, rtol=1e-5)

    # The merged loop writes q into a strided step view of its final cache.
    cache = torch.full((2, 3, _TARGET_VOCAB_SIZE), -7.0, device="npu:0")
    proposer._sample_dspark_draft_from_logits(logits[:2], mixed, probs_out=cache[:, 0])
    previous_probs = cache[:, 0].clone()
    step_out = cache[:, 1]
    assert not step_out.is_contiguous()
    tokens, probabilities = proposer._sample_dspark_draft_from_logits(logits[:2], mixed, probs_out=step_out)
    assert probabilities.data_ptr() == step_out.data_ptr()
    torch.testing.assert_close(probabilities.cpu(), expected.expand(2, -1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cache[:, 0], previous_probs)
    assert torch.all(cache[:, 2] == -7)

    greedy = _metadata([0.0, 0.0])
    greedy.temperature = greedy.temperature.to("npu:0")
    greedy.all_random = False
    tokens, probabilities = proposer._sample_dspark_draft_from_logits(logits[:2], greedy)
    assert tokens.cpu().tolist() == [1, 1]
    assert probabilities is None
