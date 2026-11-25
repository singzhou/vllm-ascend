# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MRV2 GQA DSpark target/draft contract."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.config.attention import AttentionConfig
from vllm.model_executor.model_loader.utils import get_model_cls
from vllm.model_executor.models import ModelRegistry
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

from vllm_ascend.models import register_model
from vllm_ascend.models.qwen3_dspark import (
    AscendQwen3DSparkForCausalLM,
    process_weight,
)
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import (
    AscendDSparkSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.tree.layout import TreeBatch

_HIDDEN = 8


def _spec(vllm_config, draft_hf_config) -> AscendDSparkSpeculator:
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.vllm_config = vllm_config
    spec.draft_model_config = SimpleNamespace(hf_config=draft_hf_config)
    vllm_config.speculative_config = SimpleNamespace(draft_model_config=spec.draft_model_config)
    return spec


def _target() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(embed_tokens=object()),
        lm_head=object(),
        set_dspark_aux_capture_materialized=MagicMock(),
    )


def _gqa_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["Qwen3DSparkModel"],
        model_type="qwen3",
        dspark_aux_hidden_state_format="materialized",
    )


def _vllm_config(*, quarot: bool) -> SimpleNamespace:
    quant_config = None
    if quarot:
        quant_config = SimpleNamespace(
            quant_description={"optional": {"quarot": {"rotation_map": {"global_rotation": "rotation.safetensors"}}}}
        )
    return SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(model="/target"),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )


def _draft():
    draft = AscendQwen3DSparkForCausalLM.__new__(AscendQwen3DSparkForCausalLM)
    torch.nn.Module.__init__(draft)
    draft.config = _gqa_config()
    return draft


@pytest.mark.parametrize("architecture", ["Qwen3DSparkModel", "Qwen3OmniDSparkModel", "DSparkDraftModel"])
def test_registered_draft_class_declares_capabilities(architecture, monkeypatch):
    monkeypatch.setattr(ModelRegistry, "models", ModelRegistry.models.copy())
    register_model()
    config = SimpleNamespace(
        model=f"/test/{architecture}",
        convert_type="none",
        runner_type="generate",
        trust_remote_code=False,
        model_impl="vllm",
        hf_config=SimpleNamespace(architectures=[architecture]),
        registry=ModelRegistry,
        _get_transformers_backend_cls=lambda: "TransformersForCausalLM",
    )
    draft_cls = get_model_cls(config)
    if architecture == "DSparkDraftModel":
        assert not hasattr(draft_cls, "configure_target_aux_hidden_capture")
    else:
        assert draft_cls is AscendQwen3DSparkForCausalLM


def test_draft_without_hook_preserves_target_capture(monkeypatch):
    config = SimpleNamespace(architectures=["DSparkDraftModel"])
    target = _target()
    draft = object()

    def load(*args):
        return draft

    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", load)
    assert _spec(_vllm_config(quarot=True), config).load_draft_model(target, set()) is draft
    target.set_dspark_aux_capture_materialized.assert_not_called()
    assert vars(config) == {"architectures": ["DSparkDraftModel"]}


def test_qwen3_class_selects_materialized_target_capture():
    target = _target()
    _draft().configure_target_aux_hidden_capture(target)
    target.set_dspark_aux_capture_materialized.assert_called_once_with(True)


def test_undeclared_format_uses_materialized_capture():
    target = _target()
    draft = _draft()
    del draft.config.dspark_aux_hidden_state_format
    draft.configure_target_aux_hidden_capture(target)
    target.set_dspark_aux_capture_materialized.assert_called_once_with(True)


def test_wrapped_target_capture():
    target = _target()
    _draft().configure_target_aux_hidden_capture(SimpleNamespace(get_language_model=lambda: target))
    target.set_dspark_aux_capture_materialized.assert_called_once_with(True)


@pytest.mark.parametrize("wrapped", [False, True])
def test_missing_target_setter_preserves_native_behavior(wrapped):
    target = SimpleNamespace()
    model = SimpleNamespace(get_language_model=lambda: target) if wrapped else target
    _draft().configure_target_aux_hidden_capture(model)
    assert vars(target) == {}


def test_configures_capture_after_loading_draft(monkeypatch):
    events = []
    target = _target()
    target.set_dspark_aux_capture_materialized = lambda enabled: events.append(("capture", enabled))
    draft = _draft()
    draft.post_process = MagicMock()
    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.dspark.speculator.set_current_vllm_config", lambda _: nullcontext()
    )

    def _load(self, target_model, target_attn_layer_names):
        events.append(("load", None))
        return draft

    monkeypatch.setattr(DSparkSpeculator, "load_draft_model", _load)
    spec = _spec(_vllm_config(quarot=False), _gqa_config())

    assert spec.load_draft_model(target, set()) is draft
    assert events == [("load", None), ("capture", True)]


def test_propose_publishes_a_branched_tree_alongside_chain_drafts(monkeypatch):
    spec = AscendDSparkSpeculator.__new__(AscendDSparkSpeculator)
    spec.replicated_pcp = False
    spec.enable_adaptive_verification = False
    spec.use_dcp = False
    spec.num_speculative_steps = 3
    spec.max_num_tokens = 8
    spec.sample_indices = torch.arange(6)
    spec._anchor_idx = torch.tensor([0, 3])
    spec._draft_topk = None
    spec.input_buffers = SimpleNamespace(input_ids=torch.zeros(6, dtype=torch.long), positions=torch.arange(6))
    spec.vllm_config = SimpleNamespace(parallel_config=None, attention_config=AttentionConfig())
    spec.requires_non_causal = True
    spec.draft_tokens = torch.tensor([[9, 8, 7], [9, 8, 7]])
    spec.model = SimpleNamespace(
        model=SimpleNamespace(confidence_head=object()),
        compute_draft_logits=lambda hidden: hidden.new_tensor([2.0, 1.0, -1.0, -2.0]).expand(hidden.shape[0], -1),
        markov_embed=lambda tokens: tokens,
        markov_bias=lambda tokens: torch.nn.functional.one_hot(tokens, num_classes=4) * 0.1,
        compute_confidence=lambda hidden, tokens: hidden.new_full(tokens.shape, 0.25),
        map_draft_to_target=lambda tokens: tokens,
    )
    spec.init_tree_drafting(topk=2)

    module = "vllm_ascend.worker.v2.spec_decode.dspark.speculator"
    monkeypatch.setattr(f"{module}.build_attn_metadata_wrapper", nullcontext)
    monkeypatch.setattr(f"{module}.build_attn_metadata_factory", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(DSparkSpeculator, "_sample_sequential", lambda self, num_reqs, head_hidden: None)

    def propose(self, batch, _attn_metadata, _slot_mappings, hidden, *args, **kwargs):
        self._sample_sequential(batch.num_reqs, hidden)
        return self.draft_tokens

    monkeypatch.setattr(DSparkSpeculator, "propose", propose)
    batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["chain", "tree"],
        is_prefilling_np=np.array([False, False]),
        tree_draft_rows=(1,),
        query_start_loc_np=[0, 1, 5],
        num_computed_tokens_np=[4, 5],
        positions=torch.tensor([4, 5, 6, 7, 8]),
    )
    hidden = torch.zeros(6, 2)
    counts = torch.zeros(2, dtype=torch.int32)
    proposal_inputs = ({}, {}, hidden, None, counts, counts, counts, counts, counts.float(), counts)

    tokens = spec.propose(batch, *proposal_inputs)

    assert tokens.tolist() == [[9, 8, 7], [0, 1, 0]]
    assert spec.draft_tokens.tolist() == [[9, 8, 7], [9, 8, 7]]
    assert set(spec.tree_proposals) == {"tree"}
    layout = TreeBatch.from_schedule(batch, spec.tree_proposals, {"tree": tokens[1].tolist()})
    (row,) = layout.tree_rows
    assert row.parents == (0, 0, 0, 1)
    assert row.walk([1, 3, 2, 3]) == ([1, 2], [2])

    monkeypatch.setattr(DSparkSpeculator, "propose", lambda self, *args, **kwargs: self.draft_tokens)
    with pytest.raises(RuntimeError, match="did not produce a draft tree"):
        spec.propose(batch, *proposal_inputs)
    assert spec._tree_draft_rows == ()
    assert spec._draft_trees is None


def test_process_weight_preserves_the_unrotated_projection():
    generator = torch.Generator().manual_seed(7)
    rotation, _ = torch.linalg.qr(torch.randn(_HIDDEN, _HIDDEN, dtype=torch.float64, generator=generator))
    inputs = torch.randn(3, 5, _HIDDEN, dtype=torch.float64, generator=generator)
    weight = torch.randn(_HIDDEN, 5 * _HIDDEN, dtype=torch.float64, generator=generator)

    expected = torch.nn.functional.linear(inputs.reshape(3, -1), weight)
    actual = torch.nn.functional.linear((inputs @ rotation).reshape(3, -1), process_weight(weight, rotation))

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-6)
