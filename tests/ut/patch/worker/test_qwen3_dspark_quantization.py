# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import vllm.envs as vllm_envs

from vllm_ascend.patch.worker import patch_qwen3_dflash
from vllm_ascend.platform import _fix_incompatible_config
from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig, get_quant_type_for_layer
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod


@pytest.mark.parametrize(
    ("draft_model", "use_v2_model_runner", "expected"),
    [("draft", True, "ascend"), ("target", True, None), ("draft", False, None)],
)
def test_platform_retains_standalone_dspark_quantization(monkeypatch, draft_model, use_v2_model_runner, expected):
    monkeypatch.setattr(vllm_envs, "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", 3000)
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="target", quantization=None),
        speculative_config=SimpleNamespace(
            method="dspark",
            quantization="ascend",
            draft_model_config=SimpleNamespace(model=draft_model, hf_config=SimpleNamespace(model_type="qwen3")),
        ),
        use_v2_model_runner=use_v2_model_runner,
        additional_config=None,
        parallel_config=SimpleNamespace(enable_eplb=False),
        cache_config=None,
        observability_config=None,
        scheduler_config=None,
        kv_transfer_config=None,
        attention_config=None,
        compilation_config=None,
    )

    _fix_incompatible_config(config)

    assert config.speculative_config.quantization == expected
    assert config.model_config.quantization is None


def test_dspark_reuses_draft_quantization_with_runtime_layer_names():
    quant_config = AscendModelSlimConfig(
        {
            "layers.0.self_attn.q_proj.weight": "W8A8_DYNAMIC",
            "layers.0.self_attn.k_proj.weight": "W8A8_DYNAMIC",
            "layers.0.self_attn.v_proj.weight": "W8A8_DYNAMIC",
            "layers.1.self_attn.o_proj.weight": "FLOAT",
            "fc.weight": "W8A8_DYNAMIC",
            "markov_head.markov_w2.weight": "FLOAT",
        }
    )
    quant_config.packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
    config = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(model="target", get_total_num_hidden_layers=lambda: 64),
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=SimpleNamespace(
                model="draft",
                quantization="ascend",
                hf_config=SimpleNamespace(model_type="qwen3", num_hidden_layers=2),
            ),
        ),
        quant_config=quant_config,
    )

    assert patch_qwen3_dflash._get_draft_quant_config(config) is quant_config
    description = quant_config.quant_description
    assert (
        get_quant_type_for_layer(description, "model.layers.64.self_attn.qkv_proj", quant_config.packed_modules_mapping)
        == "W8A8_DYNAMIC"
    )
    assert description["model.layers.65.self_attn.o_proj.weight"] == "FLOAT"
    assert description["model.fc.weight"] == "W8A8_DYNAMIC"
    assert description["model.markov_head.markov_w2.weight"] == "FLOAT"


@pytest.mark.parametrize("second_projection_is_quantized", [True, False])
def test_dspark_context_kv_buffers_preserve_tp_local_scales(monkeypatch, second_projection_is_quantized):
    monkeypatch.setattr(patch_qwen3_dflash, "maybe_trans_nz", lambda weight: weight)
    model = SimpleNamespace(hidden_norm=SimpleNamespace(weight=torch.ones(2)))
    layers = []
    for index, is_quantized in enumerate((True, second_projection_is_quantized)):
        scheme = AscendW8A8DynamicLinearMethod() if is_quantized else None
        weight_dtype = torch.int8 if is_quantized else torch.bfloat16
        qkv_proj = SimpleNamespace(
            weight=torch.arange(index * 6, (index + 1) * 6, dtype=weight_dtype).view(3, 2),
            weight_scale=torch.arange(index * 3, (index + 1) * 3, dtype=torch.float32).view(3, 1),
            quant_method=SimpleNamespace(quant_method=scheme),
        )
        layers.append(SimpleNamespace(qkv_proj=qkv_proj, q_size=1, k_norm=SimpleNamespace(weight=torch.ones(2))))

    patch_qwen3_dflash._build_dspark_context_kv_buffers(model, layers, has_bias=False)

    if second_projection_is_quantized:
        expected_weight = torch.tensor([[2, 4, 8, 10], [3, 5, 9, 11]], dtype=torch.int8)
        torch.testing.assert_close(model._fused_kv_weight, expected_weight)
        torch.testing.assert_close(model._fused_kv_weight_scale, torch.tensor([1.0, 2.0, 4.0, 5.0]))
    else:
        assert model._fused_kv_weight is None
