# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Pooling wrappers around vLLM's native Qwen text backbones."""

import torch
from torch import nn
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_ascend.decision.config import KevConfig, validate_runtime
from vllm_ascend.ops.kev_pointer import KevPointerPooler


class KevDecisionModel(nn.Module):
    is_pooling_model = True
    default_seq_pooling_type = "LAST"
    default_tok_pooling_type = "ALL"
    attn_type = "decoder"
    backbone_type = None

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        validate_runtime(vllm_config)
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_text_config
        settings = KevConfig.from_dict(vllm_config.model_config.hf_config.kev_config)
        hybrid = "linear_attention" in (getattr(self.config, "layer_types", None) or [])
        if hybrid != (self.attn_type == "hybrid"):
            raise ValueError("Kev architecture does not match backbone layer types")
        self.model = self.backbone_type(vllm_config=vllm_config, prefix=f"{prefix}.model" if prefix else "model")
        self.pooler = KevPointerPooler(self.config.hidden_size, settings.head_dim)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights):
        # Exported keys are model.<HF backbone key> and pooler.{q,k}.{weight,bias}.
        # AutoWeightsLoader delegates model.* to the native fusion/TP loader.
        loaded = AutoWeightsLoader(self).load_weights(weights)
        required = {f"pooler.{p}" for p in ("q.weight", "q.bias", "k.weight", "k.bias")}
        if not required.issubset(loaded):
            raise ValueError(f"Missing Kev pointer weights: {sorted(required - loaded)}")
        if any(p.dtype != torch.float32 for p in self.pooler.parameters()):
            raise ValueError("Kev head was cast away from FP32 during loading")
        return loaded


class AscendKevQwen2ForDecision(KevDecisionModel):
    def __init__(self, *, vllm_config, prefix=""):
        from vllm.model_executor.models.qwen2 import Qwen2Model

        self.backbone_type = Qwen2Model
        super().__init__(vllm_config=vllm_config, prefix=prefix)


class AscendKevQwen3ForDecision(KevDecisionModel):
    def __init__(self, *, vllm_config, prefix=""):
        from vllm.model_executor.models.qwen3 import Qwen3Model

        self.backbone_type = Qwen3Model
        super().__init__(vllm_config=vllm_config, prefix=prefix)


class AscendKevQwen35ForDecision(KevDecisionModel, HasInnerState, IsHybrid, SupportsMRoPE):
    attn_type = "hybrid"

    def __init__(self, *, vllm_config, prefix=""):
        from vllm.model_executor.models.qwen3_5 import Qwen3_5Model

        self.backbone_type = Qwen3_5Model
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # Text-only exports retain mrope_section in their RoPE config, so the
        # runner requires this interface on the outer wrapper, not the backbone.
        # For text, T/H/W all use the same absolute positions (delta = 0).
        if mm_features:
            raise ValueError("Kev decision models only support text M-RoPE inputs")
        positions = torch.arange(len(input_tokens), dtype=torch.long, device="cpu")
        return positions.unsqueeze(0).repeat(3, 1), 0

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

        return Qwen3_5ForCausalLMBase.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

        return Qwen3_5ForCausalLMBase.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

        return Qwen3_5ForCausalLMBase.get_mamba_state_copy_func()
