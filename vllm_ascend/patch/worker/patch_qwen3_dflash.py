import torch
import torch.nn.functional as F
import torch_npu
import vllm.model_executor.models.qwen3_dflash as dflash_model
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkModel
from vllm.model_executor.models.utils import WeightsMapper

from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import maybe_trans_nz

_original_get_draft_quant_config = dflash_model.get_draft_quant_config
_original_build_context_kv_buffers = DFlashQwen3Model._build_context_kv_buffers


def _get_draft_quant_config(vllm_config: VllmConfig):
    speculative_config = vllm_config.speculative_config
    draft_config = speculative_config.draft_model_config
    quant_config = vllm_config.quant_config
    if (
        not vllm_config.use_v2_model_runner
        or speculative_config.method != "dspark"
        or draft_config.hf_config.model_type != "qwen3"
        or draft_config.model == vllm_config.model_config.model
        or draft_config.quantization != AscendModelSlimConfig.get_name()
        or not isinstance(quant_config, AscendModelSlimConfig)
    ):
        return _original_get_draft_quant_config(vllm_config)

    # The draft loader has already attached the packed QKV and gate/up
    # mappings to this config. Keep them when the backbone asks for it again.
    # ModelSlim labels the checkpoint keys, while the DSpark layers are named
    # after the last target layer at runtime.
    start_layer = vllm_config.model_config.get_total_num_hidden_layers()
    prefixes = {
        f"layers.{index}.": f"model.layers.{start_layer + index}."
        for index in range(draft_config.hf_config.num_hidden_layers)
    }
    prefixes.update({"fc.": "model.fc.", "markov_head.": "model.markov_head."})
    quant_config.apply_vllm_mapper(WeightsMapper(orig_to_new_prefix=prefixes))
    return quant_config


def _build_dspark_context_kv_buffers(
    self: Qwen3DSparkModel, layers_attn: list[torch.nn.Module], has_bias: bool
) -> None:
    dynamic_projections = [
        isinstance(getattr(attn.qkv_proj.quant_method, "quant_method", None), AscendW8A8DynamicLinearMethod)
        for attn in layers_attn
    ]
    if any(dynamic_projections) and not all(dynamic_projections):
        # A checkpoint can keep some QKV layers in floating point. Let each
        # layer apply its own method when a single GEMM cannot serve them all.
        self._fused_kv_weight = None
        return

    _original_build_context_kv_buffers(self, layers_attn, has_bias)
    if all(dynamic_projections):
        # Weight loading calls this before the individual QKV projections are
        # packed. Preserve their TP-local K/V rows and pack one fused matrix.
        self._fused_kv_weight = maybe_trans_nz(self._fused_kv_weight.t().contiguous())
        self._fused_kv_weight_scale = torch.cat(
            [attn.qkv_proj.weight_scale[attn.q_size :].flatten() for attn in layers_attn]
        )


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | None = None,
) -> None:
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    normed_context_states = self.hidden_norm(context_states)
    if self._fused_kv_weight is None:
        all_kv_flat = torch.cat(
            [
                layer.self_attn.qkv_proj(normed_context_states)[0][..., layer.self_attn.q_size :]
                for layer in self.layers
            ],
            dim=-1,
        )
    elif (weight_scale := getattr(self, "_fused_kv_weight_scale", None)) is not None:
        quantized, per_token_scale = torch_npu.npu_dynamic_quant(normed_context_states)
        all_kv_flat = torch_npu.npu_quant_matmul(
            quantized,
            self._fused_kv_weight,
            weight_scale,
            pertoken_scale=per_token_scale,
            bias=self._fused_kv_bias,
            output_dtype=normed_context_states.dtype,
        )
    else:
        all_kv_flat = F.linear(normed_context_states, self._fused_kv_weight, self._fused_kv_bias)
    # Single contiguous copy that separates K/V and transposes to
    # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
    # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
    all_kv = all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
    all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
    all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

    # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
    all_k_normed = torch.empty_like(all_k)
    for i in range(L):
        k_norm_layer = self.layers[i].self_attn.k_norm
        all_k_normed[i] = k_norm_layer(all_k[i])

    # --- Fused RoPE across all layers ---
    # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
    # In-place RoPE: pass K as the "query" arg with key=None.
    all_k_flat = all_k_normed.view(L * num_ctx, kv)
    positions_repeated = context_positions.repeat(L)
    tmpv = all_k_flat.clone()
    self.layers[0].self_attn.rotary_emb(positions_repeated, all_k_flat, tmpv)

    if context_slot_mapping is None:
        return

    # --- Per-layer cache insert ---
    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
    per_layer = isinstance(context_slot_mapping, (list, tuple))
    for i in range(L):
        slot_mapping = context_slot_mapping[i] if per_layer else context_slot_mapping
        if slot_mapping is None:
            continue
        attn = self._attn_layers[i]
        kv_cache = attn.kv_cache
        attn.impl.do_kv_cache_update(
            attn,
            all_k_final[i],
            all_v[i],
            kv_cache,
            slot_mapping,
        )


DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv
Qwen3DSparkModel._build_context_kv_buffers = _build_dspark_context_kv_buffers
dflash_model.get_draft_quant_config = _get_draft_quant_config

_orig_read_mask_embedding = DFlashQwen3ForCausalLM._read_mask_embedding


def _patched_read_mask_embedding(self):
    try:
        return _orig_read_mask_embedding(self)
    except Exception:
        return None


DFlashQwen3ForCausalLM._read_mask_embedding = _patched_read_mask_embedding
