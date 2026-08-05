import torch
import torch.nn.functional as F
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)


def _copy_context_edge_rows(dst: torch.Tensor | None, layer_idx: int, src: torch.Tensor) -> None:
    """Copy the first/last context row into a persistent DSpark probe."""
    if dst is None:
        return
    src_2d = src.reshape(src.shape[0], -1)
    width = min(dst.shape[-1], src_2d.shape[-1])
    dst[layer_idx, 0, :width].copy_(src_2d[0, :width])
    dst[layer_idx, 1, :width].copy_(src_2d[-1, :width])


def _copy_cached_context_edge_rows(
    dst: torch.Tensor | None,
    layer_idx: int,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Read back first/last context slots after reshape-and-cache."""
    if dst is None:
        return
    cache_2d = cache.reshape(cache.shape[0] * cache.shape[1], -1)
    edge_slots = torch.cat((slot_mapping[:1], slot_mapping[-1:])).to(torch.long)
    edge_rows = torch.index_select(cache_2d, 0, edge_slots)
    width = min(dst.shape[-1], edge_rows.shape[-1])
    dst[layer_idx, :, :width].copy_(edge_rows[:, :width])


def precompute_and_store_context_kv(
    self,
    context_states: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
) -> None:
    if not hasattr(self, "_num_attn_layers"):
        self._build_fused_kv_buffers()

    num_ctx = context_states.shape[0]
    L = self._num_attn_layers
    kv = self._kv_size
    hd = self._head_dim
    nkv = self._num_kv_heads

    # --- Fused KV projection (one GEMM for all layers) ---
    normed_context_states = self.hidden_norm(context_states)
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

    all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
    projected_k_probes = getattr(self, "_dspark_diag_context_projected_k_probes", None)
    projected_v_probes = getattr(self, "_dspark_diag_context_projected_v_probes", None)
    for i in range(L):
        _copy_context_edge_rows(projected_k_probes, i, all_k_final[i])
        _copy_context_edge_rows(projected_v_probes, i, all_v[i])

    if context_slot_mapping is None:
        return

    # --- Per-layer cache insert ---
    per_layer = isinstance(context_slot_mapping, (list, tuple))
    cached_k_probes = getattr(self, "_dspark_diag_context_cached_k_probes", None)
    cached_v_probes = getattr(self, "_dspark_diag_context_cached_v_probes", None)
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
        _copy_cached_context_edge_rows(cached_k_probes, i, kv_cache[0], slot_mapping)
        _copy_cached_context_edge_rows(cached_v_probes, i, kv_cache[1], slot_mapping)


DFlashQwen3Model.precompute_and_store_context_kv = precompute_and_store_context_kv

_orig_read_mask_embedding = DFlashQwen3ForCausalLM._read_mask_embedding


def _patched_read_mask_embedding(self):
    try:
        return _orig_read_mask_embedding(self)
    except Exception:
        return None


DFlashQwen3ForCausalLM._read_mask_embedding = _patched_read_mask_embedding
