// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_mega_chunk_gdn(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& g, const at::Tensor& beta, const at::Tensor& mask_lower,
    const at::Tensor& mask_full, const at::Tensor& minus_identity,
    const at::Tensor& cu_seqlens, const at::Tensor& initial_state, int64_t num_matrices)
{
    constexpr int64_t chunk_size = 128;
    TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() && v.dim() == 4,
                "MegaChunkGdn expects packed Q/K/V tensors");
    const auto tokens = v.size(1);
    const auto heads = v.size(2);
    const auto dim = v.size(3);
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2 && num_matrices > 0,
                "MegaChunkGdn requires sequence starts and a positive matrix count");
    const auto sequences = cu_seqlens.numel() - 1;
    TORCH_CHECK(initial_state.sizes() == at::IntArrayRef({sequences, heads, dim, dim}),
                "MegaChunkGdn initial state must match the sequence and value head counts");

    auto out = at::empty_like(v);
    auto g_sum = at::empty(g.sizes(), g.options());
    auto g_transposed = at::empty({heads, tokens}, g.options());
    auto beta_transposed = at::empty({heads, tokens}, beta.options());
    auto a = at::zeros({1, tokens, heads, chunk_size}, beta.options());
    auto a_inv_f32 = at::zeros(a.sizes(), g.options());
    auto a_inv = at::zeros(a.sizes(), beta.options());
    auto w = at::empty_like(v);
    auto u = at::empty_like(v);
    auto h = at::zeros({num_matrices, dim, dim}, v.options());
    auto v_new = at::empty_like(v);
    auto final_state = at::zeros({sequences * heads, dim, dim}, v.options());

    bool has_initial_state = true;
    EXEC_NPU_CMD(aclnnMegaChunkGdn, q, k, v, g, beta, mask_lower, mask_full, minus_identity,
                 cu_seqlens, initial_state, num_matrices, has_initial_state, out, g_sum, g_transposed,
                 beta_transposed, a, a_inv_f32, a_inv, w, u, h, v_new, final_state);
    return {out, final_state.view(initial_state.sizes())};
}

}  // namespace vllm_ascend
