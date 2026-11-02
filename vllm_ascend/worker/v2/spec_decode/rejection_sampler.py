# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 0.30 verification with per-request logprob token IDs."""

import torch
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    RejectionSampler,
    _flatten_sampled_kernel,
    _iter_request_chunks,
)


class AscendRejectionSampler(RejectionSampler):
    def _verify_in_chunks(
        self, logits, input_batch, draft_logits, draft_sampled, pos, max_chunk_logits, max_num_logprobs
    ):
        state = self.sampler.logprob_token_ids_state
        max_custom = state.max_num_token_ids(input_batch.idx_mapping_np)
        if not max_custom:
            return super()._verify_in_chunks(
                logits, input_batch, draft_logits, draft_sampled, pos, max_chunk_logits, max_num_logprobs
            )

        # Keep the same column count across chunks, even for chunks containing
        # only ordinary top-k requests. The global width is needed by cat().
        num_topk = max(0, max_num_logprobs)
        cu_cpu = input_batch.cu_num_logits_np
        chunks = [(0, input_batch.num_reqs)]
        if logits.shape[0] > max_chunk_logits:
            assert not self.enable_adaptive_verification
            chunks = _iter_request_chunks(cu_cpu, max_chunk_logits)

        sampled_chunks, count_chunks, score_chunks = [], [], []
        for start, end in chunks:
            lo, hi = int(cu_cpu[start]), int(cu_cpu[end])
            chunk_logits = logits[lo:hi]
            mapping = input_batch.expanded_idx_mapping[lo:hi]
            cu = input_batch.cu_num_logits[start : end + 1] - lo
            processed, sampled, counts = self._verify(
                chunk_logits,
                draft_logits,
                draft_sampled[lo:hi],
                pos[lo:hi],
                cu,
                input_batch.idx_mapping[start:end],
                input_batch.idx_mapping_np[start:end],
                mapping,
                input_batch.expanded_local_pos[lo:hi],
            )
            flat = sampled.new_zeros(chunk_logits.shape[0])
            _flatten_sampled_kernel[(end - start,)](flat, sampled, sampled.stride(0), counts, cu, num_warps=1)
            boundaries = None
            if chunk_logits.shape[0] != end - start:
                # Adaptive verification compacts the device layout only.
                boundaries = cu if self.enable_adaptive_verification else (cu_cpu[start : end + 1] - lo).tolist()
            mode = self.sampler.logprobs_mode
            scores = compute_topk_scores(
                processed if mode in PROCESSED_LOGPROBS_MODES else chunk_logits,
                num_topk,
                flat,
                boundaries,
                logprob_token_ids_state=state,
                expanded_idx_mapping=mapping,
                max_per_req_token_ids=max_custom,
                logits_mode=mode in ("raw_logits", "processed_logits"),
            )
            sampled_chunks.append(sampled)
            count_chunks.append(counts)
            score_chunks.append(scores)
            del processed

        if len(score_chunks) == 1:
            return sampled_chunks[0], count_chunks[0], score_chunks[0]
        boundaries = cu_cpu.tolist() if logits.shape[0] != input_batch.num_reqs else None
        return torch.cat(sampled_chunks), torch.cat(count_chunks), LogprobsTensors.cat(score_chunks, boundaries)
