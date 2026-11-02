from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.sample.logprob import compute_topk_logprobs


@pytest.mark.parametrize(
    "batch_size,vocab_size,num_logprobs",
    [
        (48, 1024, 5),
        (96, 1024, 0),
        (24, 1519, 1),
        (1, 320, 10),
    ],
)
def test_compute_topk_logprobs(batch_size, vocab_size, num_logprobs):
    """Test compute_topk_logprobs for correctness of IDs, logprobs, and ranks.
    Args:
        batch_size: Number of sequences in the batch
        vocab_size: Size of the vocabulary
        num_logprobs: Number of top-k logprobs to return (excluding the sampled token)
    """
    init_device_properties_triton()
    # ========== 1. Setup test data ==========
    torch.manual_seed(42)
    device = "npu"

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
    sampled_token_ids = torch.randint(0, vocab_size, (batch_size,), device=device, dtype=torch.int64)

    # ========== 2. Execute Triton implementation ==========
    triton_output = compute_topk_logprobs(logits, num_logprobs, sampled_token_ids)
    torch.npu.synchronize()

    # ========== 3. Compute reference values using PyTorch ==========
    if num_logprobs == 0:
        ref_token_ids = sampled_token_ids.unsqueeze(-1)
    else:
        topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
        ref_token_ids = torch.cat((sampled_token_ids.unsqueeze(-1), topk_indices), dim=1)

    ref_all_logprobs = torch.log_softmax(logits, dim=-1)
    ref_logprobs = torch.gather(ref_all_logprobs, dim=1, index=ref_token_ids)

    sampled_logits = torch.gather(logits, 1, sampled_token_ids.unsqueeze(-1))
    ref_ranks = (logits > sampled_logits).sum(dim=1).to(torch.int64)

    # ========== 4. Verify results ==========
    assert torch.equal(triton_output.logprob_token_ids, ref_token_ids), (
        "Token IDs (Sampled + TopK) do not match between Triton and PyTorch."
    )

    assert torch.equal(triton_output.selected_token_ranks, ref_ranks), (
        f"Token Ranks do not match.\nTriton: {triton_output.selected_token_ranks}\nPyTorch: {ref_ranks}"
    )

    assert torch.allclose(triton_output.logprobs, ref_logprobs, rtol=1e-4, atol=1e-4), (
        f"Logprobs values differ between Triton and PyTorch.\n"
        f"Max diff: {torch.max(torch.abs(triton_output.logprobs - ref_logprobs))}"
    )


@pytest.mark.parametrize("num_topk", [0, 3])
def test_v2_custom_logprobs_mixed_rows(num_topk):
    init_device_properties_triton()
    # State slots 2 and 0 request custom IDs; slot 1 uses ordinary top-k.
    candidates = torch.arange(128, device="npu", dtype=torch.int32).repeat(3, 1)
    candidates[0, :2] = torch.tensor([42, 9], device="npu")
    state = SimpleNamespace(
        token_ids=SimpleNamespace(gpu=candidates),
        num_token_ids=SimpleNamespace(gpu=torch.tensor([2, 0, 128], device="npu", dtype=torch.int32)),
    )
    slots = torch.tensor([2, 1, 1, 0], device="npu", dtype=torch.int32)
    logits = torch.arange(256, device="npu", dtype=torch.float32).repeat(4, 1) / 100
    sampled = torch.full((4,), 255, device="npu", dtype=torch.int64)
    result = compute_topk_logprobs(logits, num_topk, sampled, [0, 1, 3, 4], state, slots, 128)
    assert result.cu_num_generated_tokens == [0, 1, 3, 4]
    assert result.logprobs.shape == (4, 129)
    reference = logits.log_softmax(-1)
    for row, labels in enumerate((list(range(128)), [255, 254, 253][:num_topk], [255, 254, 253][:num_topk], [42, 9])):
        ids = [255, *labels]
        assert result.logprob_token_ids[row, : len(ids)].tolist() == ids
        torch.testing.assert_close(result.logprobs[row, : len(ids)], reference[row, ids], atol=1e-4, rtol=1e-4)
        assert result.logprobs[row, len(ids) :].isneginf().all()
