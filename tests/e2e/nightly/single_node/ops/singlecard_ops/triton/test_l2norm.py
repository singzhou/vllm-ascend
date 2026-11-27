import gc

import pytest
import torch
import torch.nn.functional as F
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd as reference_l2norm

from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd, l2norm_fwd_from_head_first
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.mark.parametrize(
    ("B", "T", "H", "D", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-D{}-{}".format(*test))
        for test in [
            (1, 63, 1, 60, torch.float),
            (2, 500, 4, 64, torch.float),
            (2, 1000, 2, 100, torch.float),
            (3, 1024, 4, 128, torch.float),
        ]
    ],
)
def test_l2norm(B: int, T: int, H: int, D: int, dtype: torch.dtype):
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    x = torch.randn(B, T, H, D, dtype=dtype).to(device).requires_grad_(True)
    x = x * 0.5 + 0.3

    ref = F.normalize(x, dim=-1, p=2)
    tri = l2norm_fwd(x)

    assert torch.allclose(tri, ref, rtol=rtol, atol=atol)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("tokens", [17, 385, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_l2norm_head_first_produces_packed_token_major_output(tokens, dtype):
    init_device_properties_triton()
    # Slice consecutive heads out of a shared convolution output.
    head_first = torch.randn((12, tokens, 128), dtype=dtype, device="npu")[4:8]
    view = head_first.transpose(0, 1).unsqueeze(0)
    output = l2norm_fwd_from_head_first(view)
    assert output.is_contiguous()
    torch.testing.assert_close(output, reference_l2norm(view.contiguous()), rtol=0, atol=0)
