# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import lru_cache

import torch
from vllm.logger import logger
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

from vllm_ascend.ops.triton.fla.l2norm import l2norm_fwd_from_head_first

MEGA_GDN_CHUNK_SIZE = 128
MEGA_GDN_HEAD_DIM = 128
MEGA_GDN_MAX_VALUE_HEADS = 64
_MEGA_GDN_SCALE = MEGA_GDN_HEAD_DIM**-0.5


def _supports_head_configuration(
    dtype: torch.dtype, key_heads: int, value_heads: int, key_dim: int, value_dim: int
) -> bool:
    return (
        dtype in (torch.bfloat16, torch.float16)
        and key_dim == value_dim == MEGA_GDN_HEAD_DIM
        and 1 <= value_heads <= MEGA_GDN_MAX_VALUE_HEADS
        and key_heads > 0
        and value_heads % key_heads == 0
    )


def _load_mega_chunk_gdn():
    try:
        from custom_ops_lib import mega_chunk_gdn  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_XLLM_MEGA_GDN=1 requires the xLLM MegaChunkGdn "
            "vendor package and its custom_ops_lib Python extension"
        ) from exc
    return mega_chunk_gdn


def validate_xllm_mega_gdn(dtype: torch.dtype, key_heads: int, value_heads: int, key_dim: int, value_dim: int) -> None:
    if not _supports_head_configuration(dtype, key_heads, value_heads, key_dim, value_dim):
        raise RuntimeError(
            f"Unsupported xLLM MegaChunkGdn configuration: dtype={dtype}, "
            f"key_heads={key_heads}, value_heads={value_heads}, key_dim={key_dim}, value_dim={value_dim}"
        )
    _load_mega_chunk_gdn()


def _normalize_qk(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous() and x.stride(1) == x.shape[-1] and x.stride(-1) == 1:
        return l2norm_fwd_from_head_first(x)
    return l2norm_fwd(x.contiguous())


@dataclass(frozen=True)
class XllmMegaGdnMetadata:
    cu_seqlens: torch.Tensor
    num_chunks: int
    num_tokens: int
    active_rows: torch.Tensor | None = None

    @classmethod
    def build(cls, cu_seqlens_cpu: torch.Tensor, cu_seqlens: torch.Tensor) -> "XllmMegaGdnMetadata":
        # The scheduler already has CPU lengths and device offsets. Derive
        # the chunk count without reading back the device tensor.
        if cu_seqlens_cpu.device.type != "cpu":
            raise ValueError("xLLM MegaChunkGdn requires host query offsets")
        offsets = cu_seqlens_cpu.tolist()
        lengths = [end - start for start, end in zip(offsets, offsets[1:])]
        if not lengths or offsets[0] != 0 or min(lengths) < 0 or offsets[-1] == 0:
            raise ValueError("xLLM MegaChunkGdn requires ordered offsets for a nonempty prefill")
        num_chunks = sum((length + MEGA_GDN_CHUNK_SIZE - 1) // MEGA_GDN_CHUNK_SIZE for length in lengths)
        active_rows = None
        if 0 in lengths:
            rows = [row for row, length in enumerate(lengths) if length]
            # The xLLM kernel needs positive sequence lengths. Compact the
            # scheduler's empty rows and keep their cache slots untouched.
            device = cu_seqlens.device
            pin_memory = device.type == "npu"
            compact_metadata = torch.tensor(
                [0, *(offsets[row + 1] for row in rows), *rows], dtype=torch.int32, pin_memory=pin_memory
            ).to(device, non_blocking=True)
            cu_seqlens, active_rows = compact_metadata.split((len(rows) + 1, len(rows)))
        return cls(cu_seqlens.to(dtype=torch.int32).contiguous(), num_chunks, offsets[-1], active_rows)

    def supports(self, q: torch.Tensor, v: torch.Tensor) -> bool:
        return (
            q.device.type == "npu"
            and q.dtype == v.dtype
            and _supports_head_configuration(q.dtype, q.shape[-2], v.shape[-2], q.shape[-1], v.shape[-1])
            and q.shape[1] == v.shape[1] == self.num_tokens
        )


class XllmMegaGdnPrefill:
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.op = _load_mega_chunk_gdn()
        mask = torch.ones((MEGA_GDN_CHUNK_SIZE, MEGA_GDN_CHUNK_SIZE), dtype=torch.float32, device=device)
        self.mask_lower = mask.tril(-1)
        self.mask_full = mask.tril()
        self.minus_identity = -torch.eye(MEGA_GDN_CHUNK_SIZE, dtype=dtype, device=device)
        logger.info_once("Using xLLM MegaChunkGdn for GDN prefill")

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        metadata: XllmMegaGdnMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The cache gather already supplies the operator's packed K/V layout.
        results = self.op(
            _normalize_qk(q),
            _normalize_qk(k),
            v.contiguous(),
            g.to(torch.float32).contiguous(),
            beta.to(v.dtype).contiguous(),
            self.mask_lower,
            self.mask_full,
            self.minus_identity,
            metadata.cu_seqlens,
            initial_state,
            metadata.num_chunks * v.shape[-2],
            True,
        )
        out, final_state = results[0], results[-1]
        del results
        out.mul_(_MEGA_GDN_SCALE)
        return out, final_state.view_as(initial_state)


@lru_cache
def get_xllm_mega_gdn_prefill(device: torch.device, dtype: torch.dtype) -> XllmMegaGdnPrefill:
    """Share immutable masks across the layers of a device and dtype."""
    return XllmMegaGdnPrefill(device, dtype)
