import torch
import torch_npu


def apply_top_k_top_p_npu(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    if k is None and p is None:
        return logits

    return torch_npu.npu_top_k_top_p(logits, k=k, p=p)
