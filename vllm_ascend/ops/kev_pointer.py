# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Sparse pointer readout over final hidden states, with chunk accumulation."""

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from vllm.model_executor.layers.pooler.abstract import Pooler

from vllm_ascend.decision.planner import ReadoutSpec
from vllm_ascend.entrypoints.systemone.protocol import MAX_OPTIONS


@dataclass
class ReadoutState:
    spec: ReadoutSpec
    end: int = 0
    options: dict[int, torch.Tensor] = field(default_factory=dict)


class KevPointerPooler(Pooler):
    def __init__(self, hidden_size, head_dim):
        super().__init__()
        self.q = nn.Linear(hidden_size, head_dim, dtype=torch.float32)
        self.k = nn.Linear(hidden_size, head_dim, dtype=torch.float32)
        self.scale = 1 / math.sqrt(head_dim)
        self.head_dim = head_dim

    def get_supported_tasks(self):
        return {"plugin"}

    def forward(self, hidden_states, pooling_metadata):
        if self.q.weight.dtype != torch.float32 or self.k.weight.dtype != torch.float32:
            raise RuntimeError("Kev pointer head must remain FP32")
        cursor = pooling_metadata.get_pooling_cursor()
        counts = cursor.num_scheduled_tokens_cpu.tolist()
        ends = cursor.seq_lens_cpu.tolist()
        lengths = cursor.prompt_lens_cpu.tolist()

        outputs = []
        offset = 0
        for params, owner, count, end, length in zip(
            pooling_metadata.pooling_params,
            pooling_metadata.pooling_states,
            counts,
            ends,
            lengths,
        ):
            if params.task != "plugin":
                raise ValueError("Kev requires plugin pooling")
            extra = params.extra_kwargs or {}
            spec = ReadoutSpec.parse(extra.get("kev_readout"), length)
            start = end - count
            state = getattr(owner, "kev_readout", None)
            # All cache hits/resumes start at or before state_length. No option
            # readout exists in this region, so rebuilding the accumulator is
            # safe both on the first chunk and after preemption. A paused
            # request continuing beyond the state keeps its existing entries.
            if start <= spec.state_length:
                state = ReadoutState(spec, end=start)
                owner.kev_readout = state
            if state is None or state.spec != spec or state.end != start:
                raise RuntimeError("Kev readout has a gap or stale request state")
            if count <= 0 or end > length:
                raise RuntimeError("Invalid Kev prefill cursor")

            indices = [(i, p - start) for i, p in enumerate(spec.option_ends) if start <= p < end]
            if indices:
                gather = torch.tensor(
                    [offset + p for _, p in indices],
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                projected = self.k(hidden_states.index_select(0, gather).float())
                # Clone the compact projection, never retain a view of the full
                # runner hidden-state tensor across scheduler steps.
                for (i, _), value in zip(indices, projected):
                    state.options[i] = value.clone()
            state.end = end
            if end == length:
                if len(state.options) != len(spec.option_ends):
                    raise RuntimeError("Completed Kev prompt is missing option readouts")
                query = self.q(hidden_states[offset + spec.decide - start].float())
                keys = torch.stack([state.options[i] for i in range(len(spec.option_ends))])
                outputs.append((keys @ query) * self.scale)
                del owner.kev_readout
            else:
                outputs.append(None)
            offset += count
        return outputs

    def profile(self, hidden_states, counts):
        # Include worst-case per-request K projection memory in runner profiling.
        # Called explicitly by NPUModelRunner, never inferred from missing metadata.
        result, retained = [], []
        offset = 0
        for count in counts:
            h = hidden_states[offset].float()
            keys = self.k(h.expand(MAX_OPTIONS, -1))
            retained.append(keys)
            result.append((keys @ self.q(h)) * self.scale)
            offset += count
        return result
