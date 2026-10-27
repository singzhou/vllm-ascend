# SPDX-License-Identifier: Apache-2.0
"""Compare GDN graph latency before/after native legacy state-stride support.

Run with ASCEND_RT_VISIBLE_DEVICES set to an available device. Keep shapes,
device, software and load identical for both revisions; this measures an
operator, not service throughput. --profile exports one batch-32/width-9 replay.
"""

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
import torch_npu
from fla_npu.ops.ascendc import recurrent_gated_delta_rule

from vllm_ascend.ops.gdn_recurrent import legacy_recurrent_gated_delta_rule, select_gdn_recurrent_backend
from vllm_ascend.utils import enable_custom_op


def measure(batch, width, padded, samples, profile):
    nk, nv, dim = 4, 12, 128
    tokens = batch * width
    row_size = nv * dim * dim
    backing = torch.zeros((tokens + 2, row_size + (32 if padded else 0)), device="npu")
    state = backing[:, 16 : 16 + row_size] if padded else backing
    state = state.view(tokens + 2, nv, dim, dim)
    q, k = [torch.randn(tokens, nk, dim, device="npu", dtype=torch.bfloat16) * 0.05 for _ in range(2)]
    v = torch.randn(tokens, nv, dim, device="npu", dtype=torch.bfloat16) * 0.1
    kwargs = dict(
        beta=torch.full((tokens, nv), 0.2, dtype=torch.bfloat16, device="npu"),
        g=torch.full((tokens, nv), -0.1, dtype=torch.float32, device="npu"),
        scale=dim**-0.5,
        actual_seq_lengths=torch.tensor([0] + [width] * batch, dtype=torch.int32, device="npu"),
        ssm_state_indices=torch.arange(1, tokens + 1, dtype=torch.int32, device="npu"),
        num_accepted_tokens=torch.ones(batch, dtype=torch.int32, device="npu"),
    )
    backend = select_gdn_recurrent_backend(width - 1, False)
    op = legacy_recurrent_gated_delta_rule if backend == "legacy" else recurrent_gated_delta_rule
    for _ in range(3):
        op(q, k, v, state, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = op(q, k, v, state, **kwargs)
    for _ in range(20):
        graph.replay()
    torch.npu.synchronize()
    timings = []
    start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000)
    assert torch.isfinite(output).all()
    if profile and batch == 32 and width == 9 and padded:
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile),
        ):
            for _ in range(5):
                graph.replay()
            torch.npu.synchronize()
    return dict(
        batch=batch,
        width=width,
        padded=padded,
        backend=backend,
        median_us=statistics.median(timings),
        min_us=min(timings),
        samples=samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--profile")
    args = parser.parse_args()
    assert enable_custom_op()
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(41)
    results = []
    for batch in (1, 4, 32):
        for width in (8, 9, 16):
            for padded in (False, True):
                row = measure(batch, width, padded, args.samples, args.profile)
                results.append(row)
                print(json.dumps(row), flush=True)
                gc.collect()
                torch.npu.empty_cache()
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
