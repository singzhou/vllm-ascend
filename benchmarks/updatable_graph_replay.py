# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare target graph submission against the previous replay implementation.

Run on an idle NPU: python benchmarks/updatable_graph_replay.py --layers 16
Reports CPU submission and synchronized completion, not full-model TPOT.
"""

import argparse
import json
import os
import statistics
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, ModelCudaGraphManager

from vllm_ascend.attention.attention_v1 import FIAParamProvider
from vllm_ascend.attention.fia_cache import call_fia_with_virtual_pages
from vllm_ascend.compilation.updatable_graph import ContextSource, GraphUpdateTask, UpdatableGraph, _get_ws_key
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager


def baseline_bind(task, params):
    return replace(task, kwargs={**task.kwargs, **params})


def baseline_replay(manager, desc, metadata):
    graph = manager.graphs[desc]
    tasks = graph.resolve_tasks(ContextSource(metadata))
    manager.update_stream.wait_stream(torch.npu.current_stream())
    result = ModelCudaGraphManager.run_fullgraph(manager, desc)
    baseline_update(graph, manager.update_stream, tasks)
    return result


def baseline_update(graph, update_stream, resolved_tasks):
    """Original update body, including its stream context and PA guard."""
    logger.debug_once("Updating host-side attention metadata with UpdatableGraph.")
    with torch.npu.stream(update_stream):
        ws_buffer = {}
        for task in resolved_tasks:
            if task.operation == torch_npu._npu_paged_attention:
                ws_key = _get_ws_key(task.kwargs)
                if ws_buffer.get(ws_key) is None:
                    ws_kwargs = task.kwargs.copy()
                    ws_kwargs.pop("workspace")
                    workspace = torch_npu._npu_paged_attention_get_workspace(**ws_kwargs)
                    ws_buffer[ws_key] = workspace
                task.kwargs["workspace"] = ws_buffer[ws_key]
            task.apply(update_stream)


def make_case(layers):
    query = torch.randn(8, 6, 256, dtype=torch.bfloat16, device="npu")
    raw = torch.randn(6 * 2 * 128 * 256, dtype=query.dtype, device="npu")
    key = raw.as_strided((6, 128, 256), (2 * 128 * 256, 256, 1))
    value = raw.as_strided(key.shape, key.stride(), storage_offset=128 * 256)
    table = torch.tensor([[0, 3], [5, 1]], device="npu", dtype=torch.int32)
    metadata = SimpleNamespace(actual_seq_lengths_q=[4, 8], seq_lens_list=[137, 191], block_tables=table)
    op = partial(call_fia_with_virtual_pages, torch_npu.npu_fused_infer_attention_score.out)
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        block_table=table,
        input_layout="TND",
        block_size=128,
        actual_seq_lengths=[4, 8],
        actual_seq_lengths_kv=[137, 191],
        num_heads=6,
        num_key_value_heads=1,
        scale=256**-0.5,
        sparse_mode=0,
        pre_tokens=2147483647,
        next_tokens=2147483647,
    )
    workspace = call_fia_with_virtual_pages(torch_npu._npu_fused_infer_attention_score_get_max_workspace, **kwargs)
    outputs = [torch.empty_like(query) for _ in range(layers)]
    lses = [torch.empty(1, dtype=query.dtype, device=query.device) for _ in range(layers)]
    for output, lse in zip(outputs, lses):
        op(**kwargs, workspace=workspace, out=[output, lse])
    torch.npu.synchronize()
    graph = UpdatableGraph()
    with torch.npu.graph(graph, stream=torch.npu.Stream()):
        for i, (output, lse) in enumerate(zip(outputs, lses)):
            graph.register_task(
                op, {**kwargs, "workspace": workspace, "out": [output, lse]}, FIAParamProvider(f"layer{i}", None)
            )
    torch.npu.synchronize()
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens=8, num_reqs=2)
    manager = ModelAclGraphManager.__new__(ModelAclGraphManager)
    manager.device, manager.update_stream = query.device, torch.npu.Stream()
    manager._update_inputs_ready = torch.npu.Event()
    manager.graphs, manager.hidden_states = {desc: graph}, outputs[-1]
    manager.is_last_pp_rank, manager.use_aux_hidden_state_outputs = True, False
    # Capture tasks intentionally keep weak tensor references. Keep allocations
    # alive for both variants, just as the model runner owns production buffers.
    resources = (query, raw, key, value, workspace, lses)
    return manager, desc, {f"layer{i}": metadata for i in range(layers)}, outputs, resources


def summarize(values):
    values = sorted(values)
    return dict(
        median_us=statistics.median(values),
        mean_us=statistics.mean(values),
        p95_us=values[int(0.95 * (len(values) - 1))],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("updatable_graph_replay.json"))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--with-modules", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    manager, desc, metadata, outputs, _resources = make_case(args.layers)
    bind = GraphUpdateTask.bind
    variants = {
        "before": (baseline_bind, baseline_replay),
        "after": (bind, ModelAclGraphManager._updatable_graph_replay),
    }
    samples = {name: {"host_submit_us": [], "completed_us": []} for name in variants}
    # Warm both variants before sampling, including lazy event initialization.
    for name, (binder, replay) in variants.items():
        with patch.object(GraphUpdateTask, "bind", binder):
            for _ in range(20):
                replay(manager, desc, metadata)
                torch.npu.synchronize()
    for i in range(args.repeats):
        for name in list(variants) if i % 2 else list(reversed(variants)):
            binder, replay = variants[name]
            with patch.object(GraphUpdateTask, "bind", binder):
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                replay(manager, desc, metadata)
                submitted = time.perf_counter_ns()
                torch.npu.synchronize()
                done = time.perf_counter_ns()
            samples[name]["host_submit_us"].append((submitted - start) / 1000)
            samples[name]["completed_us"].append((done - start) / 1000)
    # Same graph, inputs and tasks: parameter updates must preserve numerical output.
    with patch.object(GraphUpdateTask, "bind", baseline_bind):
        baseline_replay(manager, desc, metadata)
        torch.npu.synchronize()
        expected = [x.clone() for x in outputs]
    manager._updatable_graph_replay(desc, metadata)
    torch.npu.synchronize()
    for actual, reference in zip(outputs, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    result = dict(
        layers=args.layers,
        repeats=args.repeats,
        task_queue_enable=os.environ.get("TASK_QUEUE_ENABLE"),
        summary={n: {k: summarize(v) for k, v in row.items()} for n, row in samples.items()},
        samples=samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)
    if args.profile:
        for name, (binder, replay) in variants.items():
            with (
                patch.object(GraphUpdateTask, "bind", binder),
                torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=5, repeat=1),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(args.output.parent / f"profile_{name}")
                    ),
                    with_modules=args.with_modules,
                    experimental_config=torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                    ),
                ) as prof,
            ):
                for _ in range(6):
                    replay(manager, desc, metadata)
                    torch.npu.synchronize()
                    prof.step()


if __name__ == "__main__":
    main()
