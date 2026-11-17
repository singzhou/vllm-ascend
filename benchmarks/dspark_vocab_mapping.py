# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare DSpark vocabulary expansion with index_copy_ under NPU graphs.

Example: ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/dspark_vocab_mapping.py
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.v2.spec_decode.expand_draft_logits import (
    expand_draft_logits_,
    make_draft_vocab_block_starts,
    make_target_to_draft_map,
)


def capture(fn):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        fn()
    return graph


def measure(graph, iterations, repeats):
    samples = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--draft-vocab", type=int, default=64000)
    parser.add_argument("--target-vocab", type=int, default=248320)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unsorted", action="store_true", help="Exercise the unordered-map fallback")
    args = parser.parse_args()
    if min(args.rows, args.draft_vocab, args.iterations, args.repeats) < 1 or args.draft_vocab > args.target_vocab:
        parser.error("rows/iterations/repeats must be positive and 0 < draft-vocab <= target-vocab")

    torch.npu.set_device(0)
    torch.manual_seed(17)
    dtype = getattr(torch, args.dtype)
    target_ids = torch.randperm(args.target_vocab)[: args.draft_vocab]
    if not args.unsorted:
        target_ids = target_ids.sort().values
    target_ids = target_ids.to("npu")
    inverse = make_target_to_draft_map(target_ids, args.target_vocab)
    block_starts = make_draft_vocab_block_starts(target_ids, args.target_vocab)
    logits = torch.randn(args.rows, args.draft_vocab, dtype=dtype, device="npu")
    reference = torch.full((args.rows, args.target_vocab), -float("inf"), dtype=dtype, device="npu")
    expanded = torch.empty_like(reference)
    before = capture(lambda: reference.index_copy_(1, target_ids, logits))
    after = capture(lambda: expand_draft_logits_(expanded, inverse, logits, block_starts))
    torch.testing.assert_close(expanded, reference, rtol=0, atol=0)
    result = {
        "rows": args.rows,
        "draft_vocab": args.draft_vocab,
        "target_vocab": args.target_vocab,
        "dtype": args.dtype,
        "window_path": block_starts is not None,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "index_copy": measure(before, args.iterations, args.repeats),
        "expand": measure(after, args.iterations, args.repeats),
        "bitwise_equal": True,
    }
    torch.testing.assert_close(expanded, reference, rtol=0, atol=0)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
