# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the exact v0.30 gather on real CPU tensors, without importing vLLM.

Run in an isolated process with --torch-site-packages pointing at an existing
CPU torch installation if this interpreter's environment has no torch package.
Only missing mark_unbacked annotations may be omitted; no math is replaced.
This does not test Ascend buffers, device copies, or the NPU model runner.
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

BASE = "ced6857afa0ea7b2e3f0846a62e1394e90f15607"


def source(root, path):
    return subprocess.check_output(["git", "-C", str(root), "show", f"{BASE}:{path}"]).decode()


def find(tree, name):
    return next(node for node in tree.body if getattr(node, "name", None) == name)


def execute(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, "v030-extracted-cpu-probe", "exec"), namespace)


def probe(root):
    import torch

    paths = [
        "vllm/v1/outputs.py",
        "vllm/v1/sample/ops/logprobs.py",
        "vllm/v1/sample/sampler.py",
    ]
    sources = {path: source(root, path) for path in paths}
    namespace = {
        "torch": torch,
        "NamedTuple": NamedTuple,
        "Sequence": Sequence,
        "current_platform": SimpleNamespace(simple_compile_backend="eager"),
    }
    tree = ast.parse(sources[paths[0]])
    execute([find(tree, "LogprobsTensors")], namespace)
    helper = find(ast.parse(sources[paths[1]]), "batched_count_greater_than")
    execute([helper], namespace)
    sampler = find(ast.parse(sources[paths[2]]), "Sampler")
    gather = find(sampler, "gather_specific_token_logprobs")
    omitted = 0
    if not hasattr(torch._dynamo.decorators, "mark_unbacked"):
        retained = []
        for statement in gather.body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and ast.unparse(statement.value.func) == "torch._dynamo.decorators.mark_unbacked"
            ):
                omitted += 1
            else:
                retained.append(statement)
        assert omitted == 2
        gather.body = retained
    execute([gather], namespace)
    method = namespace["gather_specific_token_logprobs"]
    sampler_self = SimpleNamespace(pin_memory=False)
    logits = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 256)
    logprobs = torch.log_softmax(logits, dim=-1)
    sampled = torch.tensor([255, 5, 200, 9], dtype=torch.int64)
    selections = {2: list(range(128)), 0: [4, 5], 1: list(reversed(range(26)))}
    result = method(sampler_self, logprobs, selections, sampled)
    assert result.logprobs.shape == result.logprob_token_ids.shape == (4, 129)
    for row in range(4):
        requested = selections.get(row, [])
        ids = [sampled[row].item(), *requested]
        assert result.logprob_token_ids[row, : len(ids)].tolist() == ids
        torch.testing.assert_close(result.logprobs[row, : len(ids)], logprobs[row, ids])
        assert torch.isneginf(result.logprobs[row, len(ids) :]).all()
        expected_rank = (logprobs[row] >= logprobs[row, sampled[row]]).sum()
        torch.testing.assert_close(result.selected_token_ranks[row], expected_rank)
    assert len(set(result.logprob_token_ids[2].tolist())) == 129
    reordered = method(sampler_self, logprobs, dict(reversed(list(selections.items()))), sampled)
    for actual, expected in zip(reordered[:3], result[:3]):
        torch.testing.assert_close(actual, expected)
    assert method(sampler_self, logprobs, {}, sampled) is None
    sampled_only = method(sampler_self, logprobs, {0: []}, sampled)
    assert sampled_only.logprobs.shape == (4, 1)
    torch.testing.assert_close(sampled_only.logprobs[:, 0], logprobs.gather(1, sampled[:, None])[:, 0])
    return {
        "status": "passed",
        "torch": torch.__version__,
        "device": "cpu",
        "source_commit": BASE,
        "source_sha256": {path: hashlib.sha256(value.encode()).hexdigest() for path, value in sources.items()},
        "helper_compile_backend": "eager",
        "omitted_missing_mark_unbacked_annotations": omitted,
        "checks": [
            "mixed 2/26/128 IDs",
            "129 padded columns",
            "padding=-inf",
            "sampled inside/outside candidates",
            "128 candidates plus outside sample",
            "request-index mapping reordering",
            "no selections",
            "empty selection",
        ],
        "boundary": ("CPU gather math and shape only; no NPU buffer, transfer or runner proof"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--torch-site-packages", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.torch_site_packages and not args.worker:
        env = os.environ | {"PYTHONPATH": str(args.torch_site_packages.resolve())}
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--vllm-root",
                str(args.vllm_root.resolve()),
                "--output",
                str(args.output.resolve()),
            ],
            env=env,
        )
        raise SystemExit(result.returncode)
    report = {"status": "failed", "source_commit": BASE}
    try:
        report = probe(args.vllm_root)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
