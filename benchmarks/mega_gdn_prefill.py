# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Compare native FLA-NPU and xLLM MegaChunkGdn prefill on one Ascend 910B.

Both paths receive identical synthetic post-convolution Q/K/V and gates.
Timing includes input preparation, the GDN operator, and state-cache I/O.
"""

import argparse
import itertools
import json
import os
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd

import vllm_ascend.ops  # noqa: F401  # Register the Ascend ops before importing DeviceAdaptor.
from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.gdn_attn_builder import _build_non_spec_chunked_prefill_metadata
from vllm_ascend.ops.triton.mamba.state_index import gather_ssm_states, scatter_ssm_states_
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.xllm_mega_gdn import (
    MEGA_GDN_HEAD_DIM,
    XllmMegaGdnMetadata,
    get_xllm_mega_gdn_prefill,
    validate_xllm_mega_gdn,
)

SCALE = MEGA_GDN_HEAD_DIM**-0.5
DEFAULT_CASES = ((128,), (384,), (2048,), (4096,), (63, 129, 257))
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def parse_case(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(field) for field in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated sequence lengths") from exc
    if not all(length > 0 for length in lengths):
        raise argparse.ArgumentTypeError("sequence lengths must be positive")
    return lengths


def make_inputs(args, lengths, dtype):
    rng = torch.Generator().manual_seed(args.seed)
    tokens, dim = sum(lengths), MEGA_GDN_HEAD_DIM
    q = torch.randn((1, tokens, args.key_heads, dim), generator=rng).to(dtype)
    key_tokens = 1 if args.correlated_keys else tokens
    k = torch.randn((1, key_tokens, args.key_heads, dim), generator=rng).to(dtype).expand_as(q)
    v = (torch.randn((1, tokens, args.value_heads, dim), generator=rng) * 0.25).to(dtype)
    g = torch.rand((1, tokens, args.value_heads), generator=rng)
    g = g * (-0.001 if args.correlated_keys else -0.12) - 0.001
    beta = torch.rand(g.shape, generator=rng)
    if args.correlated_keys:
        beta = 0.985 - 0.01 * beta
    state = torch.randn((len(lengths), args.value_heads, dim, dim), generator=rng) * 0.01
    return q, k, v, g, beta.to(dtype), state


def fp64_reference(lengths, q, k, v, g, beta, initial):
    # Serial recurrence in cache layout [sequence, head, value_dim, key_dim].
    q, k, v, g, beta = (x[0].double() for x in (q, k, v, g, beta))
    group = v.shape[1] // q.shape[1]
    q, k = ((x * (x.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).repeat_interleave(group, dim=1) for x in (q, k))
    output, final = torch.empty_like(v), initial.double().clone()
    starts = (0, *itertools.accumulate(lengths))
    for seq, (start, end) in enumerate(itertools.pairwise(starts)):
        state = final[seq]
        for token in range(start, end):
            key = k[token]
            state.mul_(g[token].exp()[:, None, None])
            prediction = torch.bmm(state, key.unsqueeze(-1)).squeeze(-1)
            residual = (v[token] - prediction) * beta[token, :, None]
            state.addcmul_(residual.unsqueeze(-1), key.unsqueeze(-2))
            output[token] = torch.bmm(state, q[token].unsqueeze(-1)).squeeze(-1) * SCALE
    return output.unsqueeze(0), final


def make_prefill_calls(lengths, inputs, device):
    q, k, v, g, beta, initial = inputs
    # Reproduce the head-first Conv1D storage used by the serving branch.
    q, k, v = (x.transpose(1, 2).contiguous().to(device).transpose(1, 2) for x in (q, k, v))
    g, beta = g.to(device), beta.to(device)
    cache = initial.to(device)
    state_rows = torch.arange(len(lengths), dtype=torch.int32, device=device)
    has_state = torch.ones(len(lengths), dtype=torch.bool, device=device)
    native_result, mega_result = (torch.empty_like(cache) for _ in range(2))
    offsets = torch.tensor((0, *itertools.accumulate(lengths)), dtype=torch.int32)
    native_meta = _build_non_spec_chunked_prefill_metadata(None, offsets, device, build_device_metadata=False)
    mega_meta = XllmMegaGdnMetadata.build(offsets, offsets.to(device))
    mega = get_xllm_mega_gdn_prefill(device, v.dtype)

    def native_prefill():
        state = gather_ssm_states(cache, state_rows, has_state)
        out, state = BaseDeviceAdaptor.fla_gdn_prefill(
            q, k, v, g, beta, state, SCALE, native_meta, chunk_gated_delta_rule_fwd
        )
        scatter_ssm_states_(native_result, state_rows, state.to(cache.dtype).contiguous())
        return out, native_result

    def mega_prefill():
        state = gather_ssm_states(cache, state_rows, has_state, output_dtype=v.dtype, transpose_last_two=True)
        out, state = mega(q, k, v, g, beta, state, mega_meta)
        scatter_ssm_states_(mega_result, state_rows, state, transpose_last_two=True)
        return out, mega_result

    return {"native": native_prefill, "mega": mega_prefill}


def time_prefill(calls, iterations):
    samples = {name: [] for name in calls}
    for group in range(10):
        order = list(calls) if group % 2 == 0 else list(reversed(calls))
        for name in order:
            run = calls[name]
            run()
            start, stop = (torch.npu.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(iterations):
                run()
            stop.record()
            stop.synchronize()
            samples[name].append(start.elapsed_time(stop) * 1000 / iterations)
    return {name: {"median_us": statistics.median(values), "samples_us": values} for name, values in samples.items()}


def error_metrics(actual, expected, args):
    actual, expected = actual.cpu().double(), expected.cpu().double()
    nonfinite = int((~actual.isfinite()).sum())
    if nonfinite or not expected.isfinite().all():
        return {"nonfinite": nonfinite, "passed": False}
    error = actual - expected
    max_abs = error.abs().max().item()
    relative_rmse = (error.norm() / expected.norm().clamp_min(1e-12)).item()
    return {
        "nonfinite": 0,
        "max_abs": max_abs,
        "relative_rmse": relative_rmse,
        "passed": max_abs <= args.abs_limit or relative_rmse < args.rmse_limit,
    }


def run_case(args, lengths, dtype, device):
    inputs = make_inputs(args, lengths, dtype)
    calls = make_prefill_calls(lengths, inputs, device)
    results = {name: run() for name, run in calls.items()}
    results["cpu_fp64"] = fp64_reference(lengths, *inputs)
    accuracy = {}
    for actual, expected in (("mega", "cpu_fp64"), ("native", "cpu_fp64"), ("mega", "native")):
        accuracy[f"{actual}_vs_{expected}"] = {
            name: error_metrics(a, e, args)
            for name, a, e in zip(("output", "final_state"), results[actual], results[expected], strict=True)
        }
    passed = all(metric["passed"] for comparison in accuracy.values() for metric in comparison.values())
    timings = time_prefill(calls, args.iterations)
    speedup = timings["native"]["median_us"] / timings["mega"]["median_us"]
    return {"lengths": lengths, "timing": timings, "speedup": speedup, "accuracy": accuracy, "passed": passed}


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="logical NPU index after ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--case", type=parse_case, action="append", help="packed sequence lengths, repeatable")
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--key-heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=20, help="timed calls per group")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--correlated-keys", action="store_true", help="reuse key directions with high beta")
    parser.add_argument("--abs-limit", type=float, default=0.01)
    parser.add_argument("--rmse-limit", type=float, default=0.05)
    parser.add_argument("--vendor", type=Path, help="xLLM vendor directory containing op_api/lib/libcust_opapi.so")
    parser.add_argument("--json", type=Path, help="write per-case timings and precision metrics")
    args = parser.parse_args()
    if args.iterations < 1 or args.abs_limit < 0 or args.rmse_limit <= 0:
        parser.error("iterations and relative RMSE limit must be positive, and absolute limit nonnegative")
    if args.vendor is not None:
        vendor = args.vendor.resolve()
        if not (vendor / "op_api/lib/libcust_opapi.so").is_file():
            parser.error(f"missing xLLM op API library under {vendor}")
        previous = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        # The binding resolves vendor paths at import time. Give the selected
        # package priority over the paths registered by other NPU packages.
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(
            (str(vendor), *(entry for entry in previous if entry and entry != str(vendor)))
        )

    dtype = DTYPES[args.dtype]
    device = torch.device("npu", args.device)
    torch.set_num_threads(2)
    torch.npu.set_device(device)
    torch.npu.set_compile_mode(jit_compile=False)
    if not get_current_hardware_profile().supports(HardwareCapability.XLLM_MEGA_GDN_PREFILL):
        parser.error("this comparison uses the Ascend 910B native FLA-NPU path")
    init_device_properties_triton()
    validate_xllm_mega_gdn(dtype, args.key_heads, args.value_heads, MEGA_GDN_HEAD_DIM, MEGA_GDN_HEAD_DIM)

    results = []
    print(f"{'Sequence lengths':>20} {'Native (us)':>14} {'Mega (us)':>14} {'Speedup':>9} {'Precision':>10}")
    for lengths in args.case or DEFAULT_CASES:
        result = run_case(args, lengths, dtype, device)
        results.append(result)
        native = result["timing"]["native"]["median_us"]
        mega = result["timing"]["mega"]["median_us"]
        status = "PASS" if result["passed"] else "FAIL"
        label = ",".join(map(str, lengths))
        print(f"{label:>20} {native:>14.2f} {mega:>14.2f} {result['speedup']:>8.2f}x {status:>10}")
        for comparison, metrics in result["accuracy"].items():
            for name, metric in metrics.items():
                details = (
                    f"max_abs={metric['max_abs']:.3g}, rel_rmse={metric['relative_rmse']:.3%}"
                    if "max_abs" in metric
                    else f"nonfinite={metric['nonfinite']}"
                )
                print(f"  {comparison:>19} {name:>12} {details}")
    if args.json:
        config = {key: value for key, value in vars(args).items() if key not in {"json", "vendor"}}
        args.json.write_text(json.dumps({"config": config, "results": results}, indent=2) + "\n", encoding="utf-8")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
