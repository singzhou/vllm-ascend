# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure real-server warm A/B, fixed-prompt gather, or choice-size workloads.

Warm: --mode warm --off-url URL --on-url URL --state-file FILE --model ALIAS.
Gather: --mode gather --url URL --prompt-file FILE --codebook JSON --model ALIAS.
Business: --mode business --url URL --state-file FILE --model ALIAS.
Provide endpoint metadata JSON files with source commits, checkpoint/tokenizer,
TP/DP, cache, graph and scheduling settings. No runtime behavior is inferred.
Gather prompt-file must contain the full rendered reply-start prompt; codebook
IDs must match the deployed tokenizer. Generation TPOT is a client chunk-arrival
estimate, not token-level ITL.
"""

import argparse
import json
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def http(base_url, path, payload, args, request_id, stream=False):
    headers = {"Content-Type": "application/json", "X-Request-Id": request_id}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = Request(
        base_url.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers,
    )
    start = time.perf_counter()
    record = {"request_id": request_id, "request": payload}
    try:
        with urlopen(request, timeout=args.timeout) as response:
            record["status"] = response.status
            if stream:
                events, first, last, usage, done = [], None, None, None, False
                for line in response:
                    if not line.startswith(b"data:"):
                        continue
                    value = line[5:].strip()
                    if value == b"[DONE]":
                        done = True
                        break
                    event = json.loads(value)
                    elapsed = (time.perf_counter() - start) * 1000
                    events.append({"arrival_ms": elapsed, "data": event})
                    if any(choice.get("delta", {}).get("content") for choice in event.get("choices", [])):
                        first = elapsed if first is None else first
                        last = elapsed
                    usage = event.get("usage") or usage
                count = (usage or {}).get("completion_tokens", 0)
                record.update(
                    body=events,
                    usage=usage,
                    stream_done=done,
                    ttft_ms=first,
                    client_tpot_estimate_ms=(last - first) / (count - 1)
                    if first is not None and last is not None and count > 1
                    else None,
                )
            else:
                record["body"] = json.load(response)
    except HTTPError as error:
        record.update(status=error.code, body=error.read().decode(errors="replace"))
    except Exception as error:
        record.update(status=None, error=f"{type(error).__name__}: {error}")
    record["e2e_ms"] = (time.perf_counter() - start) * 1000
    return record


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def question_payload(state, count, choices, model, salt):
    return {
        "model": model,
        "state": state,
        "cache_salt": salt,
        "chat_template_kwargs": {"enable_thinking": False},
        "questions": {
            f"q{i}": {
                "type": "choice",
                "instructions": f"Pick the matching option for question {i}.",
                "criteria": {f"option_{j}": f"Option {j}" for j in range(choices)},
            }
            for i in range(count)
        },
    }


def run(args, report):
    """Measure the selected HTTP workload and record raw responses alongside timing data."""
    counts = args.questions if args.mode == "warm" else [1]
    choices = args.choices or ([26, 128] if args.mode == "warm" else [26, 64, 128])
    if any(n < 1 or n > 64 for n in counts):
        raise ValueError("questions must be in 1..64")
    if any(k < 1 or k > 128 for k in choices):
        raise ValueError("choices must be in 1..128")
    if args.mode == "warm":
        if not args.off_url or not args.on_url:
            raise ValueError("warm requires --off-url and --on-url")
        endpoints = [
            ("off", args.off_url, args.off_metadata),
            ("on", args.on_url, args.on_metadata),
        ]
    else:
        if not args.url:
            raise ValueError("gather/business requires --url")
        endpoints = [("server", args.url, args.metadata)]
    state, prompt, ids = None, None, None
    if args.mode == "gather":
        if not args.prompt_file or not args.codebook:
            raise ValueError("gather requires --prompt-file and --codebook")
        prompt = args.prompt_file.read_text(encoding="utf-8")
        codebook = json.loads(args.codebook.read_text(encoding="utf-8"))
        ids = codebook["token_ids"]
        if len(ids) != 128 or len(set(ids)) != 128:
            raise ValueError("codebook must contain 128 unique verified token_ids")
        report["codebook"] = codebook
    else:
        if not args.state_file:
            raise ValueError("warm/business requires --state-file")
        state = args.state_file.read_text(encoding="utf-8")

    for label, url, metadata in endpoints:
        report["endpoints"][label] = {
            "url": url,
            "operator_metadata": json.loads(metadata.read_text(encoding="utf-8")) if metadata else None,
            "metadata_verification": "operator supplied; verify server logs separately",
            "version": http(url, "/version", None, args, uuid.uuid4().hex),
            "models": http(url, "/v1/models", None, args, uuid.uuid4().hex),
        }
        for cache in args.cache_states:
            for count in counts:
                for size in choices:
                    group = {
                        "endpoint": label,
                        "cache": cache,
                        "questions": count,
                        "choices": size,
                    }
                    for repeat in range(args.repeats):
                        salt = f"bench-{uuid.uuid4().hex}"
                        payload = (
                            {
                                "model": args.model,
                                "prompt": prompt,
                                "add_special_tokens": False,
                                "cache_salt": salt,
                                "max_tokens": 1,
                                "temperature": 0,
                                "logprobs": 0,
                                "logprob_token_ids": ids[:size],
                            }
                            if args.mode == "gather"
                            else question_payload(state, count, size, args.model, salt)
                        )
                        path = "/v1/completions" if args.mode == "gather" else "/v1/systemone"
                        entry = group | {
                            "repeat": repeat,
                            "private_cache_metrics": None,
                            "scheduled_new_tokens": None,
                        }
                        if cache == "hot":
                            entry["prime"] = http(url, path, payload, args, f"prime-{uuid.uuid4().hex}")
                            if entry["prime"]["status"] != 200:
                                report["samples"].append(entry | {"prime_failed": True})
                                continue
                        request_id = f"bench-{uuid.uuid4().hex}"
                        if args.generation_tokens:
                            generation = {
                                "model": args.model,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Explain why the sky appears blue.",
                                    }
                                ],
                                "max_tokens": args.generation_tokens,
                                "temperature": 0,
                                "stream": True,
                                "stream_options": {"include_usage": True},
                                "chat_template_kwargs": {"enable_thinking": False},
                            }
                            barrier = Barrier(2)

                            def dispatch(
                                route,
                                body,
                                identity,
                                stream=False,
                                *,
                                barrier=barrier,
                                url=url,
                            ):
                                barrier.wait()
                                return http(url, route, body, args, identity, stream)

                            with ThreadPoolExecutor(max_workers=2) as pool:
                                decision_job = pool.submit(dispatch, path, payload, request_id)
                                generation_job = pool.submit(
                                    dispatch,
                                    "/v1/chat/completions",
                                    generation,
                                    f"generation-{uuid.uuid4().hex}",
                                    True,
                                )
                                entry["decision"] = decision_job.result()
                                entry["generation"] = generation_job.result()
                        else:
                            entry["decision"] = http(url, path, payload, args, request_id)
                        report["samples"].append(entry)

    groups = {}
    for entry in report["samples"]:
        key = json.dumps(
            {k: entry[k] for k in ("endpoint", "cache", "questions", "choices")},
            sort_keys=True,
        )
        groups.setdefault(key, []).append(entry)
    for key, entries in groups.items():
        valid = [e["decision"]["e2e_ms"] for e in entries if e.get("decision", {}).get("status") == 200]
        metrics = {}
        for metric in ("ttft_ms", "client_tpot_estimate_ms"):
            values = [
                e["generation"][metric]
                for e in entries
                if e.get("generation", {}).get("status") == 200 and e["generation"][metric] is not None
            ]
            metrics[metric] = {
                "p50": percentile(values, 0.5),
                "p95": percentile(values, 0.95),
                "count": len(values),
            }
        report["summary"].append(
            json.loads(key)
            | {
                "successes": len(valid),
                "failures": len(entries) - len(valid),
                "http_e2e_ms": {
                    "p50": percentile(valid, 0.5),
                    "p95": percentile(valid, 0.95),
                },
                "generation": metrics,
            }
        )
    report["status"] = (
        "passed"
        if report["samples"]
        and all(
            entry.get("decision", {}).get("status") == 200
            and (
                not args.generation_tokens
                or entry.get("generation", {}).get("status") == 200
                and entry["generation"].get("stream_done")
            )
            for entry in report["samples"]
        )
        else "failed"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["warm", "gather", "business"])
    for name in ("url", "off-url", "on-url", "model", "api-key"):
        parser.add_argument("--" + name, required=name == "model")
    for name in (
        "state-file",
        "prompt-file",
        "codebook",
        "metadata",
        "off-metadata",
        "on-metadata",
    ):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--questions", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--choices", type=int, nargs="+")
    parser.add_argument("--cache-states", nargs="+", choices=["cold", "hot"], default=["cold", "hot"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--generation-tokens", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.generation_tokens < 0:
        parser.error("repeats must be positive and generation-tokens nonnegative")
    report = {
        "status": "failed",
        "mode": args.mode,
        "started_unix": time.time(),
        "model": args.model,
        "endpoints": {},
        "samples": [],
        "summary": [],
        "boundary": "HTTP client observations, not NPU kernel timing or quality eval",
        "cache_state": ("cold=fresh salt; hot=completed prime with same salt; actual hit unknown"),
        "missing_metrics": (
            "Join X-Request-Id to logs for cache hits, producer cost, "
            "scheduler tokens, preemption and restore evidence; "
            "missing values remain null"
        ),
        "generation_metric": (
            "TTFT=first content arrival; "
            "TPOT estimate=(last-first)/(usage.completion_tokens-1); "
            "chunks may contain multiple tokens"
        ),
    }
    try:
        run(args, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(args.output.resolve()),
                    "summary": report["summary"],
                },
                indent=2,
            )
        )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
