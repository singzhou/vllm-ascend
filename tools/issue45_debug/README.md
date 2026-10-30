# Issue 45: asynchronous long-prefill diagnostics

This temporary branch enables `ISSUE45` INFO logs automatically. No extra
environment variable is required. Keep INFO logging enabled. The patch logs host
state only: it does not read device tensors, synchronize the NPU, or change
scheduling decisions. Logging can still perturb timing; this is not a benchmark
branch or a production fix. Remove the instrumentation after diagnosis.

## Collect a run

Use the original failing launch command, with asynchronous scheduling enabled,
and capture both stdout and stderr:

```bash
set -o pipefail
bash your_original_launch_script.sh 2>&1 | tee issue45-async-14336.log
```

Submit the same long request. Record its actual prompt token count and submission
time. Keep the startup logs (including KV capacity), and collect at least 30
seconds after apparent loss of progress. Preserve the full log before stopping
the server. If possible, also capture the known-good 8129-token configuration in
a separate log, using a fresh server and the same request.

```bash
rg 'ISSUE45' issue45-async-14336.log > issue45-key.log
rg 'KV_ALLOC_FAIL|SCHEDULE_END|PHASE_ERROR' issue45-key.log
```

Every record has a PID and wall-clock `time_ns`. Worker log prefixes identify
the TP rank. `call` counts include suppressed records. Idle and allocation
messages print the first eight calls and then at most once every five seconds
per category. Nonempty scheduler results and worker phases print every call.
No prompt contents or token IDs are recorded; request IDs are included.

## Interpret the records

| Event | Meaning |
| --- | --- |
| `SCHEDULE_BEGIN` | Scheduler class, async flag, batch limit, concurrency and in-flight token budget |
| `SCHEDULE_END` | Scheduled tokens, waiting/running counts, free blocks, preemptions and up to eight request states |
| `KV_ALLOC_FAIL` | Allocation returned `None`; includes actual block-count checks, admission flag, free blocks, lookahead and reservations |
| `KV_ALLOC_OK` | Allocation succeeded; comparison with failed admissions |
| `PHASE_BEGIN` / `PHASE_END` | Worker execute, input preparation, CPU length wait, postprocess, sampling and DSpark proposal host boundaries |
| `PHASE_ERROR` | Exception propagated unchanged from an instrumented worker method |

Repeated `SCHEDULE_END` records with zero tokens and increasing call counts show
empty scheduling, rather than a worker frozen inside UVA. Correlated
`KV_ALLOC_FAIL` records distinguish KV admission failure from other scheduler
gates. The `checks.blocks` count excludes the separately reported watermark and
reservations; a check marked `admission` is the full-sequence admission check.

For a nonempty worker call, a BEGIN without its matching END narrows the blocked
host phase. Match by PID, phase and call, not by global line order. END means only
that the host call returned, not that asynchronous NPU work completed. A final
unmatched BEGIN can also be caused by terminating the process or truncating logs.
The CPU-length wait is an existing synchronization; the diagnostic adds none.

If scheduling itself stops producing records, capture EngineCore's stack too.
Device-side deadlocks still require native worker stacks or device diagnostics.

## Local checks

```bash
python3 tests/ut/patch/platform/test_issue45_debug.py
bash format.sh ci
```

The standalone unit tests do not require torch, vLLM or NPU hardware. They check
rate limiting, return/exception preservation, inherited scheduler patching and
failed-allocation telemetry. Real NPU execution remains to be validated by the
reproduction run.
