# SPDX-License-Identifier: Apache-2.0
"""Temporary, always-on host diagnostics for issue #45; remove after debugging.

No tensor reads, device synchronization, or new environment variables. BEGIN/END
measure host calls only. Repeated idle/failure messages are rate limited.
"""

import json
import logging
import os
import time
from functools import wraps

INITIAL_MESSAGES = 8
REPEAT_INTERVAL_SECONDS = 5.0
MAX_REQUESTS = 8


def emit(event, **fields):
    logging.getLogger("vllm.issue45").info(
        "ISSUE45 %s",
        json.dumps({"event": event, "pid": os.getpid(), "time_ns": time.time_ns(), **fields}, default=str),
    )


def should_log(owner, key, force=False):
    state = getattr(owner, "_issue45_log_state", None)
    if state is None:
        state = {}
        owner._issue45_log_state = state
    count, last = state.get(key, (0, 0.0))
    count += 1
    now = time.monotonic()
    selected = force or count <= INITIAL_MESSAGES or now - last >= REPEAT_INTERVAL_SECONDS
    state[key] = (count, now if selected else last)
    return selected, count


def request_state(request):
    return {
        name: getattr(request, name, None)
        for name in (
            "request_id",
            "status",
            "num_prompt_tokens",
            "num_computed_tokens",
            "num_in_flight_tokens",
            "num_output_placeholders",
            "is_prefill_chunk",
            "next_decode_eligible_step",
            "num_preemptions",
        )
    }


def trace_phase(method):
    @wraps(method)
    def traced(self, *args, **kwargs):
        phase = method.__qualname__
        scheduler_output = args[0] if args else kwargs.get("scheduler_output")
        if hasattr(scheduler_output, "total_num_scheduled_tokens"):
            self._issue45_tokens = scheduler_output.total_num_scheduled_tokens
        elif hasattr(scheduler_output, "num_tokens"):
            self._issue45_tokens = scheduler_output.num_tokens
        tokens = getattr(self, "_issue45_tokens", None)
        selected, call = should_log(self, phase, force=bool(tokens))
        fields = {"phase": phase, "call": call, "tokens": tokens}
        started = time.monotonic()
        if selected:
            emit("PHASE_BEGIN", **fields)
        try:
            result = method(self, *args, **kwargs)
        except Exception as exc:
            emit("PHASE_ERROR", **fields, error=repr(exc))
            raise
        if selected:
            emit("PHASE_END", **fields, host_ms=round((time.monotonic() - started) * 1000, 3))
        return result

    return traced
