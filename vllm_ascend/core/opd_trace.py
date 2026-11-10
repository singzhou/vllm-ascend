# SPDX-License-Identifier: Apache-2.0
"""Opt-in v0.30 scheduler trace for first-rejection replay."""

import hashlib
import json
import os

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

logger = init_logger(__name__)


class _OPDTrace:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._draft_revision = os.environ["SPECFORGE_OPD_DRAFT_REVISION"]
        spec = self.vllm_config.speculative_config
        if not self._draft_revision or self.spec_decode_metrics_level != "detailed":
            raise ValueError("OPD trace requires a draft revision and detailed per-request metrics")
        if spec is None or spec.method != "dspark" or spec.num_speculative_tokens != 7:
            raise ValueError("OPD trace expects DSpark with seven proposals")
        self._sample_from_anchor = getattr(spec.draft_model_config.hf_config, "sample_from_anchor", True)

    def update_from_output(self, scheduler_output, model_runner_output):
        pending = []
        for rid in scheduler_output.num_scheduled_tokens:
            req = self.requests.get(rid)
            if req is None or not req.sampling_params:
                continue
            trace_id = (req.sampling_params.extra_args or {}).get("opd_trace_id")
            if trace_id and req.spec_decode_metrics is not None:
                pending.append((req, trace_id, req.num_output_tokens, len(req.spec_decode_metrics.per_step_drafted)))
        result = super().update_from_output(scheduler_output, model_runner_output)
        # Hold request references across the update: the scheduler can free
        # finished requests before returning. Inspect its actual stop-trimmed IDs.
        for req, trace_id, offset, step in pending:
            ids = list(req.output_token_ids[offset:])
            if not ids:
                continue
            m = req.spec_decode_metrics
            verified = len(m.per_step_drafted) == step + 1
            event = {
                "request_id": req.request_id,
                "trace_id": trace_id,
                "draft_revision": self._draft_revision,
                "sample_from_anchor": self._sample_from_anchor,
                "offset": offset,
                "token_ids": ids,
                "drafted": m.per_step_drafted[step] if verified else None,
                "accepted": m.per_step_accepted[step] if verified else None,
                "invalid": (scheduler_output.num_invalid_spec_tokens or {}).get(req.request_id, 0),
                "preemptions": req.num_preemptions,
                "stopped": req.is_finished(),
            }
            if offset == 0:
                prompt = json.dumps(req.prompt_token_ids, separators=(",", ":")).encode()
                event["prompt_digest"] = hashlib.sha256(prompt).hexdigest()
            logger.info("DSPARK_OPD %s", json.dumps(event, separators=(",", ":")))
        return result


class OPDTraceScheduler(_OPDTrace, Scheduler):
    """Explicit --scheduler-cls for synchronous scheduling."""


class OPDTraceAsyncScheduler(_OPDTrace, AsyncScheduler):
    """Explicit --scheduler-cls for asynchronous scheduling."""
