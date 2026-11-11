# SPDX-License-Identifier: Apache-2.0
"""Opt-in v0.30 scheduler trace for first-rejection replay."""

import hashlib
import json
import logging
import os

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.core.opd_draft_digest import safetensors_digest

logger = init_logger(__name__)


def sample_with_verified_widths(batch, sample_tokens, grammar_output):
    """Snapshot device boundaries before the sampler can reuse the input batch."""
    offsets = batch.cu_num_logits[: len(batch.req_ids) + 1].tolist()
    widths = {rid: end - start - 1 for rid, start, end in zip(batch.req_ids, offsets, offsets[1:])}
    output = sample_tokens(grammar_output)
    if output is not None:
        getattr(output, "model_runner_output", output).opd_verify_widths = widths
    return output


class _OPDTrace:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._draft_revision = os.environ.get("SPECFORGE_OPD_DRAFT_REVISION", "").strip()
        spec = self.vllm_config.speculative_config
        if not self._draft_revision or self.spec_decode_metrics_level != "detailed":
            raise ValueError("OPD trace requires SPECFORGE_OPD_DRAFT_REVISION and detailed per-request metrics")
        if not logger.isEnabledFor(logging.INFO):
            raise ValueError("OPD trace needs INFO output; check VLLM_LOGGING_LEVEL and the trace logger")
        if spec is None or spec.method != "dspark" or spec.num_speculative_tokens != 7:
            raise ValueError("OPD trace expects DSpark with seven proposals")
        if bool(self.scheduler_config.async_scheduling) != isinstance(self, OPDTraceAsyncScheduler):
            raise ValueError("OPD scheduler class must match the service's async_scheduling setting")
        if getattr(spec, "num_speculative_tokens_per_batch_size", None):
            raise ValueError("variable DSpark proposal query width is not supported by OPD replay")
        self._runner_widths = bool(spec.enable_adaptive_verification)
        if self._runner_widths and not self.vllm_config.use_v2_model_runner:
            raise ValueError("OPD runtime-width telemetry requires runner V2 for adaptive verification")
        if self.scheduler_config.async_scheduling and not self.vllm_config.use_v2_model_runner:
            from vllm_ascend.ascend_config import get_ascend_config

            if get_ascend_config().dynamic_spec_config.method == "dspark":
                raise ValueError("V1 async dynamic verification has no reliable OPD width telemetry")
        self._sample_from_anchor = getattr(spec.draft_model_config.hf_config, "sample_from_anchor", True)
        load_config = spec.draft_load_config or self.vllm_config.load_config
        if load_config.load_format not in {"auto", "hf", "safetensors", "fastsafetensors", "instanttensor"}:
            raise ValueError(f"OPD draft fingerprint needs a safetensors loader, got {load_config.load_format!r}")
        self._draft_weights_sha256 = safetensors_digest(spec.draft_model_config.model)

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
            drafted = m.per_step_drafted[step] if verified else None
            effective = drafted
            if verified and self._runner_widths:
                effective = getattr(model_runner_output, "opd_verify_widths", {}).get(req.request_id)
            event = {
                "request_id": req.request_id,
                "trace_id": trace_id,
                "draft_revision": self._draft_revision,
                "sample_from_anchor": self._sample_from_anchor,
                "offset": offset,
                "token_ids": ids,
                "drafted": drafted,
                "verified_width": effective,
                "accepted": m.per_step_accepted[step] if verified else None,
                "invalid": (scheduler_output.num_invalid_spec_tokens or {}).get(req.request_id, 0),
                "preemptions": req.num_preemptions,
                "stopped": req.is_finished(),
                "stopped_on_eos": (
                    req.is_finished()
                    and req.status.name == "FINISHED_STOPPED"
                    and req.stop_reason is None
                    and ids[-1] == req.sampling_params.eos_token_id
                ),
            }
            if offset == 0:
                event["draft_weights_sha256"] = self._draft_weights_sha256
                prompt = json.dumps(req.prompt_token_ids, separators=(",", ":")).encode()
                event["prompt_digest"] = hashlib.sha256(prompt).hexdigest()
            logger.info("DSPARK_OPD %s", json.dumps(event, separators=(",", ":")))
        return result


class OPDTraceScheduler(_OPDTrace, Scheduler):
    """Explicit --scheduler-cls for synchronous scheduling."""


class OPDTraceAsyncScheduler(_OPDTrace, AsyncScheduler):
    """Explicit --scheduler-cls for asynchronous scheduling."""
