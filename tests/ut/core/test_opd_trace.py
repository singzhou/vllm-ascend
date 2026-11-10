# SPDX-License-Identifier: Apache-2.0
"""Exercise the trace wrapper without initializing the NPU runtime."""

import hashlib
import json
import logging
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


class SchedulerDouble:
    def __init__(self, request):
        self.scheduler_config = SimpleNamespace(async_scheduling=type(self).__name__ == "OPDTraceAsyncScheduler")
        self.requests = {request.request_id: request}
        self.spec_decode_metrics_level = "detailed"
        self.vllm_config = SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=SimpleNamespace(
                method="dspark",
                num_speculative_tokens=7,
                enable_adaptive_verification=False,
                draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(sample_from_anchor=True)),
            ),
        )

    def update_from_output(self, scheduler_output, model_runner_output):
        req = next(iter(self.requests.values()))
        ids, verified, req.stopped = model_runner_output.step
        if verified:
            d, r = verified
            req.spec_decode_metrics.per_step_drafted.append(d)
            req.spec_decode_metrics.per_step_accepted.append(r)
        req.output_token_ids.extend(ids)
        req.num_output_tokens += len(ids)
        if req.stopped:
            self.requests.clear()
        return model_runner_output


@pytest.mark.parametrize("trace", [True, False])
@pytest.mark.parametrize("scheduler_name", ["OPDTraceScheduler", "OPDTraceAsyncScheduler"])
def test_trace_sees_filtered_commits(trace, scheduler_name, monkeypatch, caplog):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/core/opd_trace.py"
    stubs = {
        "vllm.logger": SimpleNamespace(init_logger=logging.getLogger),
        "vllm.v1.core.sched.scheduler": SimpleNamespace(Scheduler=SchedulerDouble),
        "vllm.v1.core.sched.async_scheduler": SimpleNamespace(AsyncScheduler=SchedulerDouble),
    }
    with patch.dict(sys.modules, stubs):
        module = runpy.run_path(str(path))
        cls = module[scheduler_name]
    import numpy as np

    batch = SimpleNamespace(
        req_ids=["x", "y", "internal-17"],
        cu_num_logits=np.array([0, 1, 9, 12]),
        cu_num_logits_np=np.array([0, 8, 16, 24]),
    )
    assert module["effective_draft_widths"](batch) == {"x": 0, "y": 7, "internal-17": 2}
    monkeypatch.setenv("SPECFORGE_OPD_DRAFT_REVISION", "my-draft")
    req = SimpleNamespace(
        request_id="internal-17",
        sampling_params=SimpleNamespace(extra_args={"opd_trace_id": "unique"} if trace else None),
        num_output_tokens=0,
        spec_decode_metrics=SimpleNamespace(per_step_drafted=[], per_step_accepted=[]),
        prompt_token_ids=[4, 2],
        output_token_ids=[],
        num_preemptions=0,
    )
    req.is_finished = lambda: req.stopped
    scheduler = cls(req)
    sched = SimpleNamespace(num_scheduled_tokens={"internal-17": 1}, num_invalid_spec_tokens=None)
    with caplog.at_level(logging.INFO):
        for update in [([3], None, False), ([1, 2], (7, 1), False), ([6], None, False), ([8], (7, 3), True)]:
            scheduler._runner_widths = True
            output = SimpleNamespace(step=update, opd_verify_widths={"internal-17": 2 if update[0] == [1, 2] else 7})
            assert scheduler.update_from_output(sched, output) is output
    events = [json.loads(r.message.split("DSPARK_OPD ", 1)[1]) for r in caplog.records if "DSPARK_OPD " in r.message]
    if trace:
        assert [e["offset"] for e in events] == [0, 1, 3, 4]
        assert [e["verified_width"] for e in events] == [None, 2, None, 7]
        assert [e["accepted"] for e in events] == [None, 1, None, 3]
        assert [e["token_ids"] for e in events] == [[3], [1, 2], [6], [8]]
        assert events[-1]["stopped"] and events[0]["draft_revision"] == "my-draft"
        assert events[0]["prompt_digest"] == hashlib.sha256(b"[4,2]").hexdigest()
    else:
        assert events == []
