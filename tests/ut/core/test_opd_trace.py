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
            use_v2_model_runner=getattr(request, "v2", True),
            load_config=SimpleNamespace(load_format=getattr(request, "load_format", "auto")),
            speculative_config=SimpleNamespace(
                method="dspark",
                draft_load_config=getattr(request, "draft_load_config", None),
                num_speculative_tokens=7,
                enable_adaptive_verification=False,
                draft_model_config=SimpleNamespace(
                    model=request.draft_dir, hf_config=SimpleNamespace(sample_from_anchor=True)
                ),
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
def test_trace_sees_filtered_commits(trace, scheduler_name, monkeypatch, caplog, tmp_path):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/core/opd_trace.py"
    stubs = {
        "vllm_ascend.core.opd_draft_digest": SimpleNamespace(
            **runpy.run_path(str(path.with_name("opd_draft_digest.py")))
        ),
        "vllm.logger": SimpleNamespace(init_logger=logging.getLogger),
        "vllm.v1.core.sched.scheduler": SimpleNamespace(Scheduler=SchedulerDouble),
        "vllm.v1.core.sched.async_scheduler": SimpleNamespace(AsyncScheduler=SchedulerDouble),
    }
    with patch.dict(sys.modules, stubs):
        module = runpy.run_path(str(path))
        cls = module[scheduler_name]
    import numpy as np
    import torch
    from safetensors.torch import save_file

    save_file({"x": torch.tensor([1, 2], dtype=torch.uint8)}, tmp_path / "model.safetensors")

    batch = SimpleNamespace(
        req_ids=["x", "y", "internal-17"],
        cu_num_logits=torch.tensor([0, 1, 9, 12, 999], dtype=torch.int32),
        cu_num_logits_np=np.array([0, 8, 16, 24]),
    )
    result = SimpleNamespace()
    wrapper = SimpleNamespace(model_runner_output=result) if scheduler_name == "OPDTraceAsyncScheduler" else result

    def sample(grammar):
        assert grammar == "grammar"
        batch.cu_num_logits.zero_()  # Batch storage can be reused during sample_tokens.
        return wrapper

    assert module["sample_with_verified_widths"](batch, sample, "grammar") is wrapper
    assert result.opd_verify_widths == {"x": 0, "y": 7, "internal-17": 2}
    monkeypatch.setenv("SPECFORGE_OPD_DRAFT_REVISION", "my-draft")
    req = SimpleNamespace(
        draft_dir=tmp_path,
        request_id="internal-17",
        sampling_params=SimpleNamespace(eos_token_id=8, extra_args={"opd_trace_id": "unique"} if trace else None),
        status=SimpleNamespace(name="FINISHED_STOPPED"),
        stop_reason=None,
        num_output_tokens=0,
        spec_decode_metrics=SimpleNamespace(per_step_drafted=[], per_step_accepted=[]),
        prompt_token_ids=[4, 2],
        output_token_ids=[],
        num_preemptions=0,
    )
    req.is_finished = lambda: req.stopped
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="INFO"):
        cls(req)
    with caplog.at_level(logging.INFO):
        monkeypatch.delenv("SPECFORGE_OPD_DRAFT_REVISION")
        with pytest.raises(ValueError, match="SPECFORGE_OPD_DRAFT_REVISION"):
            cls(req)
        monkeypatch.setenv("SPECFORGE_OPD_DRAFT_REVISION", "my-draft")
        scheduler = cls(req)
        if scheduler_name == "OPDTraceScheduler":
            for req.v2 in (True, False):
                for global_format, draft_format in (("dummy", "safetensors"), ("safetensors", "dummy")):
                    req.load_format = global_format
                    req.draft_load_config = SimpleNamespace(load_format=draft_format)
                    if (global_format if req.v2 else draft_format) == "dummy":
                        with pytest.raises(ValueError, match="safetensors loader"):
                            cls(req)
                    else:
                        cls(req)
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
        assert [e["stopped_on_eos"] for e in events] == [False, False, False, True]
        assert events[-1]["stopped"] and events[0]["draft_revision"] == "my-draft"
        assert events[0]["prompt_digest"] == hashlib.sha256(b"[4,2]").hexdigest()
        vector = {"x": ["torch.uint8", [2], hashlib.sha256(bytes([1, 2])).hexdigest()]}
        assert (
            events[0]["draft_weights_sha256"]
            == hashlib.sha256(json.dumps(vector, separators=(",", ":")).encode()).hexdigest()
        )
    else:
        assert events == []


def test_platform_scheduler_override_guard():
    module = runpy.run_path(str(Path(__file__).resolve().parents[3] / "vllm_ascend/core/opd_trace_config.py"))
    validate = module["validate_opd_scheduler"]
    validate(None, None)
    for name in ("OPDTraceScheduler", "OPDTraceAsyncScheduler"):
        requested = "vllm_ascend.core.opd_trace." + name
        cls = type(name, (), {"__module__": "vllm_ascend.core.opd_trace"})
        validate(requested, cls)
        assert module["is_opd_scheduler"](cls)
        for effective in (None, "vllm_ascend.core.recompute_scheduler.RecomputeScheduler"):
            with pytest.raises(ValueError, match="was replaced"):
                validate(cls, effective)
