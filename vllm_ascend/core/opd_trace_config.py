# SPDX-License-Identifier: Apache-2.0
"""Identify the effective OPD scheduler after Ascend platform overrides."""


def is_opd_scheduler(cls):
    if isinstance(cls, type):
        cls = f"{cls.__module__}.{cls.__qualname__}"
    return cls in (
        "vllm_ascend.core.opd_trace.OPDTraceScheduler",
        "vllm_ascend.core.opd_trace.OPDTraceAsyncScheduler",
    )


def validate_opd_scheduler(requested, effective):
    if is_opd_scheduler(requested) and not is_opd_scheduler(effective):
        raise ValueError(
            f"Requested OPD trace scheduler {requested!r} was replaced by {effective!r}. "
            "Check --scheduler-cls and conflicting Ascend scheduler settings."
        )
