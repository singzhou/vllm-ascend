# SPDX-License-Identifier: Apache-2.0
"""Temporary diagnostics, not a production fix. Remove with debug_issue45.py."""

import inspect
import time
from functools import wraps

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

from vllm_ascend.debug_issue45 import MAX_REQUESTS, emit, request_state, should_log


def wrap_schedule(method):
    @wraps(method)
    def schedule(self, *args, **kwargs):
        selected, call = should_log(self, "schedule")
        if selected:
            config = self.vllm_config
            emit(
                "SCHEDULE_BEGIN",
                call=call,
                scheduler=type(self).__name__,
                async_scheduling=config.scheduler_config.async_scheduling,
                batch_tokens=config.scheduler_config.max_num_batched_tokens,
                concurrent_batches=config.max_concurrent_batches,
                max_in_flight_tokens=config.max_in_flight_tokens,
                waiting=len(self.waiting),
                running=len(self.running),
            )
        started = time.monotonic()
        result = method(self, *args, **kwargs)
        if selected or result.total_num_scheduled_tokens or result.preempted_req_ids:
            emit(
                "SCHEDULE_END",
                call=call,
                step=self.current_step,
                host_ms=round((time.monotonic() - started) * 1000, 3),
                tokens=result.total_num_scheduled_tokens,
                scheduled=result.num_scheduled_tokens,
                preempted=result.preempted_req_ids,
                waiting=len(self.waiting),
                running=len(self.running),
                free_blocks=self.kv_cache_manager.block_pool.get_num_free_blocks(),
                requests=[request_state(r) for r in list(self.requests.values())[:MAX_REQUESTS]],
            )
        return result

    return schedule


def wrap_block_count(method):
    signature = inspect.signature(method)

    @wraps(method)
    def count(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        values = signature.bind(self, *args, **kwargs)
        values.apply_defaults()
        if hasattr(self, "_issue45_allocation_checks"):
            self._issue45_allocation_checks.append(
                {
                    "blocks": result,
                    "admission": values.arguments["apply_admission_cap"],
                    "num_tokens": values.arguments["num_tokens"],
                }
            )
        return result

    return count


def wrap_allocate(method):
    signature = inspect.signature(method)

    @wraps(method)
    def allocate(self, *args, **kwargs):
        values = signature.bind(self, *args, **kwargs)
        values.apply_defaults()
        request = values.arguments["request"]
        before = self.block_pool.get_num_free_blocks()
        self.coordinator._issue45_allocation_checks = []
        result = method(self, *args, **kwargs)
        failed = result is None
        selected, call = should_log(self, "allocate_fail" if failed else "allocate_ok")
        if selected:
            emit(
                "KV_ALLOC_FAIL" if failed else "KV_ALLOC_OK",
                call=call,
                request=request_state(request),
                free_before=before,
                free_after=self.block_pool.get_num_free_blocks(),
                checks=self.coordinator._issue45_allocation_checks,
                new_tokens=values.arguments["num_new_tokens"],
                lookahead=values.arguments["num_lookahead_tokens"],
                full_sequence_must_fit=values.arguments["full_sequence_must_fit"],
                reserved_blocks=values.arguments["reserved_blocks"],
                watermark_blocks=getattr(self, "watermark_blocks", None),
            )
        return result

    return allocate


def install():
    # The async class keeps the original base even if the public Scheduler
    # name has been replaced by BalanceScheduler. Its disabled balance path
    # also delegates here; avoid wrapping both and logging nested calls twice.
    owner = next(cls for cls in AsyncScheduler.__mro__ if "schedule" in cls.__dict__)
    owner.schedule = wrap_schedule(owner.schedule)
    KVCacheCoordinator.get_num_blocks_to_allocate = wrap_block_count(KVCacheCoordinator.get_num_blocks_to_allocate)
    KVCacheManager.allocate_slots = wrap_allocate(KVCacheManager.allocate_slots)


install()
