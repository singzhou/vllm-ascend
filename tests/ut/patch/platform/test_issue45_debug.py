# SPDX-License-Identifier: Apache-2.0
"""CPU-only diagnostic tests; runnable directly with Python's unittest."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestIssue45Debug(unittest.TestCase):
    def setUp(self):
        self.helper = load_source("issue45_helper_test", "vllm_ascend/debug_issue45.py")

    def test_rate_limit_counts_suppressed_calls(self):
        owner = SimpleNamespace()
        with patch.object(self.helper.time, "monotonic", return_value=10):
            for _ in range(self.helper.INITIAL_MESSAGES):
                self.assertTrue(self.helper.should_log(owner, "idle")[0])
            self.assertEqual(self.helper.should_log(owner, "idle"), (False, 9))
        with patch.object(self.helper.time, "monotonic", return_value=16):
            self.assertEqual(self.helper.should_log(owner, "idle"), (True, 10))

    def test_phase_preserves_result_and_exception(self):
        sentinel = object()

        class Runner:
            def execute(self, scheduler_output):
                return sentinel

            def fail(self):
                raise ValueError("test failure")

        runner = Runner()
        with patch.object(self.helper, "emit") as emit:
            result = self.helper.trace_phase(Runner.execute)(runner, SimpleNamespace(total_num_scheduled_tokens=14336))
            self.assertIs(result, sentinel)
            self.assertEqual([call.args[0] for call in emit.call_args_list], ["PHASE_BEGIN", "PHASE_END"])
            with self.assertRaisesRegex(ValueError, "test failure"):
                self.helper.trace_phase(Runner.fail)(runner)
            self.assertEqual(emit.call_args.args[0], "PHASE_ERROR")

    def test_install_and_allocation_failure_snapshot(self):
        output = SimpleNamespace(total_num_scheduled_tokens=0, num_scheduled_tokens={}, preempted_req_ids=set())

        class Scheduler:
            def schedule(self):
                return output

        class AsyncScheduler(Scheduler):
            pass

        class Coordinator:
            def get_num_blocks_to_allocate(self, num_tokens, apply_admission_cap=False):
                return 50

        class Manager:
            def allocate_slots(
                self, request, num_new_tokens, num_lookahead_tokens=0, full_sequence_must_fit=True, reserved_blocks=0
            ):
                self.coordinator.get_num_blocks_to_allocate(num_new_tokens, apply_admission_cap=True)
                return None

        modules = {"vllm_ascend.debug_issue45": self.helper}
        for name, attr, value in (
            ("vllm.v1.core.kv_cache_coordinator", "KVCacheCoordinator", Coordinator),
            ("vllm.v1.core.kv_cache_manager", "KVCacheManager", Manager),
            ("vllm.v1.core.sched.async_scheduler", "AsyncScheduler", AsyncScheduler),
        ):
            module = ModuleType(name)
            setattr(module, attr, value)
            modules[name] = module
        with patch.dict(sys.modules, modules), patch.object(self.helper, "emit") as emit:
            load_source("issue45_patch_test", "vllm_ascend/patch/platform/patch_issue45_debug.py")
            manager = Manager()
            manager.coordinator = Coordinator()
            manager.block_pool = SimpleNamespace(get_num_free_blocks=lambda: 10)
            request = SimpleNamespace(request_id="test", num_computed_tokens=0)
            self.assertIsNone(manager.allocate_slots(request, 14336))
            self.assertEqual(emit.call_args.args[0], "KV_ALLOC_FAIL")
            self.assertEqual(emit.call_args.kwargs["checks"], [{"blocks": 50, "admission": True, "num_tokens": 14336}])
            scheduler = AsyncScheduler()
            scheduler.vllm_config = SimpleNamespace(
                scheduler_config=SimpleNamespace(async_scheduling=True, max_num_batched_tokens=14336),
                max_concurrent_batches=2,
                max_in_flight_tokens=28672,
            )
            scheduler.waiting = [request]
            scheduler.running = []
            scheduler.requests = {"test": request}
            scheduler.current_step = 1
            scheduler.kv_cache_manager = manager
            self.assertIs(scheduler.schedule(), output)
            self.assertEqual(emit.call_args.args[0], "SCHEDULE_END")
            self.assertEqual(emit.call_args.kwargs["tokens"], 0)


if __name__ == "__main__":
    unittest.main()
