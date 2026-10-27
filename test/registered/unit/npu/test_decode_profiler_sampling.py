"""CPU regressions for the scheduler's opt-in decode profiler hooks."""

import ast
import os
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SCHEDULER_PATH = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/scheduler.py"
)


def load_profiler_hooks(torch, clock, logger):
    # Exercise the actual hooks without importing the scheduler's GPU kernels,
    # model backends, distributed groups, or an installed torch_npu package.
    tree = ast.parse(SCHEDULER_PATH.read_text())
    functions = {
        "_positive_profiler_int",
        "_prof_bs",
        "_finish_decode_profile",
        "_sample_decode_step",
        "_time_decode_step",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and (target.id.startswith("_PROF_") or target.id == "_T_STEP")
            for target in node.targets
        ):
            nodes.append(node)
    namespace = {"os": os, "torch": torch, "time": clock, "logger": logger}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(SCHEDULER_PATH), "exec"),
        namespace,
    )
    return SimpleNamespace(**namespace)


def make_batch(size=4, decode=True):
    return SimpleNamespace(
        reqs=[object() for _ in range(size)],
        batch_size=lambda: size,
        forward_mode=SimpleNamespace(is_decode=lambda: decode),
    )


class TestDecodeProfilerSampling(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {"W0_PROF_DIR": "/unused", "W0_PROF_BS": "4", "W0_PROF_STEPS": "20"},
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.torch = SimpleNamespace(npu=SimpleNamespace(synchronize=Mock()))
        self.clock = SimpleNamespace(perf_counter=Mock())
        self.logger = Mock()
        self.hooks = load_profiler_hooks(self.torch, self.clock, self.logger)
        self.profiler = Mock()
        self.profile = self.profiler.profile.return_value
        self.trace_ready = Mock()
        self.profiler.tensorboard_trace_handler.return_value = self.trace_ready
        self.profile.stop.side_effect = lambda: self.trace_ready(self.profile)
        self.modules = patch.dict(
            "sys.modules", {"torch_npu": SimpleNamespace(profiler=self.profiler)}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def sample(self, batch):
        self.hooks._sample_decode_step(batch)

    def test_disabled_hooks_need_no_npu_or_clock(self):
        os.environ.pop("W0_PROF_DIR")
        with patch.dict("sys.modules", {"torch_npu": None}):
            self.sample(make_batch())
            self.hooks._time_decode_step(make_batch())
        self.profiler.profile.assert_not_called()
        self.torch.npu.synchronize.assert_not_called()
        self.clock.perf_counter.assert_not_called()

    def test_only_eligible_decode_batch_arms_sampler(self):
        self.sample(None)
        self.sample(make_batch(decode=False))
        self.sample(make_batch(size=3))
        self.profiler.profile.assert_not_called()
        self.sample(make_batch())
        self.profile.start.assert_called_once_with()

    def test_one_window_exports_once_despite_later_decode_batches(self):
        batch = make_batch()
        for _ in range(1000):
            self.sample(batch)
        self.profiler.profile.assert_called_once()
        self.profile.start.assert_called_once_with()
        self.assertEqual(self.profile.step.call_count, 20)
        self.profile.stop.assert_called_once_with()
        self.trace_ready.assert_called_once_with(self.profile)
        self.torch.npu.synchronize.assert_called_once_with()
        self.assertTrue(self.hooks._PROF_DONE[0])
        config = self.profiler._ExperimentalConfig.call_args.kwargs
        self.assertTrue(config["data_simplification"])

    def test_window_limit_is_fixed_when_recording_starts(self):
        batch = make_batch()
        self.sample(batch)
        os.environ["W0_PROF_STEPS"] = "1"
        for _ in range(19):
            self.sample(batch)
        self.profile.stop.assert_not_called()
        self.sample(batch)
        self.profile.stop.assert_called_once_with()

    def test_idle_flushes_short_window_and_cannot_rearm(self):
        for _ in range(5):
            self.sample(make_batch())
        self.sample(None)
        self.assertEqual(self.profile.step.call_count, 5)
        self.profile.stop.assert_called_once_with()
        self.logger.warning.assert_called_with(
            "PROF_DONE steps=%s reason=%s", 5, "idle"
        )
        for _ in range(100):
            self.sample(make_batch())
        self.profiler.profile.assert_called_once()
        self.trace_ready.assert_called_once()

    def test_prefill_flushes_before_non_decode_work(self):
        self.sample(make_batch())
        self.sample(make_batch(decode=False))
        self.profile.step.assert_called_once_with()
        self.profile.stop.assert_called_once_with()
        self.logger.warning.assert_called_with(
            "PROF_DONE steps=%s reason=%s", 1, "non_decode"
        )

    def test_disabling_sampler_flushes_active_window(self):
        self.sample(make_batch())
        os.environ.pop("W0_PROF_DIR")
        self.sample(make_batch())
        self.profile.stop.assert_called_once_with()
        self.logger.warning.assert_called_with(
            "PROF_DONE steps=%s reason=%s", 1, "disabled"
        )

    def test_bad_step_limits_fail_before_starting_profiler(self):
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value):
                os.environ["W0_PROF_STEPS"] = value
                with self.assertRaisesRegex(ValueError, "W0_PROF_STEPS.*positive"):
                    self.sample(make_batch())
        self.profiler.profile.assert_not_called()

    def test_bad_batch_thresholds_fail_before_starting_profiler(self):
        for value in ("0", "-1", "invalid"):
            with self.subTest(value=value):
                os.environ["W0_PROF_BS"] = value
                with self.assertRaisesRegex(ValueError, "W0_PROF_BS.*positive"):
                    self.sample(make_batch())
        self.profiler.profile.assert_not_called()

    def test_default_threshold_uses_decode_graph_batch_sizes(self):
        os.environ.pop("W0_PROF_BS")
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(cuda_graph_bs_decode=[1, 2, 8])
        )
        self.hooks._sample_decode_step(make_batch(size=4), scheduler)
        self.profiler.profile.assert_not_called()
        self.hooks._sample_decode_step(make_batch(size=8), scheduler)
        self.profile.start.assert_called_once_with()

    def test_timer_reports_twenty_intervals_and_integer_batch_size(self):
        self.clock.perf_counter.side_effect = [index * 0.01 for index in range(41)]
        for _ in range(41):
            self.hooks._time_decode_step(make_batch())
        self.assertEqual(self.logger.warning.call_count, 2)
        for call in self.logger.warning.call_args_list:
            self.assertAlmostEqual(call.args[1], 10.0)
            self.assertEqual(call.args[2:], (4, 4))

    def test_timer_excludes_idle_or_prefill_gaps(self):
        for boundary in (None, make_batch(decode=False)):
            with self.subTest(boundary=boundary):
                self.logger.reset_mock()
                hooks = load_profiler_hooks(self.torch, self.clock, self.logger)
                self.clock.perf_counter.side_effect = [
                    index * 0.01 for index in range(10)
                ] + [100 + index * 0.01 for index in range(21)]
                for _ in range(10):
                    hooks._time_decode_step(make_batch())
                hooks._time_decode_step(boundary)
                for _ in range(21):
                    hooks._time_decode_step(make_batch())
                self.logger.warning.assert_called_once()
                self.assertAlmostEqual(self.logger.warning.call_args.args[1], 10.0)

    def exercise_scheduler_idle(self, loop_name):
        tree = ast.parse(SCHEDULER_PATH.read_text())
        scheduler_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
        )
        loop = next(
            node
            for node in scheduler_class.body
            if isinstance(node, ast.FunctionDef) and node.name == loop_name
        )
        loop.decorator_list = []
        namespace = self.hooks._sample_decode_step.__globals__
        namespace.update(
            deque=deque,
            envs=SimpleNamespace(
                SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=SimpleNamespace(
                    get=lambda: False
                )
            ),
        )
        exec(
            compile(
                ast.Module(body=[loop], type_ignores=[]), str(SCHEDULER_PATH), "exec"
            ),
            namespace,
        )
        batch = make_batch()
        batch.copy = lambda: batch
        plans = iter([batch, None, None])
        scheduler = SimpleNamespace(
            gracefully_exit=False,
            _engine_paused=False,
            enable_unified_memory=False,
            is_generation=False,
            running_batch=None,
            last_batch=None,
            ingest_requests=Mock(),
            get_next_batch_to_run=lambda **kwargs: SimpleNamespace(
                running_batch=None, batch_to_run=next(plans)
            ),
            run_batch=Mock(return_value=object()),
            process_batch_result=Mock(),
            is_disable_overlap_for_batch=Mock(return_value=False),
            _apply_war_barrier=Mock(),
        )
        scheduler.on_idle = lambda: setattr(scheduler, "gracefully_exit", True)
        namespace[loop_name](scheduler)
        self.profile.stop.assert_called_once_with()
        self.trace_ready.assert_called_once()
        self.logger.warning.assert_called_with(
            "PROF_DONE steps=%s reason=%s", 1, "idle"
        )

    def test_normal_loop_flushes_when_no_batch_remains(self):
        self.exercise_scheduler_idle("event_loop_normal")

    def test_overlap_loop_flushes_before_draining_last_result(self):
        self.exercise_scheduler_idle("event_loop_overlap")


if __name__ == "__main__":
    unittest.main()
