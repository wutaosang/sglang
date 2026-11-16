"""NPU stream isolation and scheduler consumption of the WAR read-done event."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import allocate_distinct_stream
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODULE = "sglang.srt.managers.scheduler"


@pytest.mark.parametrize("handle_name", ["cuda_stream", "npu_stream"])
def test_pooled_streams_skip_every_excluded_native_handle(handle_name):
    streams = [SimpleNamespace(**{handle_name: i}) for i in (0, 1, 0, 2)]
    module = SimpleNamespace(Stream=Mock(side_effect=streams))
    chosen = allocate_distinct_stream(module, streams[:2], handle_name=handle_name)
    assert chosen is streams[-1]


@pytest.mark.parametrize("initial_handle", [0, 1, 2])
def test_npu_scheduler_avoids_forward_and_copy_streams(initial_handle):
    forward = SimpleNamespace(npu_stream=0)
    copy = SimpleNamespace(npu_stream=1)
    independent = SimpleNamespace(npu_stream=2)
    module = SimpleNamespace(
        Stream=Mock(side_effect=[forward, copy, independent]),
        StreamContext=lambda _: nullcontext(),
    )
    scheduler = SimpleNamespace(
        device="npu",
        enable_overlap=True,
        schedule_stream=SimpleNamespace(npu_stream=initial_handle),
        forward_stream=forward,
        copy_stream=copy,
        device_module=module,
        _gpu_work_on_forward_stream=lambda: False,
        metrics_reporter=Mock(),
    )
    with (
        patch(f"{MODULE}._is_npu", True),
        patch(f"{MODULE}._is_hip", False),
        patch(f"{MODULE}.is_cuda", return_value=False),
        patch(f"{MODULE}.use_mlx", return_value=False),
        patch(f"{MODULE}.triton_load_watch"),
        patch(f"{MODULE}.is_war_barrier_enabled", return_value=True),
        patch(f"{MODULE}.dispatch_event_loop") as dispatch,
    ):
        Scheduler.run_event_loop(scheduler)
    assert scheduler.schedule_stream.npu_stream == 2
    assert scheduler._war_barrier_enabled
    dispatch.assert_called_once_with(scheduler)
    if initial_handle == 2:
        module.Stream.assert_not_called()


@pytest.mark.parametrize("force_coarse", [False, True])
def test_read_done_is_consumed_once_and_missing_events_fall_back(force_coarse):
    event = object()
    runner = SimpleNamespace(shared_read_done_event=event)
    scheduler = SimpleNamespace(
        _war_barrier_enabled=True,
        model_worker=SimpleNamespace(last_shared_read_runner=runner),
        schedule_stream=Mock(),
        forward_stream=SimpleNamespace(npu_stream=9),
    )
    with envs.SGLANG_FORCE_COARSE_WAR_BARRIER.override(force_coarse):
        Scheduler._apply_war_barrier(scheduler)
        assert runner.shared_read_done_event is None
        if force_coarse:
            scheduler.schedule_stream.wait_event.assert_not_called()
        else:
            scheduler.schedule_stream.wait_event.assert_called_once_with(event)
        # An eager or unsupported next step cannot reuse the old early event.
        Scheduler._apply_war_barrier(scheduler)
    assert scheduler.schedule_stream.wait_stream.call_count == (
        2 if force_coarse else 1
    )


def test_disabled_barrier_adds_no_stream_waits():
    scheduler = SimpleNamespace(_war_barrier_enabled=False, schedule_stream=Mock())
    Scheduler._apply_war_barrier(scheduler)
    scheduler.schedule_stream.wait_event.assert_not_called()
    scheduler.schedule_stream.wait_stream.assert_not_called()
