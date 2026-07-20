import pytest

torch = pytest.importorskip("torch")

from backends.mlx.stub import StubBackend
from core.scheduler import ContinuousBatchScheduler, RequestStatus, SchedulerConfig


def test_real_scheduler_streams_stub_tokens_and_backfills():
    scheduler = ContinuousBatchScheduler(
        StubBackend(),
        SchedulerConfig(max_blocks=2, block_size=4, max_concurrent_sequences=1),
    )
    first = scheduler.submit([1], max_tokens=2)
    second = scheduler.submit([2], max_tokens=2)
    done = scheduler.run_until_complete()

    assert [request.request_id for request in done] == [first, second]
    assert all(request.status is RequestStatus.COMPLETED for request in done)
    assert all(request.generated_ids for request in done)
    assert scheduler.metrics_snapshot().admitted_sequences == 2
