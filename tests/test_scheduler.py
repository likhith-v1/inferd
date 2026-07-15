"""Unit tests for ``core.scheduler.ContinuousBatchScheduler`` (fake backend)."""

import unittest
from unittest.mock import patch

import torch

from core.scheduler import (
    ContinuousBatchScheduler,
    RequestStatus,
    SchedulerConfig,
)
from core.spec_decode import nucleus_probs


class FakeBackend:
    eos_token_id = 99
    device = "cpu"

    def __init__(self):
        self.prefills = 0
        self.decodes = 0

    def prefill(self, prompt_ids):
        self.prefills += 1
        return self._logits(10), {"seen": list(prompt_ids)}

    def decode(self, token_id, kv):
        self.decodes += 1
        kv["seen"].append(token_id)
        return self._logits(10 + len(kv["seen"])), kv

    def decode_batch(self, token_ids, kvs):
        logits_list, kv_list = [], []
        for token_id, kv in zip(token_ids, kvs):
            logits, kv = self.decode(token_id, kv)
            logits_list.append(logits)
            kv_list.append(kv)
        return logits_list, kv_list

    def _logits(self, token_id):
        logits = torch.full((1, 128), -1000.0)
        logits[0, token_id] = 1000.0
        return logits


class SchedulerTest(unittest.TestCase):
    def test_fcfs_admission_respects_block_budget(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=4, block_size=4, max_concurrent_sequences=8),
        )
        for rid in range(1, 4):
            scheduler.submit([1, 2], max_tokens=6, request_id=rid)

        metrics = scheduler.step()

        self.assertEqual(metrics.admitted_sequences, 2)
        self.assertEqual(metrics.completed_sequences, 0)
        self.assertEqual(metrics.waiting_sequences, 1)
        self.assertEqual(metrics.used_blocks, 4)
        self.assertEqual(scheduler.get(1).status, RequestStatus.RUNNING)
        self.assertEqual(scheduler.get(2).status, RequestStatus.RUNNING)
        self.assertEqual(scheduler.get(3).status, RequestStatus.WAITING)

    def test_finished_sequences_evict_and_free_reserved_blocks(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=2, block_size=4, max_concurrent_sequences=1),
        )
        scheduler.submit([1, 2], max_tokens=1, request_id=1)
        scheduler.submit([3, 4], max_tokens=1, request_id=2)

        first = scheduler.step()
        self.assertEqual(first.completed_sequences, 1)
        self.assertEqual(first.evicted_sequences, 1)
        self.assertEqual(first.active_sequences, 1)
        self.assertEqual(first.used_blocks, 1)
        self.assertEqual(scheduler.get(1).status, RequestStatus.COMPLETED)
        self.assertEqual(scheduler.get(2).status, RequestStatus.RUNNING)

        second = scheduler.step()
        self.assertEqual(second.completed_sequences, 2)
        self.assertEqual(second.evicted_sequences, 2)
        self.assertEqual(second.active_sequences, 0)
        self.assertEqual(second.used_blocks, 0)
        self.assertEqual(second.free_blocks, 2)

    def test_request_too_large_fails_instead_of_deadlocking(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=2, block_size=4, max_concurrent_sequences=2),
        )
        scheduler.submit([1, 2, 3, 4, 5], max_tokens=8, request_id=1)
        scheduler.submit([1], max_tokens=1, request_id=2)

        metrics = scheduler.step()

        self.assertEqual(metrics.failed_sequences, 1)
        self.assertEqual(metrics.admitted_sequences, 1)
        self.assertEqual(scheduler.get(1).status, RequestStatus.FAILED)
        self.assertIn("needs", scheduler.get(1).error)
        self.assertEqual(scheduler.get(2).status, RequestStatus.COMPLETED)

    def test_head_of_line_waiting_request_blocks_later_admission(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=3, block_size=4, max_concurrent_sequences=3),
        )
        scheduler.submit([1], max_tokens=7, request_id=1)  # reserves 2 blocks
        scheduler.step()
        scheduler.submit([2], max_tokens=7, request_id=2)  # needs 2, cannot fit yet
        scheduler.submit([3], max_tokens=1, request_id=3)  # could fit, but waits behind #2

        metrics = scheduler.step()

        self.assertEqual(metrics.admitted_sequences, 1)
        self.assertEqual(metrics.waiting_sequences, 2)
        self.assertEqual(scheduler.get(2).status, RequestStatus.WAITING)
        self.assertEqual(scheduler.get(3).status, RequestStatus.WAITING)

    def test_run_until_complete_does_not_lose_requests(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=4, block_size=4, max_concurrent_sequences=2),
        )
        for rid in range(1, 6):
            scheduler.submit([rid], max_tokens=2, request_id=rid)

        done = scheduler.run_until_complete()
        statuses = {req.request_id: req.status for req in done}
        metrics = scheduler.metrics_snapshot()

        self.assertEqual(set(statuses), {1, 2, 3, 4, 5})
        self.assertTrue(all(status == RequestStatus.COMPLETED for status in statuses.values()))
        self.assertEqual(metrics.completed_sequences, 5)
        self.assertEqual(metrics.failed_sequences, 0)
        self.assertEqual(metrics.waiting_sequences, 0)
        self.assertEqual(metrics.active_sequences, 0)
        self.assertLessEqual(metrics.max_blocks_used, 4)
        self.assertEqual(metrics.total_generated_tokens, 10)

    def test_max_model_len_failure_is_reported(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=10, block_size=4, max_model_len=4),
        )
        scheduler.submit([1, 2, 3], max_tokens=2, request_id=1)

        metrics = scheduler.step()

        self.assertEqual(metrics.failed_sequences, 1)
        self.assertEqual(metrics.active_sequences, 0)
        self.assertEqual(scheduler.get(1).status, RequestStatus.FAILED)
        self.assertIn("max_model_len", scheduler.get(1).error)

    def test_cancel_running_frees_blocks(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=4, block_size=4, max_concurrent_sequences=2),
        )
        scheduler.submit([1, 2], max_tokens=8, request_id=1)  # reserves ceil(10/4)=3 blocks
        scheduler.step()  # admit + start running (reserves blocks)
        self.assertEqual(scheduler.get(1).status, RequestStatus.RUNNING)
        used_before = scheduler.metrics_snapshot().used_blocks
        self.assertGreater(used_before, 0)

        cancelled = scheduler.cancel(1)

        self.assertTrue(cancelled)
        self.assertEqual(scheduler.get(1).status, RequestStatus.CANCELLED)
        metrics = scheduler.metrics_snapshot()
        self.assertEqual(metrics.active_sequences, 0)
        self.assertEqual(metrics.used_blocks, 0)  # blocks freed immediately
        self.assertEqual(metrics.free_blocks, 4)

    def test_cancel_waiting_request(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=1, block_size=4, max_concurrent_sequences=1),
        )
        scheduler.submit([1, 2], max_tokens=2, request_id=1)  # running after step
        scheduler.submit([3, 4], max_tokens=2, request_id=2)  # stays waiting
        scheduler.step()
        self.assertEqual(scheduler.get(2).status, RequestStatus.WAITING)

        self.assertTrue(scheduler.cancel(2))
        self.assertEqual(scheduler.get(2).status, RequestStatus.CANCELLED)
        self.assertEqual(scheduler.metrics_snapshot().waiting_sequences, 0)

    def test_submit_resolves_per_request_sampling_against_config_default(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=10, block_size=4, temperature=0.0, top_p=1.0),
        )
        scheduler.submit([1], max_tokens=5, request_id=1)
        scheduler.submit([1], max_tokens=5, request_id=2, temperature=0.8, top_p=0.5)

        self.assertEqual(scheduler.get(1).temperature, 0.0)
        self.assertEqual(scheduler.get(1).top_p, 1.0)
        self.assertEqual(scheduler.get(2).temperature, 0.8)
        self.assertEqual(scheduler.get(2).top_p, 0.5)

    def test_sample_next_reads_per_request_not_shared_config(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=10, block_size=4, temperature=0.0, top_p=1.0),
        )
        scheduler.submit([1], max_tokens=5, request_id=1)
        scheduler.submit([1], max_tokens=5, request_id=2, temperature=0.8, top_p=0.5)

        with patch("core.scheduler.nucleus_probs", wraps=nucleus_probs) as spy:
            scheduler.step()

        calls = {call.args[1:] for call in spy.call_args_list}
        self.assertIn((0.0, 1.0), calls)
        self.assertIn((0.8, 0.5), calls)

    def test_cancel_unknown_or_terminal_returns_false(self):
        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=4, block_size=4),
        )
        self.assertFalse(scheduler.cancel(999))  # never submitted
        scheduler.submit([1], max_tokens=1, request_id=1)
        scheduler.run_until_complete()  # completes
        self.assertFalse(scheduler.cancel(1))  # already terminal


if __name__ == "__main__":
    unittest.main()
