"""
End-to-end tests for the serving layer (phase 07) with NO GPU.

A FakeEngine is injected via `create_app(engine=...)`, exercising the full
HTTP / SSE / metrics surface (routing, streaming, 429, 400, disconnect→cancel,
schema shape) without loading a model. This mirrors the FakeBackend pattern in
tests/test_scheduler.py.
"""

import asyncio
import json
import unittest

from fastapi.testclient import TestClient

from core.scheduler import ContinuousBatchScheduler, RequestStatus, SchedulerConfig, SchedulerMetrics
from serve.app import create_app
from serve.engine import Done, Engine, Error, StreamChannel, TokenChunk
from serve.schemas import MetricsResponse


class FakeEngine:
    """Minimal stand-in for serve.engine.Engine (no scheduler, no GPU)."""

    def __init__(self, *, saturated=False, violation=None, n_tokens=3, delay=0.0, fail=False):
        self.model_name = "fake-model"
        self.device = "cpu"
        self.alive = True
        self._saturated = saturated
        self._violation = violation
        self._n_tokens = n_tokens
        self._delay = delay
        self._fail = fail
        self.cancels: list[int] = []
        self._next_id = 1

    def start(self): ...
    def stop(self): ...

    def encode(self, prompt: str) -> list[int]:
        return [ord(c) % 256 for c in prompt]

    def limit_violation(self, prompt_len, max_tokens):
        return self._violation

    def submit(self, prompt_ids, max_tokens):
        if self._saturated:
            return None
        rid = self._next_id
        self._next_id += 1
        channel = StreamChannel(rid, asyncio.get_running_loop())
        asyncio.get_running_loop().create_task(self._drive(channel))
        return channel

    async def _drive(self, channel: StreamChannel):
        for i in range(self._n_tokens):
            channel.push(TokenChunk(f"t{i} "))
            if self._delay:
                await asyncio.sleep(self._delay)
        if self._fail:
            channel.push(Error("boom"))
        else:
            channel.push(Done("completed", self._n_tokens))

    def cancel(self, request_id: int):
        self.cancels.append(request_id)

    def metrics(self) -> dict:
        return {
            "waiting_sequences": 0, "active_sequences": 1, "completed_sequences": 2,
            "failed_sequences": 0, "admitted_sequences": 3, "evicted_sequences": 2,
            "iterations": 10, "total_generated_tokens": 42, "used_blocks": 4,
            "free_blocks": 28, "max_blocks_used": 6, "tokens_per_second": 12.5,
            "last_ttft_s": 0.08, "peak_vram_mb": 21000.0, "uptime_s": 3.2,
            "model": self.model_name,
        }


def _parse_sse(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in text.splitlines()
            if line.startswith("data: ")]


class ServeTest(unittest.TestCase):
    def test_generate_streams_incrementally(self):
        engine = FakeEngine(n_tokens=3)
        with TestClient(create_app(engine=engine)) as client:
            r = client.post("/generate", json={"prompt": "hi", "max_tokens": 8})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.headers["content-type"].split(";")[0], "text/event-stream")
            events = _parse_sse(r.text)
        types = [e["type"] for e in events]
        self.assertEqual(types, ["token", "token", "token", "done"])
        self.assertEqual(events[-1]["finish_reason"], "completed")
        self.assertEqual(events[-1]["tokens"], 3)
        # the stream's finally always frees blocks (idempotent cancel)
        self.assertIn(1, engine.cancels)

    def test_generation_error_is_streamed(self):
        engine = FakeEngine(n_tokens=1, fail=True)
        with TestClient(create_app(engine=engine)) as client:
            events = _parse_sse(client.post("/generate", json={"prompt": "hi"}).text)
        self.assertEqual(events[-1], {"type": "error", "message": "boom"})

    def test_saturation_returns_429(self):
        engine = FakeEngine(saturated=True)
        with TestClient(create_app(engine=engine)) as client:
            r = client.post("/generate", json={"prompt": "hi"})
        self.assertEqual(r.status_code, 429)

    def test_oversized_returns_400(self):
        engine = FakeEngine(violation="request needs 999 blocks, budget is 10")
        with TestClient(create_app(engine=engine)) as client:
            r = client.post("/generate", json={"prompt": "hi", "max_tokens": 100000})
        self.assertEqual(r.status_code, 400)
        self.assertIn("blocks", r.json()["detail"])

    def test_dead_engine_returns_503(self):
        engine = FakeEngine()
        engine.alive = False  # engine thread crashed
        with TestClient(create_app(engine=engine)) as client:
            r = client.post("/generate", json={"prompt": "hi", "max_tokens": 8})
        self.assertEqual(r.status_code, 503)

    def test_empty_prompt_rejected(self):
        with TestClient(create_app(engine=FakeEngine())) as client:
            r = client.post("/generate", json={"prompt": "", "max_tokens": 8})
        self.assertEqual(r.status_code, 422)  # pydantic min_length

    def test_disconnect_triggers_cancel(self):
        engine = FakeEngine(n_tokens=10, delay=0.2)  # slow stream
        with TestClient(create_app(engine=engine)) as client:
            with client.stream("POST", "/generate", json={"prompt": "hi"}) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        break  # got first token, disconnect early
        self.assertIn(1, engine.cancels)  # blocks freed on disconnect

    def test_metrics_shape(self):
        with TestClient(create_app(engine=FakeEngine())) as client:
            r = client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        MetricsResponse(**r.json())  # raises if the shape drifts from the contract

    def test_healthz(self):
        with TestClient(create_app(engine=FakeEngine())) as client:
            r = client.get("/healthz")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["engine_alive"])
        self.assertEqual(body["model"], "fake-model")


class _RaisingScheduler:
    """Fake scheduler whose step() raises — to test engine-thread crash handling."""

    def __init__(self):
        self._has_work = False

    def submit(self, prompt_ids, max_tokens, request_id):
        self._has_work = True

    def cancel(self, request_id):
        return False

    def get(self, request_id):
        return None

    def step(self):
        raise RuntimeError("boom (simulated CUDA OOM)")

    def metrics_snapshot(self):
        n = 1 if self._has_work else 0
        return SchedulerMetrics(
            waiting_sequences=0, active_sequences=n, completed_sequences=0,
            failed_sequences=0, admitted_sequences=0, evicted_sequences=0,
            iterations=0, total_generated_tokens=0, used_blocks=0,
            free_blocks=0, max_blocks_used=0,
        )


class _FakeTokenizer:
    def __call__(self, prompt, add_special_tokens=True):
        class Result:
            input_ids = [ord(c) % 256 for c in prompt]

        return Result()

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(i) for i in token_ids)


class EngineCancelTest(unittest.IsolatedAsyncioTestCase):
    """Cancel must unblock the SSE handler with a terminal Done event."""

    async def test_cancel_waiting_request_pushes_done(self):
        from test_scheduler import FakeBackend

        scheduler = ContinuousBatchScheduler(
            FakeBackend(),
            SchedulerConfig(max_blocks=32, block_size=4, max_concurrent_sequences=1),
        )
        eng = Engine(
            scheduler,
            _FakeTokenizer(),
            model_name="m",
            device="cpu",
            max_concurrent=1,
            max_queue_depth=4,
        )
        eng.start()
        try:
            first = eng.submit([1, 2], max_tokens=8)
            second = eng.submit([3, 4], max_tokens=8)
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            eng.cancel(second.request_id)
            item = await asyncio.wait_for(second.queue.get(), timeout=2)
            self.assertIsInstance(item, Done)
            self.assertEqual(item.finish_reason, RequestStatus.CANCELLED.value)
            self.assertEqual(item.generated_tokens, 0)
        finally:
            eng.stop()


class EngineCrashTest(unittest.IsolatedAsyncioTestCase):
    """A crash in the engine thread must error in-flight clients, not hang them."""

    async def test_step_crash_errors_channel_and_marks_dead(self):
        eng = Engine(
            _RaisingScheduler(), tokenizer=None, model_name="m", device="cpu",
            max_concurrent=4, max_queue_depth=8,
        )
        eng.start()
        try:
            channel = eng.submit([1, 2, 3], max_tokens=8)
            self.assertIsNotNone(channel)
            # The client must receive an Error (not block forever on queue.get()).
            item = await asyncio.wait_for(channel.queue.get(), timeout=5)
            self.assertIsInstance(item, Error)
            self.assertIn("engine crashed", item.message)
            await asyncio.sleep(0.05)  # let the thread finish unwinding
            self.assertFalse(eng.alive)        # /healthz would report degraded
            self.assertIsNotNone(eng._fatal)
        finally:
            eng.stop()


class _RaisingSubmitScheduler:
    """Fake scheduler whose submit() raises — a bad request must not kill the loop."""

    def submit(self, prompt_ids, max_tokens, request_id):
        raise ValueError("max_tokens must be positive")

    def cancel(self, request_id):
        return False

    def get(self, request_id):
        return None

    def step(self):
        return None

    def metrics_snapshot(self):
        return SchedulerMetrics(
            waiting_sequences=0, active_sequences=0, completed_sequences=0,
            failed_sequences=0, admitted_sequences=0, evicted_sequences=0,
            iterations=0, total_generated_tokens=0, used_blocks=0,
            free_blocks=0, max_blocks_used=0,
        )


class EngineBadSubmitTest(unittest.IsolatedAsyncioTestCase):
    """A per-request submit error must fail only that request, not the engine."""

    async def test_bad_submit_errors_channel_and_engine_survives(self):
        eng = Engine(
            _RaisingSubmitScheduler(), tokenizer=None, model_name="m", device="cpu",
            max_concurrent=4, max_queue_depth=8,
        )
        eng.start()
        try:
            channel = eng.submit([1, 2, 3], max_tokens=0)
            self.assertIsNotNone(channel)
            item = await asyncio.wait_for(channel.queue.get(), timeout=5)
            self.assertIsInstance(item, Error)
            self.assertIn("rejected", item.message)
            await asyncio.sleep(0.05)  # let the loop settle back to idle
            self.assertTrue(eng.alive)          # engine survived the bad request
            self.assertIsNone(eng._fatal)
            self.assertEqual(eng._inflight, 0)  # inflight not leaked
        finally:
            eng.stop()


if __name__ == "__main__":
    unittest.main()
