"""Async serving bridge for the synchronous scheduler."""

from __future__ import annotations

import asyncio
import itertools
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import inferd.env  # noqa: F401  (CUDA preload before torch)

import torch  # noqa: E402

from core.scheduler import ContinuousBatchScheduler, RequestStatus  # noqa: E402

_RATE_WINDOW_S = 5.0


@dataclass
class TokenChunk:
    text: str


@dataclass
class Done:
    finish_reason: str
    generated_tokens: int


@dataclass
class Error:
    message: str


@dataclass
class _Submit:
    request_id: int
    prompt_ids: list[int]
    max_tokens: int
    channel: "StreamChannel"
    temperature: float | None = None
    top_p: float | None = None


@dataclass
class _Cancel:
    request_id: int


class StreamChannel:
    """Per-request queue shared by the engine thread and SSE handler."""

    def __init__(self, request_id: int, loop: asyncio.AbstractEventLoop) -> None:
        self.request_id = request_id
        self._loop = loop
        self.queue: asyncio.Queue = asyncio.Queue()
        self.t_submit = time.perf_counter()
        self.first_token_time: float | None = None
        self.seen_tokens = 0
        self.prev_text_len = 0

    def push(self, item) -> None:
        """Thread-safe hand-off into the request's asyncio.Queue."""
        self._loop.call_soon_threadsafe(self.queue.put_nowait, item)


class Engine:
    """Owns the scheduler on a background thread; bridges to async SSE clients."""

    def __init__(
        self,
        scheduler: ContinuousBatchScheduler,
        tokenizer,
        *,
        model_name: str,
        device: str,
        max_concurrent: int,
        max_queue_depth: int,
        idle_poll_s: float = 0.05,
    ) -> None:
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = device
        self.max_concurrent = max_concurrent
        self.max_queue_depth = max_queue_depth
        self._idle_poll_s = idle_poll_s

        self._inbox: queue.Queue = queue.Queue()
        self._channels: dict[int, StreamChannel] = {}  # engine-thread-owned
        self._id_counter = itertools.count(1)

        self._inflight = 0
        self._inflight_lock = threading.Lock()

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="inferd-engine", daemon=True)

        self._start_time = time.perf_counter()
        self._last_ttft_s: float | None = None
        self._rate_samples: deque[tuple[float, int]] = deque()
        self._fatal: str | None = None  # set if the engine thread dies on an error

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self, *, join_timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout=join_timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def encode(self, prompt: str) -> list[int]:
        """Tokenize a prompt to ids (CPU-cheap; safe to call on the event loop)."""
        return list(self.tokenizer(prompt, add_special_tokens=True).input_ids)

    # -- limits / admission ------------------------------------------------

    def limit_violation(self, prompt_len: int, max_tokens: int) -> str | None:
        """Return a reason string if the request can never be admitted (-> 400)."""
        cfg = self.scheduler.config
        total = prompt_len + max_tokens
        if total > cfg.max_model_len:
            return f"request length {total} exceeds max_model_len {cfg.max_model_len}"
        needed = -(-total // cfg.block_size)  # ceil
        if needed > cfg.max_blocks:
            return f"request needs {needed} blocks, budget is {cfg.max_blocks}"
        return None

    def submit(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> StreamChannel | None:
        """
        Register a request and return its StreamChannel, or None if the engine is
        saturated (caller should respond 429). Must be called from the event loop
        (it binds the channel to the running loop).
        """
        with self._inflight_lock:
            if not self.alive or self._fatal is not None:
                raise RuntimeError(self._fatal or "engine unavailable")
            if self._inflight >= self.max_concurrent + self.max_queue_depth:
                return None
            self._inflight += 1
        request_id = next(self._id_counter)
        channel = StreamChannel(request_id, asyncio.get_running_loop())
        self._inbox.put(
            _Submit(request_id, prompt_ids, max_tokens, channel, temperature=temperature, top_p=top_p)
        )
        self._wake.set()
        if not self.alive or self._fatal is not None:
            # Engine died between the alive check and here; ensure this client is
            # errored rather than left hanging on an inbox the loop won't drain.
            channel.push(Error(self._fatal or "engine unavailable"))
            self._dec_inflight()
        return channel

    def cancel(self, request_id: int) -> None:
        """Request cancellation (e.g. on client disconnect). Idempotent."""
        self._inbox.put(_Cancel(request_id))
        self._wake.set()

    def _dec_inflight(self) -> None:
        with self._inflight_lock:
            self._inflight = max(0, self._inflight - 1)

    # -- the engine thread -------------------------------------------------

    def _run_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._drain_inbox()
                snap = self.scheduler.metrics_snapshot()
                if snap.active_sequences == 0 and snap.waiting_sequences == 0:
                    self._record_rate(time.perf_counter())
                    # nothing to do: sleep until woken by a submit/cancel or timeout
                    self._wake.wait(timeout=self._idle_poll_s)
                    self._wake.clear()
                    continue
                self.scheduler.step()
                self._emit_updates()
        except BaseException as exc:  # noqa: BLE001 — a dead loop must not silently hang clients
            # An unhandled error (e.g. CUDA OOM in step()) would otherwise kill this
            # daemon thread and leave every connected client blocked forever on its
            # queue. Surface it instead: error the in-flight channels, record it for
            # /healthz, and let `alive` go False so the server reports degraded.
            self._fatal = f"{type(exc).__name__}: {exc}"
            for channel in list(self._channels.values()):
                channel.push(Error(f"engine crashed: {self._fatal}"))
            self._channels.clear()
            self._error_pending_submissions(f"engine crashed: {self._fatal}")
        finally:
            self._shutdown_channels()

    def _drain_inbox(self) -> None:
        while True:
            try:
                cmd = self._inbox.get_nowait()
            except queue.Empty:
                return
            if isinstance(cmd, _Submit):
                try:
                    self.scheduler.submit(
                        cmd.prompt_ids,
                        max_tokens=cmd.max_tokens,
                        request_id=cmd.request_id,
                        temperature=cmd.temperature,
                        top_p=cmd.top_p,
                    )
                except Exception as exc:  # a bad request must fail only itself, not kill the loop
                    cmd.channel.push(Error(f"rejected: {exc}"))
                    self._dec_inflight()
                else:
                    self._channels[cmd.request_id] = cmd.channel
            elif isinstance(cmd, _Cancel):
                channel = self._channels.pop(cmd.request_id, None)
                self.scheduler.cancel(cmd.request_id)
                if channel is not None:
                    req = self.scheduler.get(cmd.request_id)
                    tokens = len(req.generated_ids) if req is not None else channel.seen_tokens
                    channel.push(Done(RequestStatus.CANCELLED.value, tokens))
                    self._dec_inflight()

    def _error_pending_submissions(self, message: str) -> None:
        while True:
            try:
                cmd = self._inbox.get_nowait()
            except queue.Empty:
                return
            if isinstance(cmd, _Submit):
                cmd.channel.push(Error(message))
                self._dec_inflight()

    def _emit_updates(self) -> None:
        now = time.perf_counter()
        for request_id, channel in list(self._channels.items()):
            req = self.scheduler.get(request_id)
            if req is None:
                continue
            new_ids = req.generated_ids[channel.seen_tokens:]
            if new_ids:
                if channel.first_token_time is None:
                    channel.first_token_time = now
                    self._last_ttft_s = now - channel.t_submit
                channel.seen_tokens = len(req.generated_ids)
                # Incremental detok: cumulative decode + suffix diff is robust to
                # BPE merges / leading-space artefacts that per-token decode breaks.
                full_text = self.tokenizer.decode(req.generated_ids, skip_special_tokens=True)
                delta = full_text[channel.prev_text_len:]
                channel.prev_text_len = len(full_text)
                if delta:
                    channel.push(TokenChunk(delta))
            if req.status in (RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.CANCELLED):
                if req.status is RequestStatus.FAILED:
                    channel.push(Error(req.error or "generation failed"))
                else:
                    channel.push(Done(req.status.value, len(req.generated_ids)))
                self._channels.pop(request_id, None)
                self._dec_inflight()
        self._record_rate(now)

    def _shutdown_channels(self) -> None:
        for channel in list(self._channels.values()):
            channel.push(Error("server shutting down"))
        self._channels.clear()

    # -- metrics -----------------------------------------------------------

    def _record_rate(self, now: float) -> None:
        total = self.scheduler.metrics_snapshot().total_generated_tokens
        self._rate_samples.append((now, total))
        cutoff = now - _RATE_WINDOW_S
        while len(self._rate_samples) > 1 and self._rate_samples[0][0] < cutoff:
            self._rate_samples.popleft()

    def _tokens_per_second(self) -> float:
        if len(self._rate_samples) < 2:
            return 0.0
        (t0, n0), (t1, n1) = self._rate_samples[0], self._rate_samples[-1]
        dt = t1 - t0
        return (n1 - n0) / dt if dt > 0 else 0.0

    def metrics(self) -> dict:
        """Snapshot for /metrics. Safe to call from the event loop (reads are
        GIL-atomic int/len reads; eventual consistency is fine for metrics)."""
        snap = self.scheduler.metrics_snapshot().as_dict()
        peak_vram_mb = (
            torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        )
        snap.update(
            tokens_per_second=round(self._tokens_per_second(), 2),
            last_ttft_s=self._last_ttft_s,
            peak_vram_mb=round(peak_vram_mb, 1),
            uptime_s=round(time.perf_counter() - self._start_time, 1),
            model=self.model_name,
        )
        return snap
