"""
Iteration-level continuous batching: FCFS admission, batched decode, block budgeting.

Block budget is enforced here; the default backend delegates KV to ModelRunner's HF cache.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from math import ceil
from typing import Protocol

import inferd.env  # noqa: F401  (CUDA preload before torch)

import torch  # noqa: E402

from core.batched_cache import split_caches, stack_caches  # noqa: E402
from core.spec_decode import nucleus_probs, sample_from  # noqa: E402


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for FCFS continuous batching."""

    max_blocks: int
    block_size: int = 16
    max_concurrent_sequences: int = 32
    max_model_len: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    continuous: bool = True

    def __post_init__(self) -> None:
        if self.max_blocks <= 0:
            raise ValueError("max_blocks must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.max_concurrent_sequences <= 0:
            raise ValueError("max_concurrent_sequences must be positive")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")


@dataclass
class GenerationRequest:
    """One submitted generation request."""

    request_id: int
    prompt_ids: list[int]
    max_tokens: int
    prompt_text: str = ""
    status: RequestStatus = RequestStatus.WAITING
    generated_ids: list[int] = field(default_factory=list)
    error: str | None = None
    reserved_blocks: int = 0
    kv: object | None = None
    last_logits: torch.Tensor | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def generated_len(self) -> int:
        return len(self.generated_ids)

    @property
    def total_reserved_tokens(self) -> int:
        return self.prompt_len + self.max_tokens


@dataclass(frozen=True)
class SchedulerMetrics:
    """Live metrics consumed by benchmarks now and serving/dashboard later."""

    waiting_sequences: int
    active_sequences: int
    completed_sequences: int
    failed_sequences: int
    admitted_sequences: int
    evicted_sequences: int
    iterations: int
    total_generated_tokens: int
    used_blocks: int
    free_blocks: int
    max_blocks_used: int

    def as_dict(self) -> dict:
        return asdict(self)


class SchedulerBackend(Protocol):
    """Minimal model interface used by the scheduler."""

    eos_token_id: int | None
    device: str

    def prefill(self, prompt_ids: list[int]) -> tuple[torch.Tensor, object]:
        """Return next-token logits [1, vocab] and an opaque kv handle."""

    def decode_batch(
        self, token_ids: list[int], kvs: list[object]
    ) -> tuple[list[torch.Tensor], list[object]]:
        """Decode one token for each running sequence in a SINGLE forward.

        Returns per-row next-token logits ([1, vocab] each) and updated kv
        handles, aligned with the input order.
        """


class ModelRunnerBackend:
    """Scheduler backend for core.model_runner.ModelRunner."""

    def __init__(self, runner) -> None:
        self.runner = runner
        self.device = runner.device
        self.eos_token_id = runner.tokenizer.eos_token_id

    def prefill(self, prompt_ids: list[int]) -> tuple[torch.Tensor, object]:
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        logits, kv = self.runner.forward(tokens, None)
        return logits[:, -1, :], kv

    def decode(self, token_id: int, kv: object) -> tuple[torch.Tensor, object]:
        token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        logits, kv = self.runner.forward(token, kv)
        return logits[:, -1, :], kv

    def decode_batch(
        self, token_ids: list[int], kvs: list[object]
    ) -> tuple[list[torch.Tensor], list[object]]:
        """One batched decode: stack caches, forward, split back per sequence."""
        batched, lengths = stack_caches(kvs)
        batch = len(token_ids)
        max_len = max(lengths)
        input_ids = torch.tensor([[t] for t in token_ids], dtype=torch.long, device=self.device)
        attn = torch.zeros((batch, max_len + 1), dtype=torch.bool, device=self.device)
        for i, length in enumerate(lengths):
            attn[i, max_len - length:] = True
        position_ids = torch.tensor([[length] for length in lengths], dtype=torch.long, device=self.device)
        logits, batched = self.runner.forward(
            input_ids, batched, attention_mask=attn, position_ids=position_ids
        )
        caches = split_caches(batched, [length + 1 for length in lengths])
        last = logits[:, -1, :]
        return [last[i:i + 1] for i in range(batch)], caches


class ContinuousBatchScheduler:
    """FCFS scheduler; admission reserves prompt+max_tokens blocks up front."""

    def __init__(self, backend: SchedulerBackend, config: SchedulerConfig) -> None:
        self.backend = backend
        self.config = config
        self._waiting: deque[GenerationRequest] = deque()
        self._running: OrderedDict[int, GenerationRequest] = OrderedDict()
        self._completed: OrderedDict[int, GenerationRequest] = OrderedDict()
        self._failed: OrderedDict[int, GenerationRequest] = OrderedDict()
        self._next_id = 1
        self._used_blocks = 0
        self._max_blocks_used = 0
        self._admitted = 0
        self._evicted = 0
        self._iterations = 0
        self._total_generated = 0
        torch.manual_seed(config.seed)

    def submit(
        self,
        prompt_ids: list[int] | torch.Tensor,
        *,
        max_tokens: int,
        prompt_text: str = "",
        request_id: int | None = None,
    ) -> int:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        ids = _to_id_list(prompt_ids)
        rid = self._allocate_id(request_id)
        req = GenerationRequest(
            request_id=rid,
            prompt_ids=ids,
            max_tokens=max_tokens,
            prompt_text=prompt_text,
        )
        self._waiting.append(req)
        return rid

    def step(self) -> SchedulerMetrics:
        """Admit, sample, batched decode, evict finished, backfill."""
        self._maybe_admit()
        if not self._running:
            return self.metrics_snapshot()

        self._iterations += 1
        finished: list[int] = []
        active: list[GenerationRequest] = []
        for req in list(self._running.values()):
            assert req.last_logits is not None
            next_id = self._sample_next(req.last_logits)
            req.generated_ids.append(next_id)
            self._total_generated += 1
            if next_id == self.backend.eos_token_id or req.generated_len >= req.max_tokens:
                req.status = RequestStatus.COMPLETED
                finished.append(req.request_id)
            else:
                active.append(req)

        if active:
            token_ids = [req.generated_ids[-1] for req in active]
            kvs = [req.kv for req in active]
            logits_list, kv_list = self.backend.decode_batch(token_ids, kvs)
            for req, logits, kv in zip(active, logits_list, kv_list):
                req.last_logits, req.kv = logits, kv

        for rid in finished:
            self._evict_completed(rid)
        self._maybe_admit()
        return self.metrics_snapshot()

    def _maybe_admit(self) -> None:
        if self.config.continuous or not self._running:
            self._admit_waiting()

    def run_until_complete(self, *, max_iterations: int | None = None) -> list[GenerationRequest]:
        while self._waiting or self._running:
            if max_iterations is not None and self._iterations >= max_iterations:
                raise RuntimeError("scheduler exceeded max_iterations")
            self.step()
        return [*self._completed.values(), *self._failed.values()]

    def metrics_snapshot(self) -> SchedulerMetrics:
        return SchedulerMetrics(
            waiting_sequences=len(self._waiting),
            active_sequences=len(self._running),
            completed_sequences=len(self._completed),
            failed_sequences=len(self._failed),
            admitted_sequences=self._admitted,
            evicted_sequences=self._evicted,
            iterations=self._iterations,
            total_generated_tokens=self._total_generated,
            used_blocks=self._used_blocks,
            free_blocks=self.config.max_blocks - self._used_blocks,
            max_blocks_used=self._max_blocks_used,
        )

    def get(self, request_id: int) -> GenerationRequest | None:
        if request_id in self._running:
            return self._running[request_id]
        if request_id in self._completed:
            return self._completed[request_id]
        if request_id in self._failed:
            return self._failed[request_id]
        for req in self._waiting:
            if req.request_id == request_id:
                return req
        return None

    def _admit_waiting(self) -> None:
        while self._waiting and len(self._running) < self.config.max_concurrent_sequences:
            req = self._waiting[0]
            needed = self._reserved_blocks(req)
            if req.total_reserved_tokens > self.config.max_model_len:
                self._waiting.popleft()
                self._fail(req, f"request length {req.total_reserved_tokens} exceeds max_model_len")
                continue
            if needed > self.config.max_blocks:
                self._waiting.popleft()
                self._fail(req, f"request needs {needed} blocks, budget is {self.config.max_blocks}")
                continue
            if self._used_blocks + needed > self.config.max_blocks:
                break

            self._waiting.popleft()
            req.reserved_blocks = needed
            req.status = RequestStatus.RUNNING
            req.last_logits, req.kv = self.backend.prefill(req.prompt_ids)
            self._running[req.request_id] = req
            self._used_blocks += needed
            self._max_blocks_used = max(self._max_blocks_used, self._used_blocks)
            self._admitted += 1

    def _evict_completed(self, request_id: int) -> None:
        req = self._running.pop(request_id)
        self._used_blocks -= req.reserved_blocks
        req.kv = None
        req.last_logits = None
        self._completed[request_id] = req
        self._evicted += 1

    def _fail(self, req: GenerationRequest, error: str) -> None:
        req.status = RequestStatus.FAILED
        req.error = error
        self._failed[req.request_id] = req

    def _sample_next(self, logits: torch.Tensor) -> int:
        probs = nucleus_probs(logits, self.config.temperature, self.config.top_p)
        return int(sample_from(probs).item())

    def _reserved_blocks(self, req: GenerationRequest) -> int:
        return ceil(req.total_reserved_tokens / self.config.block_size)

    def _allocate_id(self, requested: int | None) -> int:
        if requested is None:
            rid = self._next_id
            self._next_id += 1
            return rid
        if self.get(requested) is not None or any(r.request_id == requested for r in self._waiting):
            raise ValueError(f"duplicate request_id: {requested}")
        self._next_id = max(self._next_id, requested + 1)
        return requested


def _to_id_list(prompt_ids: list[int] | torch.Tensor) -> list[int]:
    if isinstance(prompt_ids, torch.Tensor):
        flat = prompt_ids.detach().cpu().reshape(-1).tolist()
        return [int(x) for x in flat]
    return [int(x) for x in prompt_ids]
