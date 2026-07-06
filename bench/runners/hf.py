"""
bench.runners.hf — naive HF generate() floor runner.

Measures:
  - Single-stream: TTFT + ITL via TextIteratorStreamer (concurrency=1).
  - Concurrency sweep: batched generate() with left-padding; reports throughput.

Results are written to bench/results/<timestamp>_hf_<model>/result.json.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Thread

import torch
from transformers import TextIteratorStreamer

from bench.metrics import (
    BenchResult,
    ConcurrencySweepPoint,
    SingleStreamResult,
    VramSampler,
    env_stamp,
    itl,
    throughput,
    ttft,
    write_result_json,
)
from bench.model_loader import load
from bench.workload import (
    CANONICAL,
    GREEDY,
    MAX_TOKENS,
    PROMPTS,
    SamplingProfile,
    workload_hash,
)
from core.spec_decode import nucleus_probs

WEIGHTS_ROOT = Path(__file__).parent.parent.parent / "weights"


def _single_stream_run(
    lm,
    lm_head,
    tokenizer,
    prompt: str,
    profile: SamplingProfile,
    max_tokens: int,
    warmup_runs: int,
    seed: int,
    device: str = "cuda:0",
) -> SingleStreamResult:
    torch.manual_seed(seed)

    def _encode(text: str):
        return tokenizer(text, return_tensors="pt").input_ids.to(device)

    # Warmup: discard timing, but populate CUDA kernels & KV state.
    for _ in range(warmup_runs):
        input_ids = _encode(prompt)
        with torch.no_grad():
            hidden = lm(input_ids=input_ids).last_hidden_state
            if lm_head is not None:
                lm_head(hidden)
        torch.cuda.synchronize()

    # Timed run via streamer.
    input_ids = _encode(prompt)
    streamer = TextIteratorStreamer(
        tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    torch.cuda.reset_peak_memory_stats()
    first_token_time: list[float] = []
    generated_tokens: list[int] = [0]
    start_time = time.perf_counter()

    def _generate():
        nonlocal first_token_time, generated_tokens

        torch.manual_seed(seed)
        ids = input_ids.clone()
        tokens_so_far = 0

        for step in range(max_tokens):
            with torch.no_grad():
                out = lm(input_ids=ids)
                hidden = out.last_hidden_state
                logits = lm_head(hidden) if lm_head is not None else hidden
            next_logits = logits[:, -1, :]

            if profile.temperature > 0:
                probs = nucleus_probs(next_logits, profile.temperature, profile.top_p)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = next_logits.argmax(dim=-1, keepdim=True)

            now = time.perf_counter()
            if step == 0:
                first_token_time.append(now)

            token_id = next_id[0, 0].item()
            streamer.put(torch.tensor([[token_id]], device="cpu"))
            tokens_so_far += 1

            if token_id == tokenizer.eos_token_id:
                break

            ids = torch.cat([ids, next_id], dim=-1)

        generated_tokens[0] = tokens_so_far
        streamer.end()

    # Run generation in a background thread so we can consume the streamer.
    with VramSampler() as vs:
        gen_thread = Thread(target=_generate)
        gen_thread.start()
        for _ in streamer:
            pass
        gen_thread.join()

    end_time = time.perf_counter()
    total = end_time - start_time
    first_t = first_token_time[0] if first_token_time else end_time
    ttft_s = ttft(start_time, first_t)
    n_tokens = generated_tokens[0]
    itl_s = itl(total, ttft_s, n_tokens)
    tps = throughput(n_tokens, total)
    torch_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    return SingleStreamResult(
        ttft_s=ttft_s,
        itl_s=itl_s,
        total_time_s=total,
        tokens_generated=n_tokens,
        throughput_tok_s=tps,
        peak_vram_mb=vs.peak_mb,
        peak_vram_torch_mb=torch_peak_mb,
        prompt=prompt,
        warmup_runs=warmup_runs,
    )


def _concurrency_run(
    lm,
    lm_head,
    tokenizer,
    prompts: list[str],
    concurrency: int,
    profile: SamplingProfile,
    max_tokens: int,
    warmup_runs: int,
    seed: int,
    device: str = "cuda:0",
) -> ConcurrencySweepPoint:
    """
    Batched generate at a given concurrency level.

    Left-pads all prompts to the same length.
    Throughput = total new tokens / batch wall time.
    """
    torch.manual_seed(seed)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # Pick `concurrency` prompts (cycle if needed).
    batch_prompts = [prompts[i % len(prompts)] for i in range(concurrency)]

    # Tokenize with left-padding.
    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    enc = tokenizer(batch_prompts, return_tensors="pt", padding=True)
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    tokenizer.padding_side = orig_side

    # Warmup.
    for _ in range(warmup_runs):
        with torch.no_grad():
            out = lm(input_ids=input_ids, attention_mask=attention_mask)
            if lm_head is not None:
                lm_head(out.last_hidden_state)
        torch.cuda.synchronize()

    # Timed batched decode loop (greedy only for simplicity at batch level).
    torch.cuda.reset_peak_memory_stats()
    ids = input_ids.clone()
    attn = attention_mask.clone()
    total_new = 0
    done = torch.zeros(concurrency, dtype=torch.bool, device=device)

    with VramSampler() as vs:
        t0 = time.perf_counter()
        for step in range(max_tokens):
            with torch.no_grad():
                out = lm(input_ids=ids, attention_mask=attn)
                hidden = out.last_hidden_state
                logits = lm_head(hidden) if lm_head is not None else hidden

            next_logits = logits[:, -1, :]
            if profile.temperature > 0:
                probs = nucleus_probs(next_logits, profile.temperature, profile.top_p)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_ids = next_logits.argmax(dim=-1)

            active = ~done
            next_ids = torch.where(
                active,
                next_ids,
                torch.full_like(next_ids, pad_id),
            )
            total_new += active.sum().item()
            done = done | (active & (next_ids == tokenizer.eos_token_id))
            if done.all():
                break

            ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
            attn = torch.cat([attn, active.unsqueeze(1).to(attn.dtype)], dim=1)

        t1 = time.perf_counter()

    wall = t1 - t0
    tps = throughput(total_new, wall)
    torch_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    return ConcurrencySweepPoint(
        concurrency=concurrency,
        total_time_s=wall,
        total_tokens=total_new,
        throughput_tok_s=tps,
        peak_vram_mb=vs.peak_mb,
        peak_vram_torch_mb=torch_peak_mb,
        warmup_runs=warmup_runs,
    )


def run(
    model_name: str = "Qwen3.5-9B",
    seed: int = 0,
    max_tokens: int = MAX_TOKENS,
    concurrency_grid: list[int] | None = None,
    warmup_runs: int = 3,
    profile_name: str = "canonical",
    results_dir: Path | None = None,
    device: str = "cuda:0",
) -> BenchResult:
    """
    Run the full HF floor benchmark and return a BenchResult.

    Steps:
      1. Load the text backbone.
      2. Single-stream run on all PROMPTS (concurrency=1) for TTFT/ITL.
      3. Concurrency sweep for throughput.
      4. Write JSON result to results_dir.
    """
    if concurrency_grid is None:
        concurrency_grid = [1, 2, 4, 8, 16]
    profile = CANONICAL if profile_name == "canonical" else GREEDY

    weights_dir = WEIGHTS_ROOT / model_name
    lm, lm_head, tokenizer = load(weights_dir, device=device)

    wh = workload_hash(profile, max_tokens)
    stamp = env_stamp(seed, wh)
    torch.manual_seed(seed)

    result = BenchResult(
        engine="hf",
        role="floor",
        model=model_name,
        profile=profile_name,
        max_tokens=max_tokens,
        env=stamp,
    )

    # 1. Single-stream on all prompts.
    print(f"\n[hf] Single-stream runs ({len(PROMPTS)} prompts, warmup={warmup_runs}) ...")
    for i, prompt in enumerate(PROMPTS):
        print(f"  [{i+1}/{len(PROMPTS)}] {prompt[:60]!r}...")
        sr = _single_stream_run(
            lm, lm_head, tokenizer, prompt, profile, max_tokens, warmup_runs,
            seed + i, device,
        )
        result.single_stream.append(sr)
        print(
            f"    TTFT={sr.ttft_s*1000:.1f}ms  ITL={sr.itl_s*1000:.1f}ms/tok"
            f"  toks/s={sr.throughput_tok_s:.1f}  peak_vram={sr.peak_vram_mb:.0f}MiB"
        )

    # 2. Concurrency sweep.
    print(f"\n[hf] Concurrency sweep {concurrency_grid} ...")
    for c in concurrency_grid:
        print(f"  concurrency={c} ...", end="", flush=True)
        sp = _concurrency_run(
            lm, lm_head, tokenizer, PROMPTS, c, profile, max_tokens, warmup_runs,
            seed, device,
        )
        result.concurrency_sweep.append(sp)
        print(
            f"  toks/s={sp.throughput_tok_s:.1f}  "
            f"peak_vram={sp.peak_vram_mb:.0f}MiB"
        )

    out_path = write_result_json(
        result, f"{result.engine}_{result.model.replace('/', '_')}", results_dir
    )
    print(f"\n[hf] Result written to {out_path}")
    return result
