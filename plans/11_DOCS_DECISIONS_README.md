Preferred model: Claude Sonnet 4.6 | Effort: high

# 11 — Docs, Decisions & README

> Close the project: a `DECISIONS.md` that records every load-bearing call, a `README.md` where the three resume bullets, plots, and a one-command reproduce all line up, and a short demo video under load.

## Constraints (this phase)
local-first · every claim traces to a real number in `bench/results/` · one-command reproduction · honest framing (vLLM = ceiling; spec×batch don't multiply) · no `AGENTS.md`/`claude.md` here (created manually elsewhere).

## Scope
**In:** `DECISIONS.md` (seed from plan §9 + every decision logged en route, with absolute dates); `README.md` (the bullets, the plots, how to reproduce from one command); a short demo video under load (live α + throughput curves as the payoff).
**Out:** code; new benchmarks (consume 09's results).
**Standalone value:** "a portfolio-ready repo: bullets backed by reproducible numbers, decisions documented, demo recorded."

## Subagent breakdown
- **decisions-log author** — `DECISIONS.md`: every architecture/tooling/model decision with rationale + date; resolve all parameterized choices (corpus id, trainer stack, draft model, FP8 recipe).
- **readme / repro author** — `README.md`: the three bullets with real numbers, embedded plots, and the exact one-command reproduce path.
- **demo-script author** — record the dashboard under load (the α + throughput shot); capture script/checklist.

## Git worktree workflow
- Branch `phase-11-docs`, worktree `../inferd-wt/11-docs`. Last to merge; reads everything, writes docs only.

## Owned / Avoided files
- **Owns:** `DECISIONS.md`, `README.md`, `docs/demo.md` (capture checklist), demo asset manifest.
- **Avoids:** all code dirs (reads `bench/results/`, `core/`, etc.; edits none).

## Commands, tests, validation
```bash
# the README's promise, executed verbatim to prove it:
uv sync && uv run python bench/run_all.py --rungs hf,ours,vllm --plots
# confirm every bullet's number appears in bench/results/ and every plot renders in the README
```
- **Validation:** the one-command reproduce in the README actually runs; each of the three bullets cites a number present in `bench/results/`; every embedded plot exists; `DECISIONS.md` has no unresolved placeholders (corpus, stack, draft, FP8 all named); a final `code-review`/`security-review` pass on the repo.

## Risks / Rollback / Exit / Handoff
- **Risks:** README numbers drifting from `bench/results/`; the "one command" not actually working on a clean clone; lingering placeholders.
- **Mitigation/Rollback:** generate the bullets *from* `bench/results/` rather than hand-typing; dry-run the reproduce path on a fresh `uv sync`; grep the docs for `<...>` placeholders before merge.
- **Exit:** every resume-bullet number is real and reproducible from one command; decisions documented; demo recorded.
- **Handoff:** project complete; the README + demo are the portfolio artifact.

## Model Selection (confirm or override)
- **Claude Sonnet 4.6 | high** *(recommended)* — synthesis/writing + clean docs, cost-efficient.
- **GPT-5.4 | medium** — co-equal for the writeup.
- **Claude Opus 4.8 | high** — only if the writeup needs heavy reasoning to frame results honestly (the spec×batch nuance).
> Recommendation: Sonnet 4.6 high. **Your call.**

## Skills & MCPs to decide at implementation
- **Recommend:** `code-review` + `security-review` as the final repo gates.
- **Candidates:** `run`/`verify` to capture the demo; firecrawl only if citing external references.
- **Question:** run a final `security-review` before publishing the repo? Capture the demo with `run`, or record manually?

## Execution questions for this phase
1. Where do the final bullet numbers come from — lock them with 09 before writing.
2. Demo video: hosted where (local file, repo asset, external)? Length/target audience?
3. README audience: recruiters (lead with bullets) vs engineers (lead with architecture)?
4. License + which weights/adapters (if any) are publishable vs local-only?
