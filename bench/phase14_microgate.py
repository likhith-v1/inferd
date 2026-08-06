"""GPU cache-reconciliation equivalence gates for both Phase 14 pairs."""

from __future__ import annotations

import argparse

import torch

from bench.pair_configs import PAIR_NAMES, get_pair, validate_local_revisions
from core.model_runner import ModelRunner
from core.speculation import CROP_NO_REPLAY

BF16_LOGIT_ATOL = 32 * torch.finfo(torch.bfloat16).eps


@torch.no_grad()
def run_pair(name: str, gamma: int = 4) -> None:
    pair = get_pair(name)
    validate_local_revisions(pair)
    target = ModelRunner.load_target(pair.target)
    draft = ModelRunner.load_draft(pair.draft)
    target.validate_speculation_pair(
        draft,
        expected_tokenizer_sha256=pair.tokenizer_sha256,
        expected_vocab_size=pair.vocab_size,
    )
    prompt = target.tokenizer("Explain cache rollback briefly.", return_tensors="pt").input_ids.to(
        target.device
    )

    for accepted in (0, gamma // 2, gamma):
        _, cache = target.forward(prompt, None)
        checkpoint = target.checkpoint_speculation(cache)
        proposals = []
        logits, _ = target.forward(prompt, None)
        for _ in range(gamma + 1):
            token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            proposals.append(int(token.item()))
            logits, _ = target.forward(
                torch.tensor([proposals], device=target.device),
                None,
            )
        verify = torch.tensor([proposals[:gamma]], device=target.device)
        verified, cache = target.forward(verify, cache)
        direct_verify, _ = target.forward(
            torch.cat((prompt, verify), dim=1), None
        )
        verify_max_abs = float(
            (verified - direct_verify[:, -gamma:, :]).abs().max()
        )
        assert verify_max_abs <= BF16_LOGIT_ATOL, (
            name, "verify", verify_max_abs, BF16_LOGIT_ATOL
        )
        emitted = proposals[:accepted] + [proposals[accepted]]
        widths = []
        original_forward = target.forward

        def observed_forward(tokens, kv=None, **kwargs):
            widths.append(tokens.shape[1])
            return original_forward(tokens, kv, **kwargs)

        target.forward = observed_forward
        reconciled, _ = target.reconcile_speculation(cache, checkpoint, accepted, emitted)
        target.forward = original_forward
        direct_ids = torch.cat((prompt, torch.tensor([emitted], device=target.device)), dim=1)
        direct, _ = target.forward(direct_ids, None)
        max_abs = float((reconciled[:, -1, :] - direct[:, -1, :]).abs().max())
        assert max_abs <= BF16_LOGIT_ATOL, (name, accepted, max_abs, BF16_LOGIT_ATOL)
        expected_width = 1 if pair.reconciliation == CROP_NO_REPLAY else accepted + 1
        assert widths == [expected_width], (name, accepted, widths)
        print(
            f"[phase14_microgate] {name} accepted={accepted} "
            f"verify_max_abs={verify_max_abs:.6f} reconcile_max_abs={max_abs:.6f}"
        )
    print(f"[phase14_microgate] {name} PASS ({pair.reconciliation})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=(*PAIR_NAMES, "both"),
                        default="both")
    args = parser.parse_args()
    names = PAIR_NAMES if args.pair == "both" else (args.pair,)
    for name in names:
        run_pair(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
