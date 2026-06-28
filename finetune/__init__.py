"""inferd fine-tuning pipeline (Phase 03).

Scripts:
  train_qlora.py      — Unsloth-first QLoRA SFT
  prepare_dataset.py  — deterministic train/val splits
  eval_golden.py      — golden-set regression checks
  distill_draft.py    — sequence-level KD for draft α-lift
  export.py           — adapter export and merge-for-serving

Configs live in ``finetune/configs/``. Call ``inferd.env.bootstrap_finetune()``
before importing transformers in training entrypoints.
"""
