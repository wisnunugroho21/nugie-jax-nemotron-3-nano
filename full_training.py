"""
full_training.py — Complete Training Pipeline for Nemotron 3 Nano

This file implements the full training recipe described in:
  "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid
   Mamba-Transformer Model for Agentic Reasoning"
  https://arxiv.org/abs/2512.20848

Pipeline Overview
-----------------
The training is divided into two major phases:

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                          PRE-TRAINING                                   │
  │                                                                         │
  │  Phase 1 — Diverse data   : 23.5 T tokens  (94% of total pre-training) │
  │  Phase 2 — High-quality   :  1.5 T tokens  (final 6%)                  │
  │  LC-Phase — Long context  :  121 B tokens  (continuous fine-tuning)     │
  │                                                                         │
  │  Optimizer  : AdamW  β₁=0.9, β₂=0.95, weight_decay=0.1                │
  │  LR schedule: Warmup → Stable → Cosine-Decay  (WSD)                    │
  │  MoE        : aux-loss-free load balancing + standard balance loss      │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                         POST-TRAINING                                   │
  │                                                                         │
  │  Step 1 — SFT   : Supervised Fine-Tuning on diverse chat/agentic data  │
  │  Step 2 — RLVR  : Multi-environment RL from Verifiable Rewards (GRPO)  │
  │  Step 3 — RLHF  : RL from Human Feedback via a GenRM judge             │
  │  Step 4 — RLVR  : Second RLVR pass after RLHF for further refinement   │
  └─────────────────────────────────────────────────────────────────────────┘

Stage files
-----------
Each training stage lives in its own module and can be run independently:

  pretrain_phase1.py   — Pre-Training Phase 1: Diverse Data
  pretrain_phase2.py   — Pre-Training Phase 2: High-Quality Data
  pretrain_lc_phase.py — Pre-Training LC-Phase: Long-Context Extension
  sft.py               — Post-Training Step 1: Supervised Fine-Tuning
  rlvr.py              — Post-Training Steps 2 & 4: RLVR with GRPO
  rlhf.py              — Post-Training Step 3: RLHF with GenRM

Shared utilities (constants, helpers, loss functions, etc.) live in:

  training_shared.py

How to run
----------
  python full_training.py          # run the full pipeline
  python pretrain_phase1.py        # run only Phase 1
  python sft.py                    # run only SFT (loads pretrain checkpoint)

  You can comment out any phase(s) in main() to run a subset.

Paper reference
---------------
  NVIDIA (2025). "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts
  Hybrid Mamba-Transformer Model for Agentic Reasoning."
  arXiv:2512.20848.
"""

from transformers import AutoTokenizer

from training_shared import (
    PRETRAIN_CKPT_DIR,
    RLVR_CKPT_DIR,
    build_model,
    try_load_from_dir,
)
from pretrain_phase1   import run_pretrain_phase1
from pretrain_phase2   import run_pretrain_phase2
from pretrain_lc_phase import run_lc_phase
from sft               import run_sft
from rlvr              import run_rlvr
from rlhf              import run_rlhf


# =============================================================================
# Sanity checks
# =============================================================================

# All hyperparameter assertions are enforced on import of training_shared.


def main() -> None:
    """Run the complete Nemotron 3 Nano training pipeline.

    Phase order (§1 of the paper):
      Pre-Train Phase 1  →  Pre-Train Phase 2  →  LC-Phase
      →  SFT  →  RLVR  →  RLHF  →  RLVR (second pass)

    Each phase saves its checkpoint so the pipeline can be resumed after any
    stage by commenting out earlier phases in this function.
    """

    # ── Tokenizer ──────────────────────────────────────────────────────────
    print("Loading Nemotron tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ──────────────────────────────────────────────────────────────
    print("Building model …")
    model  = build_model(seed=0)
    config = model.config   # keep a reference for checkpoint restoration

    # ── Pre-Training ───────────────────────────────────────────────────────
    # Paper §2: 25 T token pretraining in two phases + long-context extension.
    run_pretrain_phase1(model, tokenizer)
    run_pretrain_phase2(model, tokenizer)
    run_lc_phase(model, tokenizer)

    # ── Post-Training ──────────────────────────────────────────────────────
    # Paper §3: SFT → RLVR → RLHF → RLVR (second pass).
    run_sft(model, tokenizer)
    run_rlvr(model, tokenizer)       # First RLVR pass (immediately after SFT)
    run_rlhf(model, tokenizer)
    run_rlvr(model, tokenizer)       # Second RLVR pass (after RLHF, §3.2)

    print("\n✓ Full training pipeline complete.")
    print("  Final checkpoint is in:", RLVR_CKPT_DIR)


if __name__ == "__main__":
    main()
