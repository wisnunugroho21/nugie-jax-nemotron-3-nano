"""
pretrain_lc_phase.py — Pre-Training LC-Phase: Long-Context Extension

Paper §2.5: after the main pre-training run, the model is further trained on
a mixture of very long sequences (up to 512 K tokens) and normal-length sequences
(4 K tokens) to extend its effective context window without degrading short-context
benchmarks.

Run standalone:
    python pretrain_lc_phase.py

Resumes from the latest pretrain checkpoint in PRETRAIN_CKPT_DIR if available.
"""

import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer

from training_shared import (
    LC_SEQ_LEN,
    LC_PHASE_STEPS,
    LC_PHASE_LR,
    PRETRAIN_WD,
    PRETRAIN_B1,
    PRETRAIN_B2,
    PRETRAIN_BATCH,
    PRETRAIN_CKPT_DIR,
    build_model,
    collect_moe_layers,
    make_constant_lr_optimizer,
    load_pretrain_data,
    make_batches,
    pretrain_step,
    update_moe_biases,
    make_checkpoint_manager,
    save_checkpoint,
    try_load_from_dir,
    NemotronNanoBlock,
)


def run_lc_phase(model: NemotronNanoBlock, tokenizer) -> None:
    """Continuous pre-training to extend the model's context window.

    Paper §2.5: uses a constant LR of 1e-5 on a mixture of 512 K and 4 K
    sequences.  Using only very long sequences degraded short-context benchmarks;
    mixing in 4 K sequences preserves them.
    """
    print("\n=== Pre-Training LC-Phase: Long-Context Extension ===")

    lc_chunks = load_pretrain_data(
        split="train", max_samples=100, seq_len=LC_SEQ_LEN,
        tokenizer=tokenizer, skip=700,
    )

    tx         = make_constant_lr_optimizer(LC_PHASE_LR, PRETRAIN_WD, PRETRAIN_B1, PRETRAIN_B2)
    optimizer  = nnx.Optimizer(model, tx, wrt=nnx.Param)
    moe_layers = collect_moe_layers(model)
    ckpt_mgr   = make_checkpoint_manager(PRETRAIN_CKPT_DIR)

    step = 0
    for _ in range(LC_PHASE_STEPS):
        for batch_np in make_batches(lc_chunks, PRETRAIN_BATCH):
            if step >= LC_PHASE_STEPS:
                break
            batch = jnp.array(batch_np)
            loss  = pretrain_step(model, optimizer, batch)
            update_moe_biases(moe_layers)
            step += 1

            if step % 50 == 0:
                print(f"  LC-Phase step {step:3d} | loss={float(loss):.4f}")

    # Offset the step key to avoid colliding with Phase 1/2 checkpoint keys.
    save_checkpoint(ckpt_mgr, model, step + 10_000)
    print("LC-Phase complete.\n")


if __name__ == "__main__":
    print("Loading Nemotron tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building model …")
    model = build_model(seed=0)
    try_load_from_dir(PRETRAIN_CKPT_DIR, model, model.config)

    run_lc_phase(model, tokenizer)
