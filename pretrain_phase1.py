"""
pretrain_phase1.py — Pre-Training Phase 1: Diverse Data

Paper §2.3: trains on 23.5 T tokens from a broad, diverse data mixture including
web crawl, code, math, Wikipedia, academic text, multilingual text, and SFT-style
synthetic data.  The full Warmup-Stable-Decay (WSD) schedule is used.

Run standalone:
    python pretrain_phase1.py
"""

from flax import nnx
from transformers import AutoTokenizer

from training_shared import (
    PRETRAIN_SEQ_LEN,
    PRETRAIN_BATCH,
    PHASE1_STEPS,
    PRETRAIN_PEAK_LR,
    PRETRAIN_MIN_LR,
    PRETRAIN_WARMUP_STEPS,
    PRETRAIN_STABLE_STEPS,
    PRETRAIN_DECAY_STEPS,
    PRETRAIN_WD,
    PRETRAIN_B1,
    PRETRAIN_B2,
    PRETRAIN_CKPT_DIR,
    PRETRAIN_CKPT_EVERY,
    PRETRAIN_VAL_STEPS,
    build_model,
    collect_moe_layers,
    make_decayed_lr_optimizer,
    load_pretrain_data,
    make_batches,
    pretrain_step,
    update_moe_biases,
    evaluate_pretrain,
    make_checkpoint_manager,
    save_checkpoint,
    NemotronNanoBlock,
)


def run_pretrain_phase1(model: NemotronNanoBlock, tokenizer) -> None:
    """Pre-train on a broad, diverse data mixture.

    Paper §2.3: Phase 1 uses 23.5 T tokens from 15 data categories including
    web crawl (medium → high quality), code, math, Wikipedia, academic text,
    multilingual text, and SFT-style synthetic data.

    Training uses the full Warmup-Stable-Decay schedule (§2.4).
    MoE load balancing is applied at every step (both bias update and aux loss).
    """
    print("\n=== Pre-Training Phase 1: Diverse Data ===")

    train_chunks = load_pretrain_data(
        split="train", max_samples=200, seq_len=PRETRAIN_SEQ_LEN,
        tokenizer=tokenizer, skip=0,
    )
    val_chunks = load_pretrain_data(
        split="train", max_samples=50, seq_len=PRETRAIN_SEQ_LEN,
        tokenizer=tokenizer, skip=200,
    )

    tx = make_decayed_lr_optimizer(
        peak_lr=PRETRAIN_PEAK_LR,
        min_lr=PRETRAIN_MIN_LR,
        warmup_steps=PRETRAIN_WARMUP_STEPS,
        stable_steps=PRETRAIN_STABLE_STEPS,
        decay_steps=PRETRAIN_DECAY_STEPS,
        weight_decay=PRETRAIN_WD,
        b1=PRETRAIN_B1,
        b2=PRETRAIN_B2,
    )
    optimizer  = nnx.Optimizer(model, tx, wrt=nnx.Param)
    moe_layers = collect_moe_layers(model)
    ckpt_mgr   = make_checkpoint_manager(PRETRAIN_CKPT_DIR)

    step = 0
    for _ in range(PHASE1_STEPS):
        for batch_np in make_batches(train_chunks, PRETRAIN_BATCH):
            if step >= PHASE1_STEPS:
                break
            batch = batch_np
            loss  = pretrain_step(model, optimizer, batch)

            # Aux-loss-free bias update runs outside the gradient tape.
            update_moe_biases(moe_layers)

            step += 1

            if step % 100 == 0:
                val_loss, ppl = evaluate_pretrain(
                    model, val_chunks, PRETRAIN_VAL_STEPS, moe_layers
                )
                print(f"  Step {step:5d} | train_loss={float(loss):.4f} | "
                      f"val_loss={val_loss:.4f} | ppl={ppl:.1f}")

            if step % PRETRAIN_CKPT_EVERY == 0:
                save_checkpoint(ckpt_mgr, model, step)

    save_checkpoint(ckpt_mgr, model, step)
    print("Phase 1 complete.\n")


if __name__ == "__main__":
    print("Loading Nemotron tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building model …")
    model = build_model(seed=0)

    run_pretrain_phase1(model, tokenizer)
