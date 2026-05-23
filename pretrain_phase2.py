"""
pretrain_phase2.py — Pre-Training Phase 2: High-Quality Data

Paper §2.3: at the 94% point of training (after Phase 1), the data mixture
shifts to emphasise high-quality sources such as Wikipedia, curated synthetic
datasets, and premium web text.  This final 6% (1.5 T tokens) sharpens the
model's knowledge and reduces noise from web-scale data.

Run standalone:
    python pretrain_phase2.py

Resumes automatically from the latest Phase 1 / Phase 2 checkpoint in
PRETRAIN_CKPT_DIR if one exists.
"""

from flax import nnx
from transformers import AutoTokenizer

from training_shared import (
    PRETRAIN_SEQ_LEN,
    PRETRAIN_BATCH,
    PHASE2_STEPS,
    PRETRAIN_PEAK_LR,
    PRETRAIN_MIN_LR,
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
    try_load_from_dir,
    NemotronNanoBlock,
)


def run_pretrain_phase2(model: NemotronNanoBlock, tokenizer) -> None:
    """Continue pre-training on a high-quality subset.

    Paper §2.3: high-quality sources include Wikipedia, curated synthetics,
    and premium web text.  The LR schedule is re-initialised for the shorter
    Phase 2 budget (in a production run it would be one continuous schedule).
    """
    print("\n=== Pre-Training Phase 2: High-Quality Data ===")

    train_chunks = load_pretrain_data(
        split="train", max_samples=100, seq_len=PRETRAIN_SEQ_LEN,
        tokenizer=tokenizer, skip=500,
    )
    val_chunks = load_pretrain_data(
        split="train", max_samples=30, seq_len=PRETRAIN_SEQ_LEN,
        tokenizer=tokenizer, skip=600,
    )

    phase2_warmup = max(1, PHASE2_STEPS // 10)
    phase2_stable = max(1, PHASE2_STEPS // 2)
    phase2_decay  = max(1, PHASE2_STEPS - phase2_warmup - phase2_stable)

    tx = make_decayed_lr_optimizer(
        peak_lr=PRETRAIN_PEAK_LR,
        min_lr=PRETRAIN_MIN_LR,
        warmup_steps=phase2_warmup,
        stable_steps=phase2_stable,
        decay_steps=phase2_decay,
        weight_decay=PRETRAIN_WD,
        b1=PRETRAIN_B1,
        b2=PRETRAIN_B2,
    )
    optimizer  = nnx.Optimizer(model, tx, wrt=nnx.Param)
    moe_layers = collect_moe_layers(model)
    ckpt_mgr   = make_checkpoint_manager(PRETRAIN_CKPT_DIR)

    step = 0
    for _ in range(PHASE2_STEPS):
        for batch_np in make_batches(train_chunks, PRETRAIN_BATCH):
            if step >= PHASE2_STEPS:
                break
            batch = batch_np
            loss  = pretrain_step(model, optimizer, batch)
            update_moe_biases(moe_layers)
            step += 1

            if step % 100 == 0:
                val_loss, ppl = evaluate_pretrain(
                    model, val_chunks, PRETRAIN_VAL_STEPS, moe_layers
                )
                print(f"  Step {step:4d} | train_loss={float(loss):.4f} | "
                      f"val_loss={val_loss:.4f} | ppl={ppl:.1f}")

            if step % PRETRAIN_CKPT_EVERY == 0:
                save_checkpoint(ckpt_mgr, model, step)

    save_checkpoint(ckpt_mgr, model, step)
    print("Phase 2 complete.\n")


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

    run_pretrain_phase2(model, tokenizer)
