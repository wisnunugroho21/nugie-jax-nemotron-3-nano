"""
sft.py — Post-Training Step 1: Supervised Fine-Tuning (SFT)

Paper §3.1: SFT teaches the model reasoning, agentic tool use, instruction
following, safety, and multilingual abilities using ~18 M samples.  Only
assistant/response tokens are included in the loss.

Key design choices:
  • Loss mask: user/system tokens = 0, assistant tokens = 1.
  • Reasoning on/off: 10% of samples have <think> trace stripped (§3.1.5).
  • Budget control: 3% of samples have a truncated reasoning trace (§3.1.5).
  • 13 000 steps at LR 5e-5 with 800-step warmup and MoE load-balance loss.

Run standalone:
    python sft.py

Loads from the latest pretrain checkpoint if PRETRAIN_CKPT_DIR exists.
"""

from flax import nnx
from transformers import AutoTokenizer

from training_shared import (
    PRETRAIN_CKPT_DIR,
    SFT_SEQ_LEN,
    SFT_BATCH,
    SFT_STEPS,
    SFT_LR,
    SFT_MIN_LR,
    SFT_WARMUP_STEPS,
    SFT_WD,
    SFT_B1,
    SFT_B2,
    SFT_CKPT_DIR,
    SFT_CKPT_EVERY,
    build_model,
    collect_moe_layers,
    make_decayed_lr_optimizer,
    load_sft_data,
    make_sft_batches,
    sft_step,
    sft_loss,
    update_moe_biases,
    make_checkpoint_manager,
    save_checkpoint,
    try_load_from_dir,
    NemotronNanoBlock,
)


def run_sft(model: NemotronNanoBlock, tokenizer) -> None:
    """Fine-tune the base model on a supervised chat + reasoning dataset.

    Paper §3.1: 13 000 steps, batch 64, sequence packing to 256 K, LR 5e-5,
    800 warmup steps, MoE load-balance loss coefficient 1e-4.
    """
    print("\n=== Post-Training Step 1: SFT ===")

    train_samples = load_sft_data("train")
    val_samples   = load_sft_data("test")

    tx = make_decayed_lr_optimizer(
        peak_lr=SFT_LR,
        min_lr=SFT_MIN_LR,
        warmup_steps=SFT_WARMUP_STEPS,
        stable_steps=max(1, SFT_STEPS - SFT_WARMUP_STEPS - 1),
        decay_steps=1,
        weight_decay=SFT_WD,
        b1=SFT_B1,
        b2=SFT_B2,
    )
    optimizer  = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ckpt_mgr   = make_checkpoint_manager(SFT_CKPT_DIR)
    moe_layers = collect_moe_layers(model)

    step = 0
    while step < SFT_STEPS:
        for inputs, labels, mask in make_sft_batches(
            train_samples, tokenizer, SFT_BATCH, SFT_SEQ_LEN
        ):
            if step >= SFT_STEPS:
                break
            loss = sft_step(model, optimizer, inputs, labels, mask)
            update_moe_biases(moe_layers)
            step += 1

            if step % 50 == 0:
                val_loss = 0.0
                val_count = 0
                for vinputs, vlabels, vmask in make_sft_batches(
                    val_samples, tokenizer, SFT_BATCH, SFT_SEQ_LEN
                ):
                    val_loss += float(sft_loss(model, vinputs, vlabels, vmask, moe_layers))
                    val_count += 1
                    if val_count >= 10:
                        break
                val_loss /= max(val_count, 1)
                print(f"  SFT step {step:4d} | train_loss={float(loss):.4f} | "
                      f"val_loss={val_loss:.4f}")

            if step % SFT_CKPT_EVERY == 0:
                save_checkpoint(ckpt_mgr, model, step)

    save_checkpoint(ckpt_mgr, model, step)
    print("SFT complete.\n")


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

    run_sft(model, tokenizer)
