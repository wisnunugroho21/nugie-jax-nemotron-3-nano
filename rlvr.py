"""
rlvr.py — Post-Training Step 2 (and Step 4): RLVR with GRPO

Paper §3.2: Multi-environment Reinforcement Learning from Verifiable Rewards
using synchronous GRPO.  Key design choices:
  • 128 prompts × 16 generations per step (scaled down for demo).
  • Curriculum sampling: difficulty increases from easy to hard (§3.2.2).
  • MoE router weights are frozen during RL to stabilise training (§3.2.5).
  • Overlong filtering: completions that hit the length cap are discarded.
  • Aux-loss-free MoE bias update is kept active during RL.

RLVR is applied twice in the full pipeline:
  Step 2 — immediately after SFT.
  Step 4 — after RLHF, for further refinement.

Run standalone:
    python rlvr.py

Loads from the latest SFT checkpoint (SFT_CKPT_DIR) if available.
"""

import numpy as np
from flax import nnx
from transformers import AutoTokenizer

from training_shared import (
    SFT_CKPT_DIR,
    RLVR_NUM_PROMPTS,
    RLVR_NUM_GENERATIONS,
    RLVR_STEPS,
    RLVR_MAX_NEW_TOKENS,
    RLVR_FREEZE_ROUTER,
    RLVR_LR,
    RLVR_WD,
    RLVR_B1,
    RLVR_B2,
    RLVR_CKPT_DIR,
    RLVR_CKPT_EVERY,
    RL_TRAIN_SEQ_LEN,
    build_model,
    collect_moe_layers,
    make_constant_lr_optimizer,
    load_sft_data,
    generate_completion_tokens,
    compute_verifiable_reward,
    compute_grpo_advantages,
    curriculum_sample,
    build_grpo_batch,
    rl_step,
    compute_log_probs,
    update_moe_biases,
    make_checkpoint_manager,
    save_checkpoint,
    try_load_from_dir,
    NemotronNanoBlock,
    SparseMoE,
)


def snapshot_router_kernels(moe_layers: list[SparseMoE]) -> list:
    """Copy router kernels so they can be restored after an optimizer step."""
    return [moe.router.kernel.get_value() for moe in moe_layers]


def restore_router_kernels(moe_layers: list[SparseMoE], kernels: list) -> None:
    """Restore router kernels to enforce router freezing during RLVR."""
    for moe, kernel in zip(moe_layers, kernels):
        moe.router.kernel.set_value(kernel)


def run_rlvr(model: NemotronNanoBlock, tokenizer) -> None:
    """Multi-environment RLVR using synchronous GRPO.

    Paper §3.2: We train on all RL environments simultaneously using GRPO.
    Multi-environment training produces stable, uniform gains across benchmarks.
    We use GSM8K math as a stand-in for all environments.
    """
    print("\n=== Post-Training Step 2: RLVR (GRPO) ===")

    train_samples = load_sft_data("train")
    moe_layers = collect_moe_layers(model)

    # Frozen copy of the current model as reference policy for KL penalty.
    graphdef, ref_state = nnx.split(model)
    ref_model = nnx.merge(graphdef, ref_state)

    tx        = make_constant_lr_optimizer(RLVR_LR, RLVR_WD, RLVR_B1, RLVR_B2)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ckpt_mgr  = make_checkpoint_manager(RLVR_CKPT_DIR)

    # Initial pass-rates for curriculum sampling (0.5 = medium difficulty).
    pass_rates = np.full(len(train_samples), 0.5, dtype=np.float32)

    for step in range(RLVR_STEPS):
        batch_samples = curriculum_sample(
            train_samples, pass_rates, step, RLVR_STEPS, RLVR_NUM_PROMPTS
        )

        prompt_ids_list: list[list[int]] = []
        completion_groups: list[list[list[int]]] = []
        reward_groups: list[list[float]] = []

        for sample in batch_samples:
            prompt_text = f"User: {sample['question']}\nAssistant: "
            p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_ids_list.append(p_ids)

            completions: list[list[int]] = []
            rewards: list[float] = []
            ground_truth = sample["answer"].split("####")[-1].strip()

            for g in range(RLVR_NUM_GENERATIONS):
                comp_ids = generate_completion_tokens(
                    model, tokenizer, p_ids, rng_seed=step * 100 + g
                )

                # Overlong filtering (§3.2.5): discard completions at the cap.
                if len(comp_ids) >= RLVR_MAX_NEW_TOKENS:
                    comp_ids = []
                    rewards.append(0.0)
                else:
                    comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                    rewards.append(compute_verifiable_reward(comp_text, ground_truth))

                completions.append(comp_ids)

            completion_groups.append(completions)
            reward_groups.append(rewards)

        advantage_groups = [compute_grpo_advantages(rg) for rg in reward_groups]

        token_ids, masks, advantages = build_grpo_batch(
            prompt_ids_list, completion_groups, advantage_groups,
            seq_len=RL_TRAIN_SEQ_LEN, pad_id=tokenizer.eos_token_id,
        )
        ref_log_probs = compute_log_probs(ref_model, token_ids)
        old_log_probs = compute_log_probs(model, token_ids)

        router_kernels_before = snapshot_router_kernels(moe_layers) if RLVR_FREEZE_ROUTER else None
        loss = rl_step(model, optimizer, token_ids, masks, advantages, ref_log_probs, old_log_probs)
        if router_kernels_before is not None:
            restore_router_kernels(moe_layers, router_kernels_before)

        update_moe_biases(moe_layers)

        # Update per-sample pass-rate estimates for next curriculum step.
        for sample, rewards in zip(batch_samples, reward_groups):
            idx = train_samples.index(sample)
            pass_rates[idx] = float(np.mean([r > 0 for r in rewards]))

        if step % 10 == 0:
            mean_reward = float(np.mean([r for rg in reward_groups for r in rg]))
            print(f"  RLVR step {step:3d} | loss={loss:.4f} | "
                  f"mean_reward={mean_reward:.3f}")

        if step % RLVR_CKPT_EVERY == 0:
            save_checkpoint(ckpt_mgr, model, step)

    save_checkpoint(ckpt_mgr, model, RLVR_STEPS)
    print("RLVR complete.\n")


if __name__ == "__main__":
    print("Loading Nemotron tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building model …")
    model = build_model(seed=0)
    try_load_from_dir(SFT_CKPT_DIR, model, model.config)

    run_rlvr(model, tokenizer)
