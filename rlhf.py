"""
rlhf.py — Post-Training Step 3: RLHF with GenRM

Paper §3.3: after RLVR, RLHF is applied to improve behaviour on chat-style
tasks where rewards are harder to verify automatically.  A Generative Reward
Model (GenRM) scores responses; Group Relative Length Control (GRLC) penalises
verbosity relative to the group, and a Quality-Gated Conciseness Bonus rewards
the shortest high-quality response.

Key design choices:
  • Circular pairwise comparison to score N responses in O(N) GenRM calls.
  • Group Relative Length Control (§3.3.2, Eq. 6) — zero-sum length penalty.
  • Quality-Gated Conciseness Bonus — bonus for shortest high-quality response.
  • Same GRPO update as RLVR with KL penalty anchored to the RLVR checkpoint.

Run standalone:
    python rlhf.py

Loads from the latest RLVR checkpoint (RLVR_CKPT_DIR) if available.
"""

import numpy as np
from flax import nnx
from transformers import AutoTokenizer

from training_shared import (
    RLVR_CKPT_DIR,
    RLHF_NUM_PROMPTS,
    RLHF_NUM_RESPONSES,
    RLHF_LAMBDA_THINK,
    RLHF_LAMBDA_ANSWER,
    RLHF_BETA_THINK,
    RLHF_BETA_ANSWER,
    RLHF_PERCENTILE,
    RLHF_LR,
    RLHF_WD,
    RLHF_B1,
    RLHF_B2,
    RLHF_STEPS,
    RLHF_CKPT_DIR,
    RLHF_CKPT_EVERY,
    RL_TRAIN_SEQ_LEN,
    build_model,
    collect_moe_layers,
    make_constant_lr_optimizer,
    load_sft_data,
    generate_completion_tokens,
    compute_grpo_advantages,
    build_grpo_batch,
    rl_step,
    compute_log_probs,
    update_moe_biases,
    make_checkpoint_manager,
    save_checkpoint,
    try_load_from_dir,
    NemotronNanoBlock,
)


def simulated_genrm_score(response_text: str, prompt: str) -> float:
    """Simulate a GenRM helpfulness score in [1, 5].

    Paper §3.3.1: the real GenRM is a large LLM (Qwen3-235B) trained to
    evaluate pairwise response quality.  We approximate with heuristic proxies.
    """
    score = 1.0
    if "<think>" in response_text:
        score += 1.0
    if "</think>" in response_text:
        score += 1.0
    if len(response_text.split()) > 10:
        score += 1.0
    return min(score, 5.0)


def apply_group_relative_length_control(
    base_scores: list[float],
    responses: list[str],
) -> list[float]:
    """Adjust base scores with Group Relative Length Control (§3.3.2, Eq. 6).

    The correction is zero-sum within the group: penalising long responses
    automatically rewards shorter ones.  A Quality-Gated Conciseness Bonus
    is additionally awarded to the shortest response that meets the quality bar.
    """
    N = len(responses)
    if N == 0:
        return base_scores

    think_lens:  list[int] = []
    answer_lens: list[int] = []
    for r in responses:
        if "</think>" in r:
            think_part  = r.split("</think>")[0]
            answer_part = r.split("</think>")[1]
        else:
            think_part  = ""
            answer_part = r
        think_lens. append(len(think_part.split()))
        answer_lens.append(len(answer_part.split()))

    think_arr  = np.array(think_lens,  dtype=np.float32)
    answer_arr = np.array(answer_lens, dtype=np.float32)

    def centered_weights(lengths: np.ndarray) -> np.ndarray:
        lo, hi = lengths.min(), lengths.max()
        if hi == lo:
            return np.zeros_like(lengths)
        w = 1.0 - (lengths - lo) / (hi - lo)
        return w - w.mean()

    w_think  = centered_weights(think_arr)
    w_answer = centered_weights(answer_arr)

    scores = np.array(base_scores, dtype=np.float32)
    scores += RLHF_LAMBDA_THINK  * w_think
    scores += RLHF_LAMBDA_ANSWER * w_answer

    threshold = float(np.percentile(base_scores, RLHF_PERCENTILE))

    min_think_idx  = int(np.argmin(think_arr))
    min_answer_idx = int(np.argmin(answer_arr))

    if base_scores[min_think_idx]  >= threshold:
        scores[min_think_idx]  += RLHF_BETA_THINK

    if base_scores[min_answer_idx] >= threshold:
        scores[min_answer_idx] += RLHF_BETA_ANSWER

    return scores.tolist()


def circular_pairwise_comparison(
    responses: list[str],
    prompt: str,
) -> list[float]:
    """Score all responses using O(N) circular comparison (§3.3.2).

    Each response appears in exactly two comparisons, providing an unbiased
    score while avoiding the O(N²) cost of all-pairs comparison.
    """
    N = len(responses)
    accumulated = np.zeros(N, dtype=np.float32)

    for i in range(N):
        j = (i + 1) % N
        s_i = simulated_genrm_score(responses[i], prompt)
        s_j = simulated_genrm_score(responses[j], prompt)

        if abs(s_i - s_j) < 0.01:
            sr = 3.5
            s_i += 3.5 - sr
            s_j += sr  - 3.5

        accumulated[i] += s_i
        accumulated[j] += s_j

    return (accumulated / 2.0).tolist()


def run_rlhf(model: NemotronNanoBlock, tokenizer) -> None:
    """RLHF using a Generative Reward Model (GenRM) with Group Relative Length Control.

    Paper §3.3: applies GRPO with GenRM-scored rewards and length control to
    improve chat-style behaviour.  Verbosity decreases ~30% without accuracy loss.
    """
    print("\n=== Post-Training Step 3: RLHF with GenRM ===")

    train_samples = load_sft_data("train")

    # Frozen copy of the RLVR checkpoint as reference policy for KL penalty.
    graphdef, ref_state = nnx.split(model)
    ref_model = nnx.merge(graphdef, ref_state)

    tx        = make_constant_lr_optimizer(RLHF_LR, RLHF_WD, RLHF_B1, RLHF_B2)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    ckpt_mgr  = make_checkpoint_manager(RLHF_CKPT_DIR)

    for step in range(RLHF_STEPS):
        batch_samples = train_samples[
            step * RLHF_NUM_PROMPTS : (step + 1) * RLHF_NUM_PROMPTS
        ]
        if not batch_samples:
            break

        prompt_ids_list: list[list[int]] = []
        completion_groups: list[list[list[int]]] = []
        reward_groups: list[list[float]] = []

        for sample in batch_samples:
            prompt_text = f"User: {sample['question']}\nAssistant: "
            p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_ids_list.append(p_ids)

            responses_text: list[str] = []
            completions:    list[list[int]] = []
            for g in range(RLHF_NUM_RESPONSES):
                comp_ids  = generate_completion_tokens(
                    model, tokenizer, p_ids,
                    max_new_tokens=200, rng_seed=step * 50 + g,
                )
                comp_text = tokenizer.decode(comp_ids, skip_special_tokens=True)
                responses_text.append(comp_text)
                completions.append(comp_ids)

            base_scores     = circular_pairwise_comparison(responses_text, prompt_text)
            adjusted_scores = apply_group_relative_length_control(base_scores, responses_text)

            completion_groups.append(completions)
            reward_groups.append(adjusted_scores)

        advantage_groups = [compute_grpo_advantages(rg) for rg in reward_groups]
        token_ids, masks, advantages = build_grpo_batch(
            prompt_ids_list, completion_groups, advantage_groups,
            seq_len=RL_TRAIN_SEQ_LEN, pad_id=tokenizer.eos_token_id,
        )
        ref_log_probs = compute_log_probs(ref_model, token_ids)
        old_log_probs = compute_log_probs(model, token_ids)
        loss = rl_step(model, optimizer, token_ids, masks, advantages, ref_log_probs, old_log_probs)
        update_moe_biases(collect_moe_layers(model))

        if step % 10 == 0:
            mean_score = float(np.mean([s for rg in reward_groups for s in rg]))
            print(f"  RLHF step {step:3d} | loss={loss:.4f} | "
                  f"mean_genrm_score={mean_score:.3f}")

        if step % RLHF_CKPT_EVERY == 0:
            save_checkpoint(ckpt_mgr, model, step)

    save_checkpoint(ckpt_mgr, model, RLHF_STEPS)
    print("RLHF complete.\n")


if __name__ == "__main__":
    print("Loading Nemotron tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building model …")
    model = build_model(seed=0)
    try_load_from_dir(RLVR_CKPT_DIR, model, model.config)

    run_rlhf(model, tokenizer)
