"""
Latent-space Mixture-of-Experts (LatentMoE) in JAX/Flax NNX.

Based on "Efficient Large Language Models via Latent Space Routing for
Mixture-of-Experts" (arXiv:2601.18089), as used in Nemotron 3 Super
(arXiv:2604.12374, §2.1.1 and §2.4).

The core efficiency insight is about memory bandwidth, not FLOPs.
In a standard MoE layer, each routed expert has weight matrices of size d×m
(where d is the model dimension and m is the expert hidden dim). Loading those
weights for a dispatched token costs O(d·m) memory reads. When d is large
(e.g. 4096), this becomes the bottleneck — not arithmetic.

LatentMoE fixes this by routing and computing in a compressed latent space ℓ:
  - A shared down-projection W↓ ∈ ℝ^(ℓ×d) maps tokens into latent space first.
  - Each routed expert only has weight matrices of size ℓ×m (not d×m).
    Expert weight loading cost drops by a factor of d/ℓ = α.
  - A shared up-projection W↑ ∈ ℝ^(d×ℓ) maps the aggregate back to full d.

The memory savings from the α-factor are reinvested into the model:
Super uses α times as many experts (N' = 512 instead of 128) and activates
α times as many per token (top-22 instead of ~6), which improves accuracy
without increasing per-token FLOPs relative to the Nano counterpart.
This is why LatentMoE achieves "more with the same budget".

Two parts of the MoE intentionally stay in full d-space:
  - The router: computing per-expert sigmoid scores is cheap (one linear layer),
    and keeping it in full d means the gating signal is maximally information-rich.
    Projecting to ℓ first would discard representational detail that helps route
    tokens to the right specialist experts.
  - Shared experts: these run unconditionally on EVERY token, so their weight-
    loading cost is already perfectly amortized across the whole batch.
    The bandwidth argument that motivated latent routing does not apply here,
    and keeping them in full d preserves their modeling capacity.

Routing and load balancing are identical to SparseMoE in moe.py:
  - Sigmoid gate scores (experts score independently, no probability competition)
    - Aux-loss-free load balancing via per-expert bias scalars (Wang et al. 2024)
    - Standard load-balancing loss (GShard-style) can be added during training
  - Biased scores for top-k selection; original unbiased scores for gate weights

Nemotron 3 Super hyperparameters (Table 1):
  d_model                  = 4096
  latent_size (ℓ)          = 1024   → α = d/ℓ = 4
  num_experts              = 512    = 128 × α
  top_k                    = 22     ≈ 6  × α  (rounded for HW efficiency)
  num_shared_experts       = 2
  expert_hidden_dim        = 2688   (for routed experts)
  shared_expert_hidden_dim = 5376   = 2 × 2688
"""

import jax
import jax.numpy as jnp
from flax import nnx


# =============================================================================
# LatentMoE Block
# =============================================================================


class LatentMoE(nnx.Module):
    """
    LatentMoE layer for Nemotron 3 Super (arXiv:2604.12374, §2.1.1 and §2.4).

    Routing, gating, and load balancing are identical to SparseMoE in moe.py.
    The key structural difference is where the expert FNN computations happen:

      SparseMoE (Nano):  experts see the full d-dimensional token representation.
      LatentMoE  (Super): experts see a compressed ℓ-dimensional representation.
        - W↓ (down_proj) projects d → ℓ before dispatching to routed experts.
        - Expert weight matrices shrink from (d, m) to (ℓ, m).
        - W↑ (up_proj) projects ℓ → d after aggregating the expert outputs.
        - Granularity is NOT used — latent projection is the efficiency mechanism.

    The shared experts remain in full d-space. See the module docstring above
    for the rationale.

    Args:
        d_model: Full hidden dimension d (e.g. 4096).
        latent_size: Compressed latent dimension ℓ (e.g. 1024). Must be < d_model.
        num_experts: Total routed experts N', already α-scaled (e.g. 512).
        num_shared_experts: Always-on experts in full d-space (e.g. 2).
        top_k: Active routed experts per token K', already α-scaled (e.g. 22).
        expert_hidden_dim: FFN intermediate dimension m for routed experts (e.g. 2688).
        shared_expert_hidden_dim: FFN intermediate dimension for shared experts
            (e.g. 5376 = 2 × expert_hidden_dim). Shared experts get more capacity
            because they process every token without competing for routing slots.
        bias_update_rate: Step size for the aux-loss-free bias update (1e-3 per paper).
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        latent_size: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        shared_expert_hidden_dim: int,
        bias_update_rate: float = 1e-3,
    ):
        assert latent_size < d_model, (
            f"latent_size ({latent_size}) must be strictly less than d_model ({d_model})"
        )
        assert num_experts > 0, "num_experts must be > 0"
        assert top_k > 0, "top_k must be > 0"
        assert top_k <= num_experts, "top_k must be <= num_experts"
        assert num_shared_experts >= 0, "num_shared_experts must be >= 0"

        self.d_model = d_model
        self.latent_size = latent_size
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.expert_hidden_dim = expert_hidden_dim
        self.shared_expert_hidden_dim = shared_expert_hidden_dim
        self.bias_update_rate = bias_update_rate

        # ── Latent projections (shared across all routed experts) ─────────────

        # W↓: compress every token from d → ℓ before it enters a routed expert.
        # This is the key bandwidth-saving step: routed expert weight matrices
        # are ℓ×m instead of d×m, so loading them costs (d/ℓ) × less per token.
        # W↓ is loaded ONCE per forward pass (shared), not per-expert — cheap.
        self.down_proj = nnx.Linear(d_model, latent_size, use_bias=False, rngs=rngs)

        # W↑: decompress the aggregated latent expert output back to full d-space.
        # Applied once after the weighted sum of all selected expert outputs.
        # Same logic: one shared load cost, not one per expert.
        self.up_proj = nnx.Linear(latent_size, d_model, use_bias=False, rngs=rngs)

        # ── Router (deliberately kept in full d-space) ─────────────────────────

        # Routing in full d preserves the richest available signal for gating.
        # If we routed from ℓ instead, we would potentially discard information
        # that distinguishes which specialist expert should handle this token.
        # The cost of one linear(d → num_experts) per step is negligible relative
        # to the expert FFN itself, so there is no efficiency reason to compress it.
        self.router = nnx.Linear(d_model, num_experts, use_bias=False, rngs=rngs)

        # Per-expert load-balancing bias (aux-loss-free, Wang et al. 2024).
        # NOT updated by the gradient optimizer — call update_expert_bias() after
        # each training step instead.  Stored as nnx.Variable (not nnx.Param) so
        # optimizers skip it when they filter by nnx.Param.
        self.expert_bias = nnx.Variable(jnp.zeros(num_experts))

        # Stash routing indices from the most recent forward pass.
        # The training loop reads this to call update_expert_bias() outside the
        # gradient — see SparseMoE.update_expert_bias in moe.py for the full
        # explanation. Placeholder shape here; overwritten on first forward pass.
        self.last_topk_indices = nnx.Variable(
            jnp.zeros((1, top_k), dtype=jnp.int32)
        )

        # Stores the most recent standard load-balancing loss term.
        # Training code can aggregate this across MoE layers and scale it
        # (paper uses coefficient 1e-4).
        self.last_load_balance_loss = nnx.Variable(jnp.array(0.0, dtype=jnp.float32))

        init = nnx.initializers.lecun_normal()

        # ── Routed expert weights (stacked, operate in latent space ℓ) ─────────

        # Pre-stacking all expert matrices along axis-0 lets topk_indices index
        # them directly in __call__ — no per-step stack assembly overhead.
        #
        # routed_W1: (num_experts, latent_size, expert_hidden_dim)
        #   FC1 in each expert: ℓ → m
        # routed_W2: (num_experts, expert_hidden_dim, latent_size)
        #   FC2 in each expert: m → ℓ  (W↑ handles the final ℓ → d step)
        self.routed_W1 = nnx.Param(
            init(rngs.params(), (num_experts, latent_size, expert_hidden_dim))
        )
        self.routed_W2 = nnx.Param(
            init(rngs.params(), (num_experts, expert_hidden_dim, latent_size))
        )

        # ── Shared expert weights (stacked, operate in FULL d-space) ─────────

        # Shared experts are unconditional — they run on 100% of tokens.
        # Their weight loading is already amortized, so the latent projection
        # offers no savings and is omitted. Full d-space also gives them more
        # capacity, which is appropriate since they handle the universal patterns
        # (e.g. structuring a reasoning step) seen across every domain.
        #
        # shared_W1: (num_shared_experts, d_model, shared_expert_hidden_dim)
        # shared_W2: (num_shared_experts, shared_expert_hidden_dim, d_model)
        if num_shared_experts > 0:
            self.shared_W1 = nnx.Param(
                init(rngs.params(), (num_shared_experts, d_model, shared_expert_hidden_dim))
            )
            self.shared_W2 = nnx.Param(
                init(rngs.params(), (num_shared_experts, shared_expert_hidden_dim, d_model))
            )

    def _collect_routed_outputs(
        self,
        x_latent: jax.Array,
        topk_indices: jax.Array,
    ) -> jax.Array:
        """
        Run only the top-k selected routed experts, entirely in latent space.

        The experts never see the full d-dimensional token representation.
        They receive x_latent (shape: num_tokens × ℓ), which has already been
        compressed by W↓.  This is the memory-bandwidth saving step:
        loading W1_sel and W2_sel costs ℓ×m reads per expert instead of d×m.

        Args:
            x_latent:    (num_tokens, latent_size)   — compressed token representations.
            topk_indices: (num_tokens, top_k)         — which experts each token selected.
        Returns:
            routed_out:  (num_tokens, top_k, latent_size) — expert outputs in latent space.
                         The caller applies W↑ (up_proj) to project back to d.
        """
        W1 = self.routed_W1.get_value()  # (num_experts, latent_size, expert_hidden_dim)
        W2 = self.routed_W2.get_value()  # (num_experts, expert_hidden_dim, latent_size)

        # Gather the weight matrices of each token's top-k selected experts.
        # W1_sel: (num_tokens, top_k, latent_size, expert_hidden_dim)
        # W2_sel: (num_tokens, top_k, expert_hidden_dim, latent_size)
        W1_sel = W1[topk_indices]
        W2_sel = W2[topk_indices]

        # FC1: "tl, tklh -> tkh"
        #   t = num_tokens, l = latent_size, k = top_k, h = expert_hidden_dim
        #   For each token t and each of its selected experts k,
        #   dot x_latent[t] (size ℓ) with W1_sel[t,k] (size ℓ×h) → hidden h.
        h = jnp.einsum("tl,tklh->tkh", x_latent, W1_sel)

        # Squared-ReLU activation: relu(x)^2 — consistent with SparseMoE and the paper.
        h = jax.nn.relu(h)
        h = h * h

        # FC2: "tkh, tkhl -> tkl"
        #   Projects hidden vectors (size h) back to latent space (size ℓ).
        #   Output stays in latent space; W↑ handles the final ℓ → d step.
        out = jnp.einsum("tkh,tkhl->tkl", h, W2_sel)
        return out  # (num_tokens, top_k, latent_size)

    def _collect_shared_outputs(self, x_flat: jax.Array) -> jax.Array:
        """
        Run all shared experts on every token via a single batched einsum.

        Shared experts are unconditional and operate in full d-space.
        All experts run in parallel using pre-stacked weight matrices.

        Args:
            x_flat: (num_tokens, d_model)
        Returns:
            shared_out: (num_tokens, num_shared_experts, d_model),
                        or (num_tokens, 0, d_model) when there are no shared experts.
        """
        if self.num_shared_experts == 0:
            return jnp.zeros((x_flat.shape[0], 0, self.d_model), dtype=x_flat.dtype)

        # "td, edh -> teh": for each of the E shared experts, project every token
        # from d_model up to shared_expert_hidden_dim simultaneously.
        h = jnp.einsum("td,edh->teh", x_flat, self.shared_W1.get_value())
        # Squared-ReLU — same activation as routed experts.
        h = jax.nn.relu(h)
        h = h * h
        # "teh, ehd -> ted": project all shared expert hidden states back to d_model.
        return jnp.einsum("teh,ehd->ted", h, self.shared_W2.get_value())

    def update_expert_bias(self, topk_indices: jax.Array) -> None:
        """
        Update the per-expert load-balancing bias after each training step.

        Identical algorithm to SparseMoE.update_expert_bias in moe.py.
        See that function for the full explanation of aux-loss-free balancing.

        IMPORTANT: Call this AFTER the optimizer step, outside the gradient tape.
        expert_bias is stored as nnx.Variable (not nnx.Param), so the optimizer
        already skips it — you just need to call this function manually each step.

        Args:
            topk_indices: (num_tokens, top_k) from the most recent forward pass.
        """
        num_tokens = topk_indices.shape[0]

        # Count selections per expert over the current batch.
        # one_hot: (num_tokens, top_k, num_experts) → sum to (num_experts,)
        actual_count = jnp.sum(
            jax.nn.one_hot(topk_indices, self.num_experts),
            axis=(0, 1),
        )

        # Ideal count if load were perfectly balanced across all experts.
        expected_count = num_tokens * self.top_k / self.num_experts

        # sign(actual - expected):
        #   +1 (overloaded)  → decrease bias → harder to pick next time
        #   -1 (underloaded) → increase bias → easier to pick next time
        #    0 (balanced)    → no change
        self.expert_bias.set_value(
            self.expert_bias.get_value()
            - self.bias_update_rate * jnp.sign(actual_count - expected_count)
        )

    def _standard_load_balance_loss(
        self,
        router_scores: jax.Array,
        topk_indices: jax.Array,
    ) -> jax.Array:
        """Compute a standard MoE load-balancing loss term.

        This complements the bias-based routing update and can be scaled in the
        global training objective (paper coefficient: 1e-4).

        Args:
            router_scores: (num_tokens, num_experts) sigmoid scores.
            topk_indices: (num_tokens, top_k) selected experts.
        Returns:
            Scalar load-balancing loss.
        """
        # Fraction of top-k assignment slots used by each expert.
        expert_load = jnp.mean(
            jax.nn.one_hot(topk_indices, self.num_experts, dtype=router_scores.dtype),
            axis=(0, 1),
        )  # (num_experts,)

        # Mean router score per expert over tokens.
        expert_importance = jnp.mean(router_scores, axis=0)  # (num_experts,)

        # GShard-style balancing term.
        return self.num_experts * jnp.sum(expert_load * expert_importance)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through the LatentMoE layer.

        Full data flow:
            x  (batch, seq, d)
            → router(x) in d-space → sigmoid scores → bias-nudged top-k selection
            → down_proj(x) → x_latent  (batch·seq, ℓ)
            → top-k routed experts run in ℓ-space
            → weighted aggregate in ℓ-space
            → up_proj → (batch·seq, d)
            → + shared_experts(x) in d-space (unconditional)
            → output  (batch, seq, d)

        The routing decision (top-k selection, gate weight construction) always
        uses x in full d-space. Only the expert FFN forward passes see the latent
        projection — this is intentional (see module docstring for the rationale).

        Args:
            x: (batch, seqlen, d_model)
        Returns:
            y: (batch, seqlen, d_model)
        """
        batch, seqlen, d_model = x.shape
        assert d_model == self.d_model, "Input d_model does not match LatentMoE config"

        # Routing is purely token-wise — flatten batch and sequence together.
        num_tokens = batch * seqlen
        x_flat = jnp.reshape(x, (num_tokens, d_model))  # (num_tokens, d_model)

        # ── Routing (in full d-space) ──────────────────────────────────────────

        # One sigmoid score per expert per token. Scores are independent
        # (sigmoid, not softmax), so experts do not compete for probability mass.
        router_logits = self.router(x_flat)            # (num_tokens, num_experts)
        router_scores = jax.nn.sigmoid(router_logits)  # independent per-expert scores

        # Add the load-balancing bias to steer selection toward under-used experts.
        # Bias is ONLY used for top-k selection — not for the actual gate weights.
        biased_scores = router_scores + self.expert_bias.get_value()

        # Top-k selection uses biased scores for load-aware routing.
        _, topk_indices = jax.lax.top_k(biased_scores, self.top_k)
        # topk_indices: (num_tokens, top_k)

        # Store indices so the training loop can call update_expert_bias() after
        # the optimizer step without needing to re-run the forward pass.
        self.last_topk_indices.set_value(topk_indices)

        # Store standard load-balancing loss from this forward pass.
        self.last_load_balance_loss.set_value(
            self._standard_load_balance_loss(router_scores, topk_indices)
        )

        # Gate weights use UNBIASED scores — the bias is a routing hint, not a
        # magnitude signal. Using original scores preserves each expert's
        # learned output scale.
        token_ids = jnp.arange(num_tokens)[:, None]         # (num_tokens, 1)
        selected_scores = router_scores[token_ids, topk_indices]  # (num_tokens, top_k)

        # Renormalize so selected gate weights sum to 1 per token.
        # Keeps output scale stable regardless of the raw sigmoid values.
        selected_gates = selected_scores / (
            jnp.sum(selected_scores, axis=-1, keepdims=True) + 1e-6
        )  # (num_tokens, top_k)

        # ── Routed path (experts run in latent space ℓ) ───────────────────────

        # Compress each token into the latent space before dispatch.
        # Routed expert weight matrices are ℓ×m (not d×m), so loading them
        # for the dispatched tokens costs (d/ℓ) × less memory bandwidth.
        x_latent = self.down_proj(x_flat)  # (num_tokens, latent_size)

        # Run the top-k selected experts on the latent representation.
        # routed_out: (num_tokens, top_k, latent_size)
        routed_out = self._collect_routed_outputs(x_latent, topk_indices)

        # Weighted sum of selected expert outputs — still in latent space ℓ.
        # selected_gates[:, :, None] broadcasts across the latent dimension.
        routed_mix_latent = jnp.sum(
            routed_out * selected_gates[:, :, None], axis=1
        )  # (num_tokens, latent_size)

        # Project the aggregated latent output back to full d-space.
        # One up-projection for the combined output of all selected experts.
        routed_mix = self.up_proj(routed_mix_latent)  # (num_tokens, d_model)

        # ── Shared path (experts run in full d-space) ─────────────────────────

        if self.num_shared_experts > 0:
            # Shared experts run unconditionally on every token.
            # shared_out: (num_tokens, num_shared_experts, d_model)
            shared_out = self._collect_shared_outputs(x_flat)

            # Sum each shared expert's contribution directly.
            shared_mix = jnp.sum(shared_out, axis=1)  # (num_tokens, d_model)
            y_flat = routed_mix + shared_mix
        else:
            y_flat = routed_mix

        # Restore the original (batch, seqlen, d_model) shape.
        return jnp.reshape(y_flat, (batch, seqlen, d_model))
