"""
Minimal DeepSeekMoE-style Sparse Mixture-of-Experts (MoE) in JAX/Flax NNX.

This module keeps the code simple and educational while reflecting the key ideas:
- Granular routed experts: split base routed experts into finer experts.
- Shared experts: always-on experts, isolated from routing competition.
- Routed-only top-k sparse routing with softmax probabilities.
- Squared-ReLU experts.
- Simple Switch-style load-balancing auxiliary loss on routed experts.

The implementation is explicit (loop-based) for readability, not performance.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class MoEExpert(nnx.Module):
    """
    A single FFN expert using Squared-ReLU.

    Structure:
        x -> Linear(d_model, hidden_dim) -> SquaredReLU -> Linear(hidden_dim, d_model)
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        hidden_dim: int,
        use_bias: bool = False,
    ):
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.fc1 = nnx.Linear(
            self.d_model, self.hidden_dim, use_bias=use_bias, rngs=rngs
        )
        self.fc2 = nnx.Linear(
            self.hidden_dim, self.d_model, use_bias=use_bias, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Squared-ReLU is common in modern MoE variants.
        h = self.fc1(x)
        h = jax.nn.relu(h)
        h = h * h
        return self.fc2(h)


class SparseMoE(nnx.Module):
    """
    Minimal sparse MoE with granular routed experts and isolated shared experts.

    Granular MoE idea (DeepSeekMoE style):
    - Start with base routed experts (num_experts).
    - Split each into `granularity_factor` smaller experts.
    - Effective routed experts = num_experts * granularity_factor.
    - Optionally scale top-k by granularity.
    - Reduce each routed expert hidden size by the same factor.

    Shared experts:
    - Always active for all tokens.
    - Never participate in router competition.
    - Added as a deterministic shared pathway.

    Args:
        granularity_factor:
            1 keeps standard MoE behavior.
            >1 enables granular routed expert segmentation.
        scale_top_k_with_granularity:
            If True, effective routed top-k = top_k * granularity_factor.
            If False, effective routed top-k = top_k.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_experts: int,
        num_shared_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        use_bias: bool = False,
        granularity_factor: int = 1,
        scale_top_k_with_granularity: bool = True,
    ):
        self.d_model = d_model

        # Keep base values for compatibility/debug clarity.
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.expert_hidden_dim = expert_hidden_dim

        self.granularity_factor = granularity_factor
        self.scale_top_k_with_granularity = scale_top_k_with_granularity

        assert self.num_experts > 0, "num_experts must be > 0"
        assert self.top_k > 0, "top_k must be > 0"
        assert self.top_k <= self.num_experts, "top_k must be <= num_experts"
        assert self.num_shared_experts >= 0, "num_shared_experts must be >= 0"
        assert self.granularity_factor > 0, "granularity_factor must be > 0"

        # Effective routed expert count after granular segmentation.
        self.num_routed_experts = self.num_experts * self.granularity_factor

        # Effective top-k for routed experts.
        if self.scale_top_k_with_granularity:
            self.routed_top_k = self.top_k * self.granularity_factor
        else:
            self.routed_top_k = self.top_k

        assert self.routed_top_k <= self.num_routed_experts, (
            "effective routed top-k must be <= num_routed_experts"
        )

        # Granular routed experts are smaller to keep parameter growth controlled.
        self.routed_expert_hidden_dim = max(
            1, self.expert_hidden_dim // self.granularity_factor
        )

        # Shared experts keep full hidden size.
        self.shared_expert_hidden_dim = self.expert_hidden_dim

        # Router maps tokens to routed experts ONLY.
        self.router = nnx.Linear(
            self.d_model,
            self.num_routed_experts,
            use_bias=use_bias,
            rngs=rngs,
        )

        # Routed granular experts.
        for i in range(self.num_routed_experts):
            setattr(
                self,
                f"routed_expert_{i}",
                MoEExpert(
                    d_model=self.d_model,
                    hidden_dim=self.routed_expert_hidden_dim,
                    use_bias=use_bias,
                    rngs=rngs,
                ),
            )

        # Shared always-on experts.
        for i in range(self.num_shared_experts):
            setattr(
                self,
                f"shared_expert_{i}",
                MoEExpert(
                    d_model=self.d_model,
                    hidden_dim=self.shared_expert_hidden_dim,
                    use_bias=use_bias,
                    rngs=rngs,
                ),
            )

    def _collect_routed_outputs(self, x_flat: jax.Array) -> jax.Array:
        """
        Runs all routed experts and stacks outputs.

        Args:
            x_flat: (num_tokens, d_model)
        Returns:
            routed_outputs: (num_tokens, num_routed_experts, d_model)
        """
        outputs = []
        for i in range(self.num_routed_experts):
            expert = getattr(self, f"routed_expert_{i}")
            outputs.append(expert(x_flat))
        return jnp.stack(outputs, axis=1)

    def _collect_shared_outputs(self, x_flat: jax.Array) -> jax.Array:
        """
        Runs all shared experts and stacks outputs.

        Shared experts are always-on deterministic experts (not routed).

        Args:
            x_flat: (num_tokens, d_model)
        Returns:
            shared_outputs: (num_tokens, num_shared_experts, d_model)
        """
        if self.num_shared_experts == 0:
            return jnp.zeros((x_flat.shape[0], 0, self.d_model), dtype=x_flat.dtype)

        outputs = []
        for i in range(self.num_shared_experts):
            expert = getattr(self, f"shared_expert_{i}")
            outputs.append(expert(x_flat))
        return jnp.stack(outputs, axis=1)

    def _load_balancing_aux_loss(
        self, routed_probs: jax.Array, topk_indices: jax.Array
    ) -> jax.Array:
        """
        Simple Switch-style load-balancing loss for routed experts only.

        We combine:
        1) Dispatch fraction: empirical token assignments per routed expert.
        2) Mean router probability: expected mass per routed expert.

        Args:
            routed_probs: softmax probabilities over routed experts,
                          shape (num_tokens, num_routed_experts)
            topk_indices: selected routed experts per token,
                          shape (num_tokens, routed_top_k)
        Returns:
            aux_loss: scalar
        """
        num_tokens = routed_probs.shape[0]

        dispatch_mask = jnp.zeros_like(routed_probs)
        token_ids = jnp.arange(num_tokens)[:, None]
        dispatch_mask = dispatch_mask.at[token_ids, topk_indices].set(1.0)

        dispatch_fraction = jnp.mean(dispatch_mask / self.routed_top_k, axis=0)
        mean_router_prob = jnp.mean(routed_probs, axis=0)

        aux_loss = self.num_routed_experts * jnp.sum(
            dispatch_fraction * mean_router_prob
        )
        return aux_loss

    def __call__(
        self, x: jax.Array, return_aux_loss: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """
        Args:
            x: (batch, seqlen, d_model)
            return_aux_loss: if True, also return routed balancing auxiliary loss

        Returns:
            y: (batch, seqlen, d_model)
            aux_loss (optional): scalar
        """
        batch, seqlen, d_model = x.shape
        assert d_model == self.d_model, "Input d_model does not match MoE config"

        # Flatten sequence tokens so routing is token-wise.
        num_tokens = batch * seqlen
        x_flat = jnp.reshape(x, (num_tokens, d_model))

        # 1) Routed path: softmax routing over routed experts only.
        routed_logits = self.router(x_flat)
        routed_probs = jax.nn.softmax(routed_logits, axis=-1)

        # 2) Sparse top-k selection among routed experts.
        # We keep softmax weights of selected experts directly (no post-top-k renorm).
        topk_values, topk_indices = jax.lax.top_k(routed_probs, self.routed_top_k)

        aux_loss = self._load_balancing_aux_loss(routed_probs, topk_indices)

        routed_gates = jnp.zeros_like(routed_probs)
        token_ids = jnp.arange(num_tokens)[:, None]
        routed_gates = routed_gates.at[token_ids, topk_indices].set(topk_values)

        routed_outputs = self._collect_routed_outputs(x_flat)
        routed_mix = jnp.sum(routed_outputs * routed_gates[:, :, None], axis=1)

        # 3) Shared path: always-on deterministic experts.
        if self.num_shared_experts > 0:
            shared_outputs = self._collect_shared_outputs(x_flat)
            # Sum shared experts directly to form a shared capacity pathway.
            shared_mix = jnp.sum(shared_outputs, axis=1)
            y_flat = routed_mix + shared_mix
        else:
            y_flat = routed_mix

        y = jnp.reshape(y_flat, (batch, seqlen, d_model))
        if return_aux_loss:
            return y, aux_loss
        return y
