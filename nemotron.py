"""
Minimal Nemotron-style Hybrid Model in JAX/Flax NNX.

Reference inspiration:
"Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid
 Mamba-Transformer Model for Agentic Reasoning"

What this minimal implementation keeps from the paper:
- Hybrid stack: Mamba mixer + Attention mixer (alternating pattern by default)
- MoE after each mixer block
- Sparse top-k MoE routing with shared experts
- Squared-ReLU experts
- RMSNorm + residual pre-norm structure
- No positional embeddings, no dropout, and bias-free linear layers by default

What is intentionally simplified:
- Tiny default dimensions for local experimentation
- Alternating layer pattern instead of large paper-scale block scheduling
- No distributed/expert-parallel optimization
- Uses a simple auxiliary load-balancing loss (minimal implementation)
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from attention import GroupedQueryAttention
from mamba_2 import Mamba2Block
from moe import SparseMoE


class RMSNorm(nnx.Module):
    """
    Minimal RMSNorm implementation.

    RMSNorm normalizes by root-mean-square instead of mean+variance, which is
    the normalization style used in the Nemotron architecture description.
    """

    def __init__(self, dim: int, rngs: nnx.Rngs, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.scale = nnx.Param(jnp.ones((dim,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.scale[...]


@dataclass
class NemotronConfig:
    """
    Config with tiny local defaults.

    Notes:
    - Defaults are intentionally small for easy local runs.
    - Paper-like behavior is preserved as configurable knobs.
    """

    # Token/model sizes
    vocab_size: int = 1000
    d_model: int = 128

    patterns: list[tuple[str, int]] = field(
        default_factory=[
            ("mamba_moe", 2),
            ("mamba_attention_moe", 1),
            ("mamba_moe", 2),
            ("mamba_attention_moe", 1),
            ("mamba_moe", 2),
            ("mamba_attention_moe", 1),
            ("mamba_moe", 1)
        ]
    )

    # Attention (GQA)
    num_attention_heads: int = 4
    num_kv_heads: int = 1
    attention_head_dim: int = 32

    # Mamba-2 settings (reusing existing Mamba2Block implementation)
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 64

    # MoE settings
    num_experts: int = 4
    num_shared_experts: int = 1
    top_k: int = 2
    expert_hidden_dim: int = 256
    granularity_factor: int = 1
    scale_top_k_with_granularity: bool = True
    moe_aux_loss_weight: float = 1e-4

    # Normalization and numerical stability
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_preset(cls, preset: str = "tiny") -> "NemotronConfig":
        """
        Builds a config from a named preset.

        Presets:
        - tiny: default local-friendly profile (fallback)
        - paper_close: larger profile that is closer to Nemotron-3-Nano style

        The paper_close preset increases attention heads, expert count, and
        uses top_k=6 routing with stronger granular MoE settings while still
        keeping this implementation simple.
        """
        key = preset.strip().lower()

        if key in ("tiny", "default"):
            return cls(
                patterns=[
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 1)
                ]
            )

        if key in ("paper_close", "paper-close", "paper"):
            return cls(
                # Bigger model than tiny defaults.
                d_model=2048,
                patterns=[
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe", 2),
                    ("mamba_attention_moe", 1),

                    ("mamba_moe", 3),

                    ("mamba_attention_moe", 1),

                    ("mamba_moe", 4),
                ],
                # Closer to paper-style GQA shape choices.
                num_attention_heads=32,
                num_kv_heads=2,
                attention_head_dim=64,
                # Closer to paper-style Mamba settings.
                mamba_d_state=128,
                mamba_d_conv=4,
                mamba_expand=2,
                mamba_headdim=64,
                mamba_ngroups=8,
                mamba_chunk_size=64,
                # Closer to paper-style MoE settings.
                num_experts=64,
                num_shared_experts=2,
                top_k=6,
                expert_hidden_dim=1856,
                granularity_factor=2,
                scale_top_k_with_granularity=True,
                moe_aux_loss_weight=1e-4,
                rms_norm_eps=1e-6,
            )

        raise ValueError(
            f"Unknown preset '{preset}'. Supported presets: tiny, paper_close"
        )

    def validate(self) -> None:
        """Checks shape constraints that must hold for this architecture."""
        assert len(self.patterns) > 0, "patterns cannot be empty"

        # Attention output must map cleanly back to d_model.
        assert self.num_attention_heads * self.attention_head_dim == self.d_model, (
            "d_model must equal num_attention_heads * attention_head_dim"
        )
        assert self.num_attention_heads % self.num_kv_heads == 0, (
            "num_attention_heads must be divisible by num_kv_heads"
        )

        # Mamba internal shape constraints.
        mamba_d_inner = self.mamba_expand * self.d_model
        assert mamba_d_inner % self.mamba_headdim == 0, (
            "(mamba_expand * d_model) must be divisible by mamba_headdim"
        )
        mamba_nheads = mamba_d_inner // self.mamba_headdim
        assert mamba_nheads % self.mamba_ngroups == 0, (
            "Mamba nheads must be divisible by mamba_ngroups"
        )

        # MoE routing constraints.
        assert self.top_k > 0, "top_k must be > 0"
        assert self.top_k <= self.num_experts, "top_k must be <= num_experts"
        assert self.granularity_factor > 0, "granularity_factor must be > 0"

        effective_num_routed_experts = self.num_experts * self.granularity_factor
        if self.scale_top_k_with_granularity:
            effective_top_k = self.top_k * self.granularity_factor
        else:
            effective_top_k = self.top_k
        assert effective_top_k <= effective_num_routed_experts, (
            "effective routed top-k must be <= effective routed experts"
        )


class MambaMoEBlock(nnx.Module):
    """
    Hybrid Mamba & MoE block.

    Block structure (pre-norm residual):
      x = x + Mamba(RMSNorm(x))
      x = x + MoE(RMSNorm(x))

    This keeps the architecture simple and easy to inspect.
    """

    def __init__(
            self,
            rngs: nnx.Rngs,
            config: NemotronConfig
    ):
        self.norm_mamba = RMSNorm(config.d_model, eps=config.rms_norm_eps, rngs=rngs)
        self.norm_moe = RMSNorm(config.d_model, eps=config.rms_norm_eps, rngs=rngs)

        # Reuse the already-implemented Mamba-2 block
        self.mamba = Mamba2Block(
            d_model=config.d_model,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            headdim=config.mamba_headdim,
            ngroups=config.mamba_ngroups,
            chunk_size=config.mamba_chunk_size,
            rngs=rngs,
        )

        # MoE stage after every mixer layer.
        self.moe = SparseMoE(
            d_model=config.d_model,
            num_experts=config.num_experts,
            num_shared_experts=config.num_shared_experts,
            top_k=config.top_k,
            expert_hidden_dim=config.expert_hidden_dim,
            granularity_factor=config.granularity_factor,
            scale_top_k_with_granularity=config.scale_top_k_with_granularity,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(
            self, x: jax.Array, return_aux_loss: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        # Mamba residual path.
        x = x + self.mamba(self.norm_mamba(x))

        # MoE residual path.
        if return_aux_loss:
            moe_out, moe_aux_loss = self.moe(self.norm_moe(x), return_aux_loss=True)
            x = x + moe_out
            return x, moe_aux_loss

        x = x + self.moe(self.norm_moe(x), return_aux_loss=False)
        return x


class MambaAttentionMoEBlock(nnx.Module):
    """
    Hybrid Mamba, Attention & MoE block.

    Block structure (pre-norm residual):
      x = x + Mamba(RMSNorm(x))
      x = x + Attention(RMSNorm(x))
      x = x + MoE(RMSNorm(x))

    This keeps the architecture simple and easy to inspect.
    """

    def __init__(
            self,
            rngs: nnx.Rngs,
            config: NemotronConfig
    ):
        self.norm_mamba = RMSNorm(config.d_model, eps=config.rms_norm_eps, rngs=rngs)
        self.norm_attention = RMSNorm(config.d_model, eps=config.rms_norm_eps, rngs=rngs)
        self.norm_moe = RMSNorm(config.d_model, eps=config.rms_norm_eps, rngs=rngs)

        # Reuse the already-implemented Mamba-2 block as the mixer.
        self.mamba = Mamba2Block(
            d_model=config.d_model,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
            expand=config.mamba_expand,
            headdim=config.mamba_headdim,
            ngroups=config.mamba_ngroups,
            chunk_size=config.mamba_chunk_size,
            rngs=rngs,
        )

        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.attention_head_dim,
            use_bias=False,
            rngs=rngs,
        )

        # MoE stage after every mixer layer.
        self.moe = SparseMoE(
            d_model=config.d_model,
            num_experts=config.num_experts,
            num_shared_experts=config.num_shared_experts,
            top_k=config.top_k,
            expert_hidden_dim=config.expert_hidden_dim,
            granularity_factor=config.granularity_factor,
            scale_top_k_with_granularity=config.scale_top_k_with_granularity,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(
            self, x: jax.Array, return_aux_loss: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        # Mamba residual path.
        x = x + self.mamba(self.norm_mamba(x))

        # Attention residual path.
        x = x + self.attention(self.norm_attention(x))

        # MoE residual path.
        if return_aux_loss:
            moe_out, moe_aux_loss = self.moe(self.norm_moe(x), return_aux_loss=True)
            x = x + moe_out
            return x, moe_aux_loss

        x = x + self.moe(self.norm_moe(x), return_aux_loss=False)
        return x


class NemotronNanoLM(nnx.Module):
    """
    Minimal Nemotron-style language model.

    Layout:
      token embedding -> N x NemotronBlock -> RMSNorm -> LM head

    We intentionally do not add positional embeddings to stay aligned with the
    architecture note in the paper.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronConfig):
        config.validate()
        self.config = config

        self.embedding = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)
        self.num_layers = 0

        for i in range(len(config.patterns)):
            if config.patterns[i][0] == "mamba_moe":
                for j in range(config.patterns[i][1]):
                    setattr(
                        self,
                        f"block_{self.num_layers + j}",
                        MambaMoEBlock(config=config, rngs=rngs),
                    )
            elif config.patterns[i][0] == "mamba_attention_moe":
                for j in range(config.patterns[i][1]):
                    setattr(
                        self,
                        f"block_{self.num_layers + j}",
                        MambaAttentionMoEBlock(config=config, rngs=rngs),
                    )

            self.num_layers += config.patterns[i][1]

        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps, rngs=rngs)

        # Untied output head (separate from embeddings).
        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(
            self, token_ids: jax.Array, return_aux_loss: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        x = self.embedding(token_ids)
        total_aux_loss = jnp.array(0.0, dtype=jnp.float32)

        for i in range(self.num_layers):
            block = getattr(self, f"block_{i}")
            if return_aux_loss:
                x, block_aux_loss = block(x, return_aux_loss=True)
                total_aux_loss = total_aux_loss + block_aux_loss
            else:
                x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        if return_aux_loss:
            # Average over blocks to keep scale more stable across depth.
            total_aux_loss = total_aux_loss / self.num_layers
            return logits, total_aux_loss
        return logits


# -----------------------------------------------------------------------------
# Minimal demo training utilities
# -----------------------------------------------------------------------------


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Standard language-model cross-entropy loss."""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return optax.softmax_cross_entropy(logits, one_hot).mean()


def total_training_loss(
        logits: jax.Array,
        labels: jax.Array,
        moe_aux_loss: jax.Array,
        moe_aux_loss_weight: float,
) -> jax.Array:
    """
    Combines language-model cross-entropy and MoE auxiliary loss.
    """
    ce_loss = cross_entropy_loss(logits, labels)
    return ce_loss + moe_aux_loss_weight * moe_aux_loss


def demo(preset: str = "tiny") -> None:
    """
    Tiny smoke-test demo:
    - builds the model
    - runs a few training steps
    - confirms shapes and stable execution
    """
    print("Initializing simple & minimal Nemotron 3 Nano...")

    config = NemotronConfig.from_preset(preset)
    rngs = nnx.Rngs(0)

    print(f"Using config preset: {preset}")

    model = NemotronNanoLM(rngs=rngs, config=config)
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=1e-3), wrt=nnx.Param)

    batch_size = 2
    seqlen = 64

    # Mamba-2 path requires sequence length divisible by chunk_size.
    assert seqlen % config.mamba_chunk_size == 0, (
        "For this demo, seqlen must be divisible by mamba_chunk_size"
    )

    x = jax.random.randint(rngs(), (batch_size, seqlen), 0, config.vocab_size)
    y_target = jax.random.randint(rngs(), (batch_size, seqlen), 0, config.vocab_size)

    # Check forward shape once before training.
    logits, aux_loss = model(x, return_aux_loss=True)
    assert logits.shape == (batch_size, seqlen, config.vocab_size), (
        "Unexpected logits shape"
    )
    print(f"Forward shape OK: {logits.shape}")
    print(f"Initial MoE aux loss: {aux_loss:.6f}")

    @nnx.jit
    def train_step(model, optimizer, x_batch, y_batch):
        def loss_fn(model):
            logits_local, moe_aux_local = model(x_batch, return_aux_loss=True)
            total_loss = total_training_loss(
                logits=logits_local,
                labels=y_batch,
                moe_aux_loss=moe_aux_local,
                moe_aux_loss_weight=config.moe_aux_loss_weight,
            )
            ce_loss = cross_entropy_loss(logits_local, y_batch)
            return total_loss, (ce_loss, moe_aux_local)

        (loss, (ce_loss, moe_aux_loss)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)
        optimizer.update(model, grads)
        return loss, ce_loss, moe_aux_loss

    print("Training (5 steps):")
    for step in range(5):
        loss, ce_loss, moe_aux_loss = train_step(model, optimizer, x, y_target)
        print(
            f"  Step {step + 1}/5 | Total: {loss:.4f} | "
            f"CE: {ce_loss:.4f} | MoE Aux: {moe_aux_loss:.4f}"
        )

    print("Done.")


if __name__ == "__main__":
    # Keep tiny as fallback; choose a larger profile by setting:
    #   NEMOTRON_PRESET=paper_close
    # before running this script.
    import os

    selected_preset = os.getenv("NEMOTRON_PRESET", "tiny")
    demo(preset=selected_preset)
