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
- No auxiliary load-balancing loss
"""

from dataclasses import dataclass

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
    num_layers: int = 4

    # Hybrid mixer pattern; repeated across layers.
    # Example: ("mamba", "attention") means alternating layers.
    mixer_pattern: tuple[str, ...] = ("mamba", "attention")

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

    # Normalization and numerical stability
    rms_norm_eps: float = 1e-6

    def validate(self) -> None:
        """Checks shape constraints that must hold for this architecture."""
        assert self.num_layers > 0, "num_layers must be > 0"
        assert len(self.mixer_pattern) > 0, "mixer_pattern cannot be empty"

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

        # MoE routing constraints.
        assert self.top_k > 0, "top_k must be > 0"
        assert self.top_k <= self.num_experts, "top_k must be <= num_experts"


class NemotronBlock(nnx.Module):
    """
    One hybrid Nemotron block.

    Block structure (pre-norm residual):
      x = x + Mixer(RMSNorm(x))
      x = x + MoE(RMSNorm(x))

    Mixer is either:
    - Mamba-2 block, or
    - Grouped-query attention block

    This keeps the architecture simple and easy to inspect.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        config: NemotronConfig,
        layer_index: int,
    ):
        self.layer_index = layer_index
        self.layer_type = config.mixer_pattern[layer_index % len(config.mixer_pattern)]

        self.norm_mixer = RMSNorm(config.d_model, eps=config.rms_norm_eps, rngs=rngs)
        self.norm_moe = RMSNorm(config.d_model, eps=config.rms_norm_eps, rngs=rngs)

        if self.layer_type == "mamba":
            # Reuse the already-implemented Mamba-2 block as the mixer.
            self.mixer = Mamba2Block(
                d_model=config.d_model,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                headdim=config.mamba_headdim,
                ngroups=config.mamba_ngroups,
                chunk_size=config.mamba_chunk_size,
                rngs=rngs,
            )
        elif self.layer_type == "attention":
            self.mixer = GroupedQueryAttention(
                d_model=config.d_model,
                num_query_heads=config.num_attention_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.attention_head_dim,
                use_bias=False,
                rngs=rngs,
            )
        else:
            raise ValueError(
                f"Unsupported layer type: {self.layer_type}. "
                "Expected one of {'mamba', 'attention'}."
            )

        # MoE stage after every mixer layer.
        self.moe = SparseMoE(
            d_model=config.d_model,
            num_experts=config.num_experts,
            num_shared_experts=config.num_shared_experts,
            top_k=config.top_k,
            expert_hidden_dim=config.expert_hidden_dim,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # Mixer residual path.
        x = x + self.mixer(self.norm_mixer(x))

        # MoE residual path.
        x = x + self.moe(self.norm_moe(x))
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

        self.num_layers = config.num_layers
        for i in range(config.num_layers):
            setattr(
                self,
                f"block_{i}",
                NemotronBlock(config=config, layer_index=i, rngs=rngs),
            )

        self.final_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps, rngs=rngs)

        # Untied output head (separate from embeddings).
        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        x = self.embedding(token_ids)

        for i in range(self.num_layers):
            block = getattr(self, f"block_{i}")
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------------------------------------------------------
# Minimal demo training utilities
# -----------------------------------------------------------------------------


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Standard language-model cross-entropy loss."""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return optax.softmax_cross_entropy(logits, one_hot).mean()


def demo() -> None:
    """
    Tiny smoke-test demo:
    - builds the model
    - runs a few training steps
    - confirms shapes and stable execution
    """
    print("Initializing minimal Nemotron-style model...")

    config = NemotronConfig()
    rngs = nnx.Rngs(0)

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
    logits = model(x)
    assert logits.shape == (batch_size, seqlen, config.vocab_size), (
        "Unexpected logits shape"
    )
    print(f"Forward shape OK: {logits.shape}")

    @nnx.jit
    def train_step(model, optimizer, x_batch, y_batch):
        def loss_fn(model):
            logits_local = model(x_batch)
            return cross_entropy_loss(logits_local, y_batch)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    print("Training (5 steps):")
    for step in range(5):
        loss = train_step(model, optimizer, x, y_target)
        print(f"  Step {step + 1}/5 | Loss: {loss:.4f}")

    print("Done.")


if __name__ == "__main__":
    demo()
