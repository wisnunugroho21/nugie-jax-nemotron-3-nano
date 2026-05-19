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
- No positional embeddings, no dropout, and bias-free linear layers

What is intentionally simplified:
- Tiny default dimensions for local experimentation
- Alternating layer pattern instead of large paper-scale block scheduling
- No distributed/expert-parallel optimization

Large Reasoning Model (LRM) note:
Converting this model into an LRM requires no architectural changes.
The approach is Supervised Fine-Tuning (SFT) on Chain-of-Thought (CoT) data:
  1. Use a dataset with explicit reasoning traces, e.g. open-thoughts/OpenThoughts-114k.
  2. Add <think> and </think> as special tokens to the tokenizer.
  3. Train with the standard next-token loss, supervising on the FULL assistant
     turn — including the thinking trace inside <think>...</think>.
     (Loss mask: user/system tokens = 0.0, assistant tokens = 1.0.)
  4. At inference, the model generates <think>...</think> before the answer.
The Mamba + Attention + MoE hybrid is well-suited for this because:
  - Mamba handles the long reasoning traces efficiently (linear-time SSM).
  - Attention allows cross-reference back to earlier reasoning steps.
  - MoE experts specialize in different reasoning domains over time.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from attention import GroupedQueryAttention
from latent_moe import LatentMoE
from mamba_2 import Mamba2Block
from moe import SparseMoE
from multi_token_prediction import MultiTokenPrediction, mtp_loss

# =============================================================================
# Config
# =============================================================================


def _default_patterns() -> list[tuple[str, int]]:
    return [
        ("mamba_moe", 2),
        ("mamba_attention_moe", 1),
        ("mamba_moe", 2),
        ("mamba_attention_moe", 1),
        ("mamba_moe", 2),
        ("mamba_attention_moe", 1),
        ("mamba_moe", 1),
    ]


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

    patterns: list[tuple[str, int]] = field(default_factory=_default_patterns)

    # Attention (GQA)
    num_attention_heads: int = 4
    num_kv_heads: int = 1
    attention_head_dim: int = 32

    # Mamba-2 settings
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
                patterns=_default_patterns(),
            )
        
        if key in ("kaggle", "colab"):
            return cls(
                # Bigger model than tiny defaults.
                patterns = [
                    ("mamba_moe",         2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe",         2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe",         2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe",         2),
                    ("mamba_attention_moe", 1),
                    ("mamba_moe",         2),
                ],
                d_model              = 256,
                # Attention: 4 heads × 64 = 256 = d_model ✓
                num_attention_heads  = 4,
                attention_head_dim   = 64,
                num_kv_heads         = 2,
                # Mamba: d_inner = 2 × 256 = 512, nheads = 8, ngroups = 2 — all divisible ✓
                mamba_d_state        = 128,
                mamba_expand         = 2,
                mamba_headdim        = 64,
                mamba_ngroups        = 2,
                mamba_chunk_size     = 64,
                # MoE: more experts, wider hidden dim
                num_experts          = 8,
                num_shared_experts   = 1,
                top_k                = 2,
                expert_hidden_dim    = 512,
                granularity_factor   = 1,
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
                # Keep exactly 6 activated routed experts (paper behavior).
                scale_top_k_with_granularity=False,
                rms_norm_eps=1e-6,
            )

        raise ValueError(
            f"Unknown preset '{preset}'. Supported presets: tiny, kaggle, colab, paper_close"
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


# =============================================================================
# Helper
# =============================================================================


def _build_mamba(config: NemotronConfig, rngs: nnx.Rngs) -> Mamba2Block:
    return Mamba2Block(
        d_model=config.d_model,
        d_state=config.mamba_d_state,
        d_conv=config.mamba_d_conv,
        expand=config.mamba_expand,
        headdim=config.mamba_headdim,
        ngroups=config.mamba_ngroups,
        chunk_size=config.mamba_chunk_size,
        rngs=rngs,
    )


def _build_moe(config: NemotronConfig, rngs: nnx.Rngs) -> SparseMoE:
    return SparseMoE(
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


# =============================================================================
# Mamba MoE Block
# =============================================================================


class MambaMoEBlock(nnx.Module):
    """
    Hybrid Mamba & MoE block.

    Block structure (pre-norm residual):
      x = x + Mamba(RMSNorm(x))
      x = x + MoE(RMSNorm(x))

    This keeps the architecture simple and easy to inspect.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronConfig):
        self.norm_mamba = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.norm_moe = nnx.RMSNorm(config.d_model, rngs=rngs)

        # Reuse the already-implemented Mamba-2 block
        self.mamba = _build_mamba(config=config, rngs=rngs)

        # MoE stage after every mixer layer.
        self.moe = _build_moe(config=config, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # LRM note: Mamba processes sequences in linear time (O(n)), which is
        # important for LRM training because reasoning traces can be thousands
        # of tokens long. This block is used for the majority of layers in the
        # hybrid stack, giving the model efficient long-context processing.

        # Mamba residual path.
        x = x + self.mamba(self.norm_mamba(x))

        # LRM note: The MoE layer after every mixer lets different experts
        # specialize. Over training, some experts activate more on mathematical
        # derivations, others on code, others on natural language reasoning.
        # The shared experts handle universal patterns like step-by-step breakdowns.

        # MoE residual path.
        return x + self.moe(self.norm_moe(x))


# =============================================================================
# Mamba Attention MoE Block
# =============================================================================


class MambaAttentionMoEBlock(nnx.Module):
    """
    Hybrid Mamba, Attention & MoE block.

    Block structure (pre-norm residual):
      x = x + Mamba(RMSNorm(x))
      x = x + Attention(RMSNorm(x))
      x = x + MoE(RMSNorm(x))

    This keeps the architecture simple and easy to inspect.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronConfig):
        self.norm_mamba = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.norm_attention = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.norm_moe = nnx.RMSNorm(config.d_model, rngs=rngs)

        # Reuse the already-implemented Mamba-2 block as the mixer.
        self.mamba = _build_mamba(config=config, rngs=rngs)

        self.attention = GroupedQueryAttention(
            d_model=config.d_model,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.attention_head_dim,
            rngs=rngs,
        )

        # MoE stage after every mixer layer.
        self.moe = _build_moe(config=config, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # LRM note: Mamba handles the bulk of long-context sequence processing
        # (linear time). It runs first so subsequent layers see a rich summary.

        # Mamba residual path.
        x = x + self.mamba(self.norm_mamba(x))

        # LRM note: Attention is placed in select layers (not every layer) so
        # the model can look back at specific earlier positions in the reasoning
        # trace — e.g., recalling a formula defined two paragraphs ago or
        # checking an intermediate result. This targeted retrieval ability is
        # important for multi-step reasoning that references its own prior work.

        # Attention residual path.
        x = x + self.attention(self.norm_attention(x))

        # MoE residual path.
        return x + self.moe(self.norm_moe(x))


# =============================================================================
# Nemotron 3 Nano Block
# =============================================================================


class NemotronNanoBlock(nnx.Module):
    """
    Minimal Nemotron-style language model.

    Layout:
      token embedding -> N x NemotronBlock -> RMSNorm -> LM head

            No positional embeddings are used.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronConfig):
        config.validate()
        self.config = config

        self.embedding = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)
        block_factories = {
            "mamba_moe": MambaMoEBlock,
            "mamba_attention_moe": MambaAttentionMoEBlock,
        }

        blocks: list[MambaMoEBlock | MambaAttentionMoEBlock] = []
        for block_type, repeats in config.patterns:
            block_factory = block_factories.get(block_type)
            if block_factory is None:
                raise ValueError(f"Unknown block type '{block_type}'")
            for _ in range(repeats):
                blocks.append(block_factory(config=config, rngs=rngs))
        self.blocks = nnx.List(blocks)

        self.final_norm = nnx.RMSNorm(config.d_model, rngs=rngs)

        # Untied output head (separate from embeddings).
        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        # LRM note: During LRM training the token_ids include <think> and </think>
        # as real tokens in the vocabulary (added via setup_reasoning_tokenizer).
        # The model sees the full thinking trace as part of the supervised target.
        # It learns to produce: <|assistant|> <think> {reasoning} </think> {answer}
        # The loss is computed on ALL assistant tokens including the thinking part.
        x = self.embedding(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits


# =============================================================================
# Nemotron 3 Super — Config
# =============================================================================


def _default_super_patterns() -> list[tuple[str, int]]:
    # Mirrors the Nano default pattern but uses latent-MoE block types.
    return [
        ("latent_mamba_moe", 2),
        ("latent_mamba_attention_moe", 1),
        ("latent_mamba_moe", 2),
        ("latent_mamba_attention_moe", 1),
        ("latent_mamba_moe", 2),
        ("latent_mamba_attention_moe", 1),
        ("latent_mamba_moe", 1),
    ]


@dataclass
class NemotronSuperConfig:
    """
    Configuration for Nemotron 3 Super.

    Extends NemotronConfig with two new architectural components:

    1. LatentMoE (arXiv:2601.18089):
       Replaces SparseMoE.  Routed experts operate in a compressed latent space
       of size `latent_size` (ℓ) instead of the full model dimension d_model.
       - `latent_size`: compressed dimension ℓ; compression ratio α = d_model / ℓ.
       - `num_experts` and `top_k` should already be α-scaled relative to Nano
         (i.e. multiply Nano values by α before setting them here).
       - `shared_expert_hidden_dim`: intermediate size for always-on experts.
         Typically 2 × expert_hidden_dim because shared experts process every
         token and therefore benefit from more capacity.
       - `granularity_factor` is NOT present — latent projection is the
         efficiency mechanism in Super; granularity is a Nano-only concept.

    2. Multi-Token Prediction (arXiv:2604.12374 §2.1.2 and §2.4):
       Jointly trains the model to predict `num_mtp_heads` tokens further into
       the future in addition to the standard next-token objective.
       - `num_mtp_heads`: number of extra prediction depths (2 for Super).
       - `mtp_loss_scale`: weight applied to the MTP auxiliary loss (0.3).
       Training batches must be `num_mtp_heads + 1` tokens longer than the
       main model sequence length so the extra future tokens are available.

        3. Load balancing in MoE routing:
             Nemotron 3 Super uses both bias-based (aux-loss-free) balancing and a
             standard load-balancing loss term during training.
             - `load_balancing_loss_coef`: coefficient for the standard MoE
                 load-balancing loss (1e-4 in the paper).

    Nano code (NemotronConfig, NemotronNanoBlock) is completely untouched.

    Paper-faithful values for Nemotron 3 Super (Table 1):
      d_model                  = 4096
      num_attention_heads      = 32
      num_kv_heads             = 2
      attention_head_dim       = 128
      mamba_d_state            = 128
      mamba_ngroups            = 8
      latent_size              = 1024   (α = 4)
      num_experts              = 512
      top_k                    = 22
      num_shared_experts       = 2
      expert_hidden_dim        = 2688
      shared_expert_hidden_dim = 5376
      num_mtp_heads            = 2
      mtp_loss_scale           = 0.3
    load_balancing_loss_coef = 1e-4
    """

    # Token / model sizes
    vocab_size: int = 1000
    d_model: int = 128

    # Layer pattern — use "latent_mamba_moe" and "latent_mamba_attention_moe"
    patterns: list[tuple[str, int]] = field(
        default_factory=_default_super_patterns
    )

    # Attention (GQA)
    num_attention_heads: int = 4
    num_kv_heads: int = 1
    attention_head_dim: int = 32

    # Mamba-2 settings
    mamba_d_state: int = 64
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 64

    # LatentMoE settings
    latent_size: int = 32                  # compressed dimension ℓ (must be < d_model)
    num_experts: int = 8                   # total routed experts (α-scaled)
    num_shared_experts: int = 1            # always-on experts (in full d-space)
    top_k: int = 2                         # active routed experts per token (α-scaled)
    expert_hidden_dim: int = 256           # FFN intermediate dim for routed experts
    shared_expert_hidden_dim: int = 512    # FFN intermediate dim for shared experts

    # Multi-Token Prediction settings
    num_mtp_heads: int = 2                 # extra prediction depths (2 for Super)
    mtp_loss_scale: float = 0.3            # auxiliary MTP loss weight

    # MoE load-balancing loss (used alongside bias-based balancing)
    load_balancing_loss_coef: float = 1e-4

    # Normalization
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_preset(cls, preset: str = "tiny_super") -> "NemotronSuperConfig":
        """
        Build a config from a named preset.

        Presets:
        - tiny_super:   local-friendly, fast to iterate on (default).
        - kaggle_super: larger, suitable for Kaggle/Colab T4/P100 runs.
        - paper_super:  paper-faithful Nemotron 3 Super dimensions (Table 1).
        """
        key = preset.strip().lower()

        if key in ("tiny_super", "tiny", "default"):
            return cls()  # tiny defaults defined in the dataclass fields above

        if key in ("kaggle_super", "kaggle", "colab_super", "colab"):
            return cls(
                patterns=[
                    ("latent_mamba_moe",           2),
                    ("latent_mamba_attention_moe",  1),
                    ("latent_mamba_moe",           2),
                    ("latent_mamba_attention_moe",  1),
                    ("latent_mamba_moe",           2),
                    ("latent_mamba_attention_moe",  1),
                    ("latent_mamba_moe",           2),
                    ("latent_mamba_attention_moe",  1),
                    ("latent_mamba_moe",           2),
                ],
                d_model=256,
                # Attention: 4 heads × 64 = 256 = d_model ✓
                num_attention_heads=4,
                num_kv_heads=2,
                attention_head_dim=64,
                # Mamba: d_inner = 512, nheads = 8 ✓
                mamba_d_state=128,
                mamba_expand=2,
                mamba_headdim=64,
                mamba_ngroups=2,
                mamba_chunk_size=64,
                # LatentMoE: α = 256/64 = 4; scale up from a Nano base
                latent_size=64,
                num_experts=32,
                num_shared_experts=1,
                top_k=4,
                expert_hidden_dim=256,
                shared_expert_hidden_dim=512,
                num_mtp_heads=2,
                mtp_loss_scale=0.3,
            )

        if key in ("paper_super", "paper-super", "paper"):
            # Exact Nemotron 3 Super values from Table 1 of arXiv:2604.12374.
            return cls(
                # 88-layer hybrid: repeating [mamba_moe×2, mamba_attention_moe×1]
                # for 29 groups = 87 layers, plus one trailing mamba_moe = 88 total.
                patterns=[
                    ("latent_mamba_moe",           2),
                    ("latent_mamba_attention_moe",  1),
                ] * 29 + [
                    ("latent_mamba_moe",           1),
                ],
                d_model=4096,
                # Attention (GQA): 32 Q-heads, 2 KV-heads, head_dim=128
                num_attention_heads=32,
                num_kv_heads=2,
                attention_head_dim=128,
                # Mamba: d_inner = 2×4096 = 8192, nheads = 128, ngroups = 8 ✓
                mamba_d_state=128,
                mamba_d_conv=4,
                mamba_expand=2,
                mamba_headdim=64,
                mamba_ngroups=8,
                mamba_chunk_size=64,
                # LatentMoE: α = 4096/1024 = 4
                latent_size=1024,
                num_experts=512,
                num_shared_experts=2,
                top_k=22,
                expert_hidden_dim=2688,
                shared_expert_hidden_dim=5376,
                # MTP
                num_mtp_heads=2,
                mtp_loss_scale=0.3,
                load_balancing_loss_coef=1e-4,
                rms_norm_eps=1e-6,
            )

        raise ValueError(
            f"Unknown preset '{preset}'. "
            "Supported presets: tiny_super, kaggle_super, paper_super"
        )

    def validate(self) -> None:
        """Check all shape constraints required by this architecture."""
        assert len(self.patterns) > 0, "patterns cannot be empty"

        # Every pattern entry must be one of the two Super block types.
        valid_block_types = {"latent_mamba_moe", "latent_mamba_attention_moe"}
        for block_type, repeats in self.patterns:
            assert block_type in valid_block_types, (
                f"Unknown block type '{block_type}'. "
                f"Valid types for NemotronSuperConfig: {valid_block_types}"
            )
            assert repeats > 0, f"Repeat count for '{block_type}' must be > 0"

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

        # LatentMoE shape constraints.
        assert self.latent_size < self.d_model, (
            "latent_size must be strictly less than d_model (it is a compression)"
        )
        assert self.top_k > 0, "top_k must be > 0"
        assert self.top_k <= self.num_experts, "top_k must be <= num_experts"

        # MTP constraints.
        assert self.num_mtp_heads >= 0, "num_mtp_heads must be >= 0"
        assert self.mtp_loss_scale > 0, "mtp_loss_scale must be > 0"

        # MoE load-balancing coefficient.
        assert self.load_balancing_loss_coef >= 0, (
            "load_balancing_loss_coef must be >= 0"
        )


# =============================================================================
# Nemotron 3 Super — Helper functions
# =============================================================================


def _build_latent_moe(config: NemotronSuperConfig, rngs: nnx.Rngs) -> LatentMoE:
    """Build a LatentMoE layer from a NemotronSuperConfig."""
    return LatentMoE(
        rngs=rngs,
        d_model=config.d_model,
        latent_size=config.latent_size,
        num_experts=config.num_experts,
        num_shared_experts=config.num_shared_experts,
        top_k=config.top_k,
        expert_hidden_dim=config.expert_hidden_dim,
        shared_expert_hidden_dim=config.shared_expert_hidden_dim,
    )


def _build_mtp(config: NemotronSuperConfig, rngs: nnx.Rngs) -> MultiTokenPrediction:
    """Build a MultiTokenPrediction module from a NemotronSuperConfig."""
    return MultiTokenPrediction(
        rngs=rngs,
        d_model=config.d_model,
        vocab_size=config.vocab_size,
        rms_norm_eps=config.rms_norm_eps,
        num_heads=config.num_mtp_heads,
        mamba_d_state=config.mamba_d_state,
        mamba_d_conv=config.mamba_d_conv,
        mamba_expand=config.mamba_expand,
        mamba_headdim=config.mamba_headdim,
        mamba_ngroups=config.mamba_ngroups,
        mamba_chunk_size=config.mamba_chunk_size,
    )


# =============================================================================
# Nemotron 3 Super — Blocks
# =============================================================================


class LatentMambaMoEBlock(nnx.Module):
    """
    Hybrid Mamba + LatentMoE block for Nemotron 3 Super.

    Drop-in replacement for MambaMoEBlock that uses LatentMoE instead of
    SparseMoE. The residual structure is identical:

        x = x + Mamba(RMSNorm(x))
        x = x + LatentMoE(RMSNorm(x))

    This block is used for the majority of layers in the Super stack —
    the Mamba SSM handles long-range sequential context in linear time,
    while LatentMoE provides expert-specialized feed-forward computation.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronSuperConfig):
        self.norm_mamba = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.norm_moe = nnx.RMSNorm(config.d_model, rngs=rngs)

        self.mamba = _build_mamba(config=config, rngs=rngs)
        self.moe = _build_latent_moe(config=config, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Mamba residual: linear-time sequence mixing.
        x = x + self.mamba(self.norm_mamba(x))
        # LatentMoE residual: expert-specialized token-wise projection.
        return x + self.moe(self.norm_moe(x))


class LatentMambaAttentionMoEBlock(nnx.Module):
    """
    Hybrid Mamba + Attention + LatentMoE block for Nemotron 3 Super.

    Drop-in replacement for MambaAttentionMoEBlock that uses LatentMoE.
    Structure:

        x = x + Mamba(RMSNorm(x))
        x = x + Attention(RMSNorm(x))
        x = x + LatentMoE(RMSNorm(x))

    These "global attention" layers appear less frequently in the stack
    (roughly 1 in every 3 blocks by default). Adding full attention lets
    the model reference arbitrary earlier positions in a reasoning trace —
    e.g. looking up a formula stated several paragraphs earlier — which
    pure Mamba layers cannot do in a single pass.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronSuperConfig):
        self.norm_mamba = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.norm_attention = nnx.RMSNorm(config.d_model, rngs=rngs)
        self.norm_moe = nnx.RMSNorm(config.d_model, rngs=rngs)

        self.mamba = _build_mamba(config=config, rngs=rngs)
        self.attention = GroupedQueryAttention(
            rngs=rngs,
            d_model=config.d_model,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.attention_head_dim,
        )
        self.moe = _build_latent_moe(config=config, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Mamba runs first: gives attention a richer, already-contextualized
        # representation to query over.
        x = x + self.mamba(self.norm_mamba(x))
        # Attention: targeted lookup into arbitrary earlier positions.
        x = x + self.attention(self.norm_attention(x))
        # LatentMoE: expert-specialized token-wise feed-forward.
        return x + self.moe(self.norm_moe(x))


# =============================================================================
# Nemotron 3 Super — Full Model
# =============================================================================


class NemotronSuperBlock(nnx.Module):
    """
    Nemotron 3 Super language model.

    Extends the Nano architecture with two components:
      1. LatentMoE: all MoE layers use the latent-space routing scheme.
      2. MultiTokenPrediction: an auxiliary head that jointly trains the model
         to predict multiple future tokens (2 by default).

    Layout:
        token embedding
        → N × (LatentMambaMoEBlock | LatentMambaAttentionMoEBlock)
        → RMSNorm → LM head                    ← standard next-token output
        → MultiTokenPrediction (MTP)            ← auxiliary multi-step output

    Inference: call __call__(token_ids) — returns main logits only.
    Training:  call forward_train(token_ids_extended) — returns
               (main_logits, main_labels, mtp_outputs) for loss computation.

    Training batch shape: (batch, T + num_mtp_heads + 1)
      - positions 0..T-1              → main model inputs
      - positions 1..T                → main model labels
      - positions 2..T+num_mtp_heads  → MTP teacher forcing and labels
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronSuperConfig):
        config.validate()
        self.config = config

        self.embedding = nnx.Embed(config.vocab_size, config.d_model, rngs=rngs)

        block_factories = {
            "latent_mamba_moe": LatentMambaMoEBlock,
            "latent_mamba_attention_moe": LatentMambaAttentionMoEBlock,
        }

        blocks: list[LatentMambaMoEBlock | LatentMambaAttentionMoEBlock] = []
        for block_type, repeats in config.patterns:
            block_factory = block_factories.get(block_type)
            if block_factory is None:
                raise ValueError(f"Unknown block type '{block_type}'")
            for _ in range(repeats):
                blocks.append(block_factory(config=config, rngs=rngs))
        self.blocks = nnx.List(blocks)

        self.final_norm = nnx.RMSNorm(config.d_model, rngs=rngs)

        # Untied LM head — separate weights from the token embedding table.
        self.lm_head = nnx.Linear(
            config.d_model, config.vocab_size, use_bias=False, rngs=rngs
        )

        # Multi-Token Prediction module (attached at the top of the main stack).
        # Only used during training (via forward_train) — __call__ skips it.
        self.mtp = _build_mtp(config=config, rngs=rngs)

    def __call__(self, token_ids: jax.Array) -> jax.Array:
        """
        Standard forward pass for inference — returns main logits only.

        Args:
            token_ids: (batch, seq_len)  — integer token IDs.
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        return self.lm_head(x)

    def forward_train(
        self, token_ids_extended: jax.Array
    ) -> tuple[jax.Array, jax.Array, list[tuple[jax.Array, jax.Array]]]:
        """
        Training forward pass — returns main logits, main labels, and MTP outputs.

        The extended batch must include extra future tokens for MTP teacher forcing
        (see NemotronSuperBlock docstring for the required batch layout).

        Args:
            token_ids_extended: (batch, T + num_mtp_heads + 1)
                Extended token ID sequence. The first T tokens are the main model
                input; the trailing tokens provide MTP teacher forcing and labels.

        Returns:
            main_logits: (batch, T, vocab_size)
            main_labels: (batch, T)             — integer IDs, aligned to main_logits.
            mtp_outputs: list of (logits, labels) tuples, one per MTP depth.
                logits: (batch, T, vocab_size)
                labels: (batch, T)

        Training loss example:
            main_logits, main_labels, mtp_outputs = model.forward_train(batch)
            main_loss = optax.softmax_cross_entropy_with_integer_labels(
                main_logits, main_labels
            ).mean()
            total_loss = (
                main_loss
                + mtp_loss(mtp_outputs, scale=config.mtp_loss_scale)
                + config.load_balancing_loss_coef * model.load_balancing_loss()
            )
        """
        T = token_ids_extended.shape[1] - self.config.num_mtp_heads - 1

        # Main model processes the first T tokens.
        inputs = token_ids_extended[:, :T]      # (batch, T)
        main_labels = token_ids_extended[:, 1:T + 1]  # (batch, T)

        # Run main model — capture hidden states BEFORE final_norm.
        # The MTP head applies its own h_norm, so passing pre-norm hidden states
        # gives it a richer, unnormalized signal to work with.
        x = self.embedding(inputs)
        for block in self.blocks:
            x = block(x)
        main_hidden = x  # (batch, T, d_model) — before final_norm

        # Project main hidden states to logits via final_norm + lm_head.
        main_logits = self.lm_head(self.final_norm(main_hidden))

        # Extended token IDs for MTP teacher forcing and label extraction.
        # extended: (batch, T + num_mtp_heads)
        # Index k maps to token at position k+1 relative to main model start.
        extended = token_ids_extended[:, 1:]

        # Run shared-weight MTP heads — same head module called num_heads times.
        mtp_outputs = self.mtp(main_hidden, extended, self.embedding)

        return main_logits, main_labels, mtp_outputs

    def load_balancing_loss(self) -> jax.Array:
        """Return the mean standard load-balancing loss across all LatentMoE layers."""
        losses = [
            moe.last_load_balance_loss.get_value()
            for moe in self.collect_moe_layers()
        ]
        if not losses:
            return jnp.array(0.0, dtype=jnp.float32)
        return jnp.mean(jnp.stack(losses))

    def collect_moe_layers(self) -> list[LatentMoE]:
        """
        Return all LatentMoE sub-modules in order of their depth in the stack.

        Use this in the training loop to call update_expert_bias() after each
        optimizer step (outside the gradient tape), exactly as with Nano:

            moe_layers = model.collect_moe_layers()
            # ... after optimizer.update(model, grads):
            for moe in moe_layers:
                moe.update_expert_bias(moe.last_topk_indices.get_value())
        """
        return [block.moe for block in self.blocks]
