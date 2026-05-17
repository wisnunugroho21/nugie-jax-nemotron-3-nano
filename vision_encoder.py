"""
Vision Encoder for Nemotron 3 Nano Omni in JAX/Flax NNX.

Simplified, educational ViT-style image encoder inspired by the C-RADIOv4-H
encoder used in the paper:
  "Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence"
  arXiv:2604.24954

Architecture overview (encoder-projector design):
  image → PatchEmbedding → CPE → N × VisionTransformerBlock
        → PixelShuffle → VisionProjector → [visual tokens for LLM]

Key design choices (matching the paper):
- 16×16 patch embedding
- Conditional Positional Encoding (CPE) via 3×3 depthwise conv on the 2D
  patch grid (no positional embedding table — position is implicit in the
  conv's receptive field)
- Bidirectional (non-causal) attention — every patch attends to all others
- Pixel shuffle 4× spatial downsampling: merges 2×2 patch neighborhoods into
  the channel dim, then projects back — cuts token count by 4
- 2-layer MLP projector bridges encoder hidden dim to LLM d_model

Simplified from real C-RADIO:
- No teacher-student distillation
- No register tokens
- No RoPE positional encoding in attention
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx


# =============================================================================
# Config
# =============================================================================


@dataclass
class VisionEncoderConfig:
    """
    Configuration for the vision encoder and its MLP projector.

    Tiny defaults allow local experimentation. Scale hidden_dim and num_layers
    for paper-like quality (paper uses ViT-H scale: hidden_dim=1280, 32 layers).
    """

    image_size: int = 64    # Input image side length (square images assumed)
    patch_size: int = 16    # Each patch covers patch_size × patch_size pixels
    in_channels: int = 3    # RGB images
    hidden_dim: int = 128   # Transformer hidden width
    num_heads: int = 4      # Attention heads (hidden_dim = num_heads × head_dim)
    head_dim: int = 32      # Dimension per attention head
    num_layers: int = 2     # Number of ViT transformer blocks
    mlp_dim: int = 256      # MLP inner width inside each transformer block
    proj_dim: int = 128     # Output dimension — must equal LLM d_model


# =============================================================================
# Patch Embedding
# =============================================================================


class PatchEmbedding(nnx.Module):
    """
    Splits an image into non-overlapping 16×16 patches and linearly embeds each.

    For a 64×64 image with patch_size=16:
      → (64/16) × (64/16) = 4×4 = 16 patches
      → Each patch is flattened: 16 × 16 × 3 = 768 values
      → A linear layer maps 768 → hidden_dim

    This is the standard ViT tokenization step — converting pixel grids into
    a sequence of patch vectors the transformer can process.

    Args:
        config: VisionEncoderConfig with patch_size, in_channels, hidden_dim.
    """

    def __init__(self, config: VisionEncoderConfig, rngs: nnx.Rngs):
        patch_flat_dim = config.patch_size * config.patch_size * config.in_channels
        self.proj = nnx.Linear(
            patch_flat_dim, config.hidden_dim, use_bias=False, rngs=rngs
        )
        self.patch_size = config.patch_size

    def __call__(self, x: jax.Array) -> tuple[jax.Array, int, int]:
        """
        Args:
            x: (batch, H, W, C) float image
        Returns:
            tokens:    (batch, h*w, hidden_dim) — flat sequence of patch embeddings
            h_patches: number of patch rows (H // patch_size)
            w_patches: number of patch cols (W // patch_size)
        """
        batch, H, W, C = x.shape
        p = self.patch_size
        h_patches, w_patches = H // p, W // p

        # Reshape image into a 2D grid of patches.
        # (B, H, W, C) → (B, h, p, w, p, C)
        x = x.reshape(batch, h_patches, p, w_patches, p, C)

        # Move patch pixel dims together: (B, h, w, p, p, C)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))

        # Flatten each patch into a vector: (B, h*w, p*p*C)
        x = x.reshape(batch, h_patches * w_patches, p * p * C)

        # Linear projection: each flattened patch → hidden_dim
        tokens = self.proj(x)  # (batch, N, hidden_dim)
        return tokens, h_patches, w_patches


# =============================================================================
# Conditional Positional Encoding (CPE)
# =============================================================================


class ConditionalPositionalEncoding(nnx.Module):
    """
    Conditional Positional Encoding via a 2D depthwise convolution.

    Why CPE instead of a learned table?
    A fixed positional embedding table maps each position index to a vector.
    CPE instead runs a small 3×3 depthwise conv over the 2D spatial token grid.
    Because the conv is local (sees each token's neighborhood), the network
    implicitly learns to encode where each token is — without any explicit
    position index. This naturally generalises to different image sizes.

    Implementation:
      1. Reshape flat sequence → 2D spatial grid: (B, N, D) → (B, H, W, D)
      2. 3×3 depthwise conv: each channel kernel sees its local neighborhood
      3. Add result as residual (preserves the original patch content)
      4. Reshape back to (B, N, D)

    Args:
        hidden_dim: Token channel width D.
    """

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        # Depthwise: feature_group_count=D means in_features/group=1, so each
        # of the D channels has its own independent 3×3 kernel.
        self.dw_conv = nnx.Conv(
            in_features=hidden_dim,
            out_features=hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),   # same padding — spatial size unchanged
            feature_group_count=hidden_dim,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, tokens: jax.Array, h: int, w: int) -> jax.Array:
        """
        Args:
            tokens: (batch, h*w, D)
            h, w:   spatial grid dimensions
        Returns:
            tokens with positional context added, same shape (batch, h*w, D)
        """
        batch, N, D = tokens.shape

        # Restore 2D spatial layout for the conv.
        grid = tokens.reshape(batch, h, w, D)

        # Depthwise conv over the spatial grid — encodes local position context.
        pos_delta = self.dw_conv(grid)  # (batch, h, w, D)

        # Residual addition: merge position hint with original patch content.
        grid = grid + pos_delta

        # Flatten back to sequence.
        return grid.reshape(batch, N, D)


# =============================================================================
# Vision Multi-Head Attention (bidirectional, no causal mask)
# =============================================================================


class VisionAttention(nnx.Module):
    """
    Standard multi-head self-attention for vision.

    Unlike text decoder attention, image patches have no causal ordering —
    patch 7 can freely attend to patch 2 or patch 15. So there is no
    lower-triangular causal mask here.

    Args:
        config: VisionEncoderConfig
    """

    def __init__(self, config: VisionEncoderConfig, rngs: nnx.Rngs):
        D = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.q_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(D, D, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (batch, N, hidden_dim) — N patches
        Returns:
            (batch, N, hidden_dim)
        """
        batch, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Project to Q, K, V and expose head dimension.
        q = jnp.transpose(self.q_proj(x).reshape(batch, N, H, d), (0, 2, 1, 3))
        k = jnp.transpose(self.k_proj(x).reshape(batch, N, H, d), (0, 2, 1, 3))
        v = jnp.transpose(self.v_proj(x).reshape(batch, N, H, d), (0, 2, 1, 3))
        # Each: (batch, H, N, d)

        # Scaled dot-product attention — no causal mask for image patches.
        scale = 1.0 / math.sqrt(d)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale  # (B, H, N, N)
        attn = jax.nn.softmax(scores, axis=-1)

        # Weighted sum over values.
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v)  # (B, H, N, d)

        # Merge heads and project back to D.
        context = jnp.transpose(context, (0, 2, 1, 3)).reshape(batch, N, D)
        return self.out_proj(context)


# =============================================================================
# Vision Transformer Block
# =============================================================================


class VisionTransformerBlock(nnx.Module):
    """
    One ViT transformer block using pre-norm residuals.

    Structure:
      x = x + Attention(RMSNorm(x))    ← multi-head self-attention
      x = x + MLP(RMSNorm(x))          ← feed-forward network

    Pre-norm (normalize before the sublayer) gives more stable gradients
    than the original post-norm design. This matches the LLM backbone style.

    Args:
        config: VisionEncoderConfig
    """

    def __init__(self, config: VisionEncoderConfig, rngs: nnx.Rngs):
        D, M = config.hidden_dim, config.mlp_dim
        self.norm_attn = nnx.RMSNorm(D, rngs=rngs)
        self.norm_mlp = nnx.RMSNorm(D, rngs=rngs)
        self.attn = VisionAttention(config, rngs=rngs)

        # 2-layer MLP: expand to mlp_dim with SiLU, then contract back.
        self.mlp_fc1 = nnx.Linear(D, M, use_bias=False, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(M, D, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Attention path.
        x = x + self.attn(self.norm_attn(x))

        # MLP path: Linear → SiLU → Linear.
        h = self.mlp_fc2(jax.nn.silu(self.mlp_fc1(self.norm_mlp(x))))
        x = x + h

        return x


# =============================================================================
# Pixel Shuffle (4× spatial downsampling)
# =============================================================================


class PixelShuffle(nnx.Module):
    """
    Reduces spatial token count by 4× by merging 2×2 patch neighborhoods.

    Pixel shuffle (space-to-depth): fold a 2×2 block of spatial tokens into
    the channel dimension, doubling the channel width twice (×4 total).
    A linear layer then projects back to the original hidden_dim.

    Why? The paper applies "pixel shuffle with 4× downsampling" after the
    visual encoder to reduce the number of visual tokens fed to the LLM —
    keeping the sequence length manageable while preserving information.

    Process:
      (B, h×w, D)
        → reshape to spatial grid  (B, h, w, D)
        → fold 2×2 blocks          (B, h//2, w//2, 4D)
        → linear project           (B, h//2, w//2, D)
        → flatten                  (B, (h//2)*(w//2), D)

    Constraint: h and w must be even (patch grid must be divisible by 2).

    Args:
        hidden_dim: Token channel width D.
    """

    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        # Projects merged 4D back down to D.
        self.proj = nnx.Linear(4 * hidden_dim, hidden_dim, use_bias=False, rngs=rngs)

    def __call__(self, tokens: jax.Array, h: int, w: int) -> jax.Array:
        """
        Args:
            tokens: (batch, h*w, D)
            h, w:   spatial grid size — both must be even
        Returns:
            downsampled: (batch, (h//2)*(w//2), D)
        """
        batch, _, D = tokens.shape

        # Step 1: restore the 2D spatial layout.
        grid = tokens.reshape(batch, h, w, D)

        # Step 2: fold 2×2 spatial blocks into the channel dimension.
        # (B, h, w, D) → (B, h//2, 2, w//2, 2, D)
        grid = grid.reshape(batch, h // 2, 2, w // 2, 2, D)
        # → (B, h//2, w//2, 2, 2, D)
        grid = jnp.transpose(grid, (0, 1, 3, 2, 4, 5))
        # → (B, h//2, w//2, 4D)
        grid = grid.reshape(batch, h // 2, w // 2, 4 * D)

        # Step 3: project merged channels back down to D.
        grid = self.proj(grid)  # (B, h//2, w//2, D)

        # Step 4: flatten spatial dims back to a flat sequence.
        return grid.reshape(batch, (h // 2) * (w // 2), D)


# =============================================================================
# Vision Encoder
# =============================================================================


class VisionEncoder(nnx.Module):
    """
    Full ViT-style image encoder.

    Pipeline:
      image → PatchEmbedding → CPE → [VisionTransformerBlock × N] → PixelShuffle

    The output tokens are still in encoder hidden_dim space.
    VisionProjector (below) maps them to the LLM's d_model dimension.

    Args:
        config: VisionEncoderConfig
    """

    def __init__(self, config: VisionEncoderConfig, rngs: nnx.Rngs):
        self.config = config
        self.patch_embed = PatchEmbedding(config, rngs=rngs)
        self.cpe = ConditionalPositionalEncoding(config.hidden_dim, rngs=rngs)
        self.blocks = nnx.List(
            [VisionTransformerBlock(config, rngs=rngs) for _ in range(config.num_layers)]
        )
        self.pixel_shuffle = PixelShuffle(config.hidden_dim, rngs=rngs)
        self.norm = nnx.RMSNorm(config.hidden_dim, rngs=rngs)

    def __call__(self, images: jax.Array) -> jax.Array:
        """
        Args:
            images: (batch, H, W, C) float image
        Returns:
            tokens: (batch, N_out, hidden_dim)
                    N_out = ((H // patch_size) // 2) * ((W // patch_size) // 2)
                    e.g. 64×64 image, patch 16 → 4×4 patches → 2×2 after shuffle = 4 tokens
        """
        # 1) Convert image pixels to patch token sequence.
        tokens, h, w = self.patch_embed(images)  # (B, h*w, D)

        # 2) Inject positional information via depthwise conv on the spatial grid.
        tokens = self.cpe(tokens, h, w)           # (B, h*w, D)

        # 3) Stack of ViT transformer blocks (bidirectional attention + MLP).
        for block in self.blocks:
            tokens = block(tokens)

        # 4) Normalize before downsampling (stabilises the pixel shuffle input).
        tokens = self.norm(tokens)

        # 5) Pixel shuffle: merge 2×2 neighborhoods → 4× fewer tokens.
        tokens = self.pixel_shuffle(tokens, h, w)  # (B, N_out, D)

        return tokens


# =============================================================================
# Vision Projector
# =============================================================================


class VisionProjector(nnx.Module):
    """
    2-layer MLP that adapts visual encoder tokens to the LLM's hidden dimension.

    This is the "adapter" or "connector" between the vision encoder and the
    language model backbone. A simple 2-layer MLP with SiLU activation is
    what the paper uses (matching standard VLM practice).

    In the training curriculum, this projector is trained first (stage 0)
    with everything else frozen, so the LLM can initially "see" visual tokens
    without disrupting its text capabilities.

    Args:
        in_dim:  Vision encoder output dimension.
        out_dim: LLM hidden dimension (d_model).
    """

    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs):
        hidden = (in_dim + out_dim) // 2  # smooth interpolation between dims
        self.fc1 = nnx.Linear(in_dim, hidden, use_bias=False, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (batch, N, in_dim)
        Returns:
            (batch, N, out_dim)
        """
        return self.fc2(jax.nn.silu(self.fc1(x)))
