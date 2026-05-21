"""
C-RADIOv4-H Vision Encoder for Nemotron 3 Nano Omni.

This implements the vision encoder used in:
  "Nemotron 3 Nano Omni: Efficient and Open Multimodal Intelligence"
  https://arxiv.org/abs/2604.24954

The vision encoder is C-RADIOv4-H (nvidia/C-RADIOv4-H):
  • An agglomerative vision backbone distilled from SigLIP2, DINOv3, and SAM3.
  • Based on ViT-H/16 (patch size 16, embed dim 1280, depth 32, 16 heads).
  • Described in the C-RADIOv4 tech report: https://arxiv.org/abs/2601.17237

After encoding, a pixel-shuffle 4× downsampling step reduces the number of
spatial tokens by 4× before they are projected into the LLM hidden dimension.

Architecture overview:
    Image (B, H, W, 3)
        ↓  split into 16×16 patches → flatten each patch
        ↓  linear patch embedding → (B, T, embed_dim)   where T = H/16 × W/16
        ↓  + CPE positional encoding (bilinear-interpolated to current resolution)
        ↓  prepend CLS token → (B, 1+T, embed_dim)
        ↓  32 × VisionBlock [pre-LN → MHSA → residual; pre-LN → MLP → residual]
        ↓  final LayerNorm
        ↓  split outputs:
    summary:  (B, embed_dim)             CLS token — global image representation
    spatial:  (B, T, embed_dim)          patch tokens — spatial features per patch
        ↓  pixel_shuffle_down(scale=2) — merge 2×2 patch blocks into one token
    spatial:  (B, T/4, embed_dim×4)     downsampled, 4× wider channel dim
        ↓  LLM projection (in NemotronMultimodal)

Key design choices:
  • Bidirectional attention: every patch can see every other patch (no causal mask).
  • CPE (Conditional Positional Encoding): positional embeddings are stored at a
    fixed maximum resolution and bilinearly interpolated at runtime, enabling the
    encoder to process any multiple-of-16 image resolution without retraining.
  • LayerNorm (not RMSNorm): standard ViT uses LayerNorm in the backbone.
  • Biases enabled in attention and MLP: standard ViT-H configuration.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx


# =============================================================================
# Attention
# =============================================================================


class VisionAttention(nnx.Module):
    """
    Bidirectional multi-head self-attention for the vision transformer.

    Unlike the language model's causal attention, vision attention is fully
    bidirectional — every patch token attends to every other patch token.
    No causal mask, no RoPE (positions are encoded via CPE before the blocks).

    Uses a single fused QKV projection (as in standard ViT) for efficiency.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        embed_dim: int,
        num_heads: int,
    ):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Single fused projection that produces Q, K, and V in one matmul.
        # Output is 3× the embed_dim so it can be split into three equal chunks.
        self.qkv = nnx.Linear(embed_dim, 3 * embed_dim, use_bias=True, rngs=rngs)

        # Output projection mixes information across heads.
        self.proj = nnx.Linear(embed_dim, embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: (B, T, embed_dim) — sequence of patch tokens (or CLS + patches)
        Returns:
            (B, T, embed_dim) — updated token representations
        """
        B, T, D = x.shape

        # Compute Q, K, V simultaneously, then reshape to expose head dimension.
        # qkv: (B, T, 3, num_heads, head_dim)
        qkv = self.qkv(x)
        qkv = jnp.reshape(qkv, (B, T, 3, self.num_heads, self.head_dim))

        # Separate into Q, K, V and move heads before sequence length.
        # Each: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention — NO causal mask (bidirectional).
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = jax.nn.softmax(scores, axis=-1)

        # Weighted sum of values, then merge heads back.
        context = jnp.einsum("bhqk,bhkd->bhqd", attn, v)  # (B, heads, T, head_dim)
        context = jnp.transpose(context, (0, 2, 1, 3))     # (B, T, heads, head_dim)
        context = jnp.reshape(context, (B, T, D))

        return self.proj(context)


# =============================================================================
# MLP
# =============================================================================


class VisionMLP(nnx.Module):
    """
    Point-wise feed-forward network used inside each ViT block.

    Expands the hidden dimension by mlp_ratio (4× for ViT-H), applies GELU
    nonlinearity, then projects back to embed_dim. This sub-network is applied
    independently to each token — it introduces no cross-token interaction.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        embed_dim: int,
        mlp_ratio: float = 4.0,
    ):
        mlp_dim = int(embed_dim * mlp_ratio)  # 1280 × 4 = 5120 for ViT-H
        self.fc1 = nnx.Linear(embed_dim, mlp_dim, use_bias=True, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.fc1(x)
        x = jax.nn.gelu(x, approximate=True)  # tanh approximation, slightly faster
        return self.fc2(x)


# =============================================================================
# Transformer Block
# =============================================================================


class VisionBlock(nnx.Module):
    """
    One ViT transformer block with pre-LayerNorm residual connections.

    Structure (pre-LN):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Pre-LN (normalise before the sub-layer, not after) is the standard in
    modern vision transformers and the RADIO family of models.
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attn = VisionAttention(rngs=rngs, embed_dim=embed_dim, num_heads=num_heads)
        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.mlp = VisionMLP(rngs=rngs, embed_dim=embed_dim, mlp_ratio=mlp_ratio)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Pre-LN attention sub-layer with residual.
        x = x + self.attn(self.norm1(x))
        # Pre-LN feed-forward sub-layer with residual.
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# RADIO Vision Encoder (C-RADIOv4-H)
# =============================================================================


class RADIOVisionEncoder(nnx.Module):
    """
    C-RADIOv4-H vision encoder as used in Nemotron 3 Nano Omni.

    This is a ViT-H/16 backbone with Conditional Positional Encoding (CPE).
    The default hyperparameters match the published C-RADIOv4-H checkpoint:
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0

    CPE (Conditional Positional Encoding):
        Positional embeddings are stored as a 2D grid at max_image_size resolution.
        At forward time they are bilinearly interpolated to match the actual input
        spatial dimensions. This lets the encoder handle any multiple-of-patch_size
        input resolution without any retraining.

    Returns:
        summary: (B, embed_dim)     CLS token — global image representation.
        spatial: (B, T, embed_dim)  Patch tokens — per-patch spatial features.
        h_patches: int              Number of patch rows (needed for pixel shuffle).
        w_patches: int              Number of patch columns (needed for pixel shuffle).
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        patch_size: int = 16,
        embed_dim: int = 1280,
        depth: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        in_channels: int = 3,
        max_image_size: int = 1840,  # largest resolution C-RADIOv4-H was trained at
    ):
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        # Maximum number of patches per side (used for CPE storage).
        # For max_image_size=1840 and patch_size=16: 1840/16 = 115 patches/side.
        max_patches = max_image_size // patch_size

        # ---- Patch embedding ------------------------------------------------
        # Each (patch_size × patch_size × C) pixel block becomes one embed_dim
        # vector via a linear projection (equivalent to a strided convolution).
        patch_area = patch_size * patch_size * in_channels
        self.patch_embed = nnx.Linear(
            patch_area, embed_dim, use_bias=True, rngs=rngs
        )

        # ---- CLS token -------------------------------------------------------
        # A single learnable vector prepended to the patch sequence.
        # After all transformer blocks, this token's output is the global
        # image representation (summary).
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))

        # ---- CPE positional embeddings ---------------------------------------
        # 2D grid of learnable positional embeddings stored at max resolution.
        # Shape: (1, max_patches^2, embed_dim) — no position for the CLS token,
        # since RADIO adds positional info only to patch tokens (not to CLS).
        pos_key = rngs.params()
        self.pos_embed = nnx.Param(
            jax.random.normal(pos_key, (1, max_patches * max_patches, embed_dim),
                              dtype=jnp.float32) * 0.02
        )
        self._max_patches = max_patches  # stored for interpolation logic

        # ---- Transformer blocks ----------------------------------------------
        self.blocks = nnx.List([
            VisionBlock(rngs=rngs, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # ---- Final normalization ---------------------------------------------
        # Applied to all output tokens (CLS + patches) after all blocks.
        self.norm = nnx.LayerNorm(embed_dim, rngs=rngs)

    def _to_channel_last(self, pixel_values: jax.Array) -> jax.Array:
        """Accept either NHWC (B, H, W, C) or NCHW (B, C, H, W) input."""
        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape (B, H, W, C) or (B, C, H, W)"
            )
        # Detect channel-first by checking if axis-1 matches the expected channel count.
        if (
            pixel_values.shape[1] == self.in_channels
            and pixel_values.shape[-1] != self.in_channels
        ):
            return jnp.transpose(pixel_values, (0, 2, 3, 1))
        if pixel_values.shape[-1] == self.in_channels:
            return pixel_values
        raise ValueError(
            f"Cannot infer channel layout from shape {pixel_values.shape}; "
            f"expected {self.in_channels} channels."
        )

    def _get_pos_enc(self, h_patches: int, w_patches: int) -> jax.Array:
        """
        Return positional encodings interpolated to the given spatial grid size.

        Stored embeddings are at (max_patches × max_patches).
        At forward time, they are bilinearly resized to (h_patches × w_patches).
        This is the core of CPE: any input resolution → correct positional context.
        """
        pos = self.pos_embed.value  # (1, max_patches^2, embed_dim)

        if h_patches == self._max_patches and w_patches == self._max_patches:
            return pos  # Already the stored resolution — no interpolation needed.

        # Reshape flat sequence to 2D spatial grid for bilinear interpolation.
        # (1, max_patches, max_patches, embed_dim)
        pos_2d = jnp.reshape(pos, (1, self._max_patches, self._max_patches, self.embed_dim))

        # Bilinear resize to the target resolution.
        # jax.image.resize uses align_corners=False convention (same as PyTorch default).
        pos_2d = jax.image.resize(
            pos_2d,
            shape=(1, h_patches, w_patches, self.embed_dim),
            method="linear",
            antialias=False,
        )

        # Flatten back to sequence format: (1, h_patches * w_patches, embed_dim)
        return jnp.reshape(pos_2d, (1, h_patches * w_patches, self.embed_dim))

    def __call__(
        self, pixel_values: jax.Array
    ) -> tuple[jax.Array, jax.Array, int, int]:
        """
        Encode an image batch with C-RADIOv4-H.

        Args:
            pixel_values: (B, H, W, 3) or (B, 3, H, W) float32 image tensor.
                          H and W must each be a multiple of patch_size (16).

        Returns:
            summary:   (B, embed_dim)     — CLS token, global image feature.
            spatial:   (B, T, embed_dim)  — patch tokens, T = (H/16)*(W/16).
            h_patches: int                — number of patch rows  (H // patch_size).
            w_patches: int                — number of patch columns (W // patch_size).
        """
        image = self._to_channel_last(pixel_values)  # ensure (B, H, W, C)
        B, H, W, C = image.shape

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        T = h_patches * w_patches  # total number of spatial patch tokens

        # ---- Step 1: Extract and embed patches -------------------------------
        # Slice image into (h_patches × w_patches) non-overlapping pixel blocks,
        # then flatten each block and project to embed_dim.
        # Reshape: (B, H, W, C) -> (B, h_p, patch_size, w_p, patch_size, C)
        x = jnp.reshape(image, (B, h_patches, self.patch_size,
                                 w_patches, self.patch_size, C))
        # Bring patch spatial dims together: (B, h_p, w_p, patch_size, patch_size, C)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        # Flatten each patch: (B, T, patch_size * patch_size * C)
        x = jnp.reshape(x, (B, T, self.patch_size * self.patch_size * C))
        # Linear projection to embed_dim: (B, T, embed_dim)
        x = self.patch_embed(x)

        # ---- Step 2: Conditional Positional Encoding (CPE) -------------------
        # Add interpolated positional embeddings to patch tokens only.
        # CLS token is positional-encoding-free, letting it focus on global content.
        x = x + self._get_pos_enc(h_patches, w_patches)

        # ---- Step 3: Prepend CLS token ---------------------------------------
        # Broadcast the single cls_token to the batch dimension, then concatenate.
        # Result: (B, 1 + T, embed_dim)
        cls = jnp.broadcast_to(self.cls_token.value, (B, 1, self.embed_dim))
        x = jnp.concatenate([cls, x], axis=1)

        # ---- Step 4: Transformer blocks ---------------------------------------
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # ---- Step 5: Split CLS and spatial outputs ---------------------------
        summary = x[:, 0]    # (B, embed_dim) — CLS token's final representation
        spatial = x[:, 1:]   # (B, T, embed_dim) — per-patch spatial features

        return summary, spatial, h_patches, w_patches


# =============================================================================
# Pixel-Shuffle Downsampling
# =============================================================================


def pixel_shuffle_down(
    spatial: jax.Array,
    h_patches: int,
    w_patches: int,
    scale: int = 2,
) -> jax.Array:
    """
    Reduce spatial token count by scale² via 2D pixel-shuffle (inverse pixel shuffle).

    This is the "pixel-shuffle 4× downsampling" described in the Nemotron 3
    Nano Omni paper. It merges scale×scale adjacent patch tokens into a single
    token with scale² more channels, effectively trading spatial resolution for
    channel width. For scale=2 (the paper default): 4× fewer tokens, 4× wider.

    The merge is purely a reshape — no learnable parameters are introduced here.
    The subsequent vision_projection layer in the multimodal model handles the
    linear mapping from the wider channel space to the LLM hidden dimension.

    Visual (scale=2, one spatial row):
        Patches: [A][B][C][D] [E][F][G][H]    (8 tokens, dim D)
        After 2×2 grouping:  [ABEF][CDGH]     (2 tokens, dim 4D)
              ↑↑  ↑↑  ↑↑  ↑↑
        Actual grouping is 2D (rows AND columns simultaneously).

    Args:
        spatial:   (B, h_patches * w_patches, D) — flat spatial token sequence.
        h_patches: Number of rows in the patch grid.
        w_patches: Number of columns in the patch grid.
        scale:     Merge factor per dimension (default 2 → 2×2 blocks → 4× fewer tokens).

    Returns:
        (B, (h_patches // scale) * (w_patches // scale), D * scale²)
    """
    B, T, D = spatial.shape
    assert h_patches % scale == 0, "h_patches must be divisible by scale"
    assert w_patches % scale == 0, "w_patches must be divisible by scale"

    h_out = h_patches // scale
    w_out = w_patches // scale

    # Restore the 2D spatial layout of patch tokens.
    x = jnp.reshape(spatial, (B, h_patches, w_patches, D))

    # Group into (scale × scale) blocks along both spatial axes.
    # Shape: (B, h_out, scale, w_out, scale, D)
    x = jnp.reshape(x, (B, h_out, scale, w_out, scale, D))

    # Bring the two scale axes adjacent to D so we can flatten them into channels.
    # (B, h_out, w_out, scale, scale, D)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))

    # Merge the scale² spatial positions into the channel dimension.
    # Output: (B, h_out * w_out, scale * scale * D)
    x = jnp.reshape(x, (B, h_out * w_out, scale * scale * D))

    return x
