"""
Minimal multimodal adapter around the existing Nemotron architecture.

Design goals:
- Keep text-only behavior available.
- Reuse existing hybrid blocks (Mamba + Attention + MoE).
- Add optional vision conditioning and optional action prediction head.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from nemotron import (
    MambaAttentionMoEBlock,
    MambaMoEBlock,
    NemotronConfig,
)
from vision_encoder import RADIOVisionEncoder, pixel_shuffle_down


@dataclass
class NemotronMultimodalConfig(NemotronConfig):
    """
    Configuration for multimodal Nemotron adapter.

    Vision encoder defaults match C-RADIOv4-H (nvidia/C-RADIOv4-H), the
    backbone used in the Nemotron 3 Nano Omni paper (arXiv:2604.24954).
    """

    use_vision: bool = False

    # Image resolution — used as the default/preferred size for preprocessing.
    # The encoder supports any multiple-of-patch_size resolution (dynamic resolution).
    image_size: int = 224

    # ViT-H/16 architecture parameters (C-RADIOv4-H defaults).
    patch_size: int = 16         # pixel width/height of each patch
    vision_in_channels: int = 3  # RGB input
    vision_dim: int = 1280       # encoder embed_dim (ViT-H)
    vision_depth: int = 32       # number of transformer blocks (ViT-H)
    vision_num_heads: int = 16   # attention heads per block (ViT-H)
    vision_mlp_ratio: float = 4.0  # MLP hidden dim = vision_dim × this ratio

    # Pixel-shuffle downsampling scale (2 → 2×2 blocks → 4× fewer tokens).
    # Applied between the vision encoder and the LLM projection.
    pixel_shuffle_scale: int = 2

    vision_fusion: str = "prepend"  # how visual tokens are inserted: prepend | append

    use_action_head: bool = False
    action_vocab_size: int = 256

    @property
    def num_vision_tokens(self) -> int:
        """
        Number of visual tokens fed to the LLM for a single image_size × image_size image,
        after pixel-shuffle downsampling.

        Used by app.py to allocate sequence length budget.
        """
        patches_per_side = self.image_size // self.patch_size
        downsampled_per_side = patches_per_side // self.pixel_shuffle_scale
        return downsampled_per_side * downsampled_per_side

    def validate(self) -> None:
        super().validate()

        if self.vision_fusion not in {"prepend", "append"}:
            raise ValueError("vision_fusion must be either 'prepend' or 'append'")

        if self.use_vision:
            if self.image_size <= 0:
                raise ValueError("image_size must be > 0")
            if self.patch_size <= 0:
                raise ValueError("patch_size must be > 0")
            stride = self.patch_size * self.pixel_shuffle_scale
            if self.image_size % stride != 0:
                raise ValueError(
                    f"image_size must be divisible by patch_size × pixel_shuffle_scale "
                    f"({self.patch_size} × {self.pixel_shuffle_scale} = {stride})"
                )
            if self.vision_in_channels <= 0:
                raise ValueError("vision_in_channels must be > 0")
            if self.vision_dim <= 0:
                raise ValueError("vision_dim must be > 0")

        if self.use_action_head and self.action_vocab_size <= 0:
            raise ValueError("action_vocab_size must be > 0 when action head is enabled")


class NemotronMultimodal(nnx.Module):
    """
    Nemotron-compatible multimodal model.

    Input pathways:
    - Text: token IDs -> token embedding
    - Vision (optional): images -> patch embeddings -> projection to d_model

    Outputs:
    - Text logits always available.
    - Action logits only when action head is enabled.
    """

    def __init__(self, rngs: nnx.Rngs, config: NemotronMultimodalConfig):
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
        self.lm_head = nnx.Linear(
            config.d_model,
            config.vocab_size,
            use_bias=False,
            rngs=rngs,
        )

        if config.use_vision:
            # C-RADIOv4-H: ViT-H/16 with CPE, 32 blocks, embed_dim=1280.
            self.vision_encoder = RADIOVisionEncoder(
                rngs=rngs,
                patch_size=config.patch_size,
                embed_dim=config.vision_dim,
                depth=config.vision_depth,
                num_heads=config.vision_num_heads,
                mlp_ratio=config.vision_mlp_ratio,
                in_channels=config.vision_in_channels,
            )
            # After pixel-shuffle (scale=2), each token has vision_dim * scale^2 channels.
            # This projection maps that wider representation to the LLM's d_model.
            ps = config.pixel_shuffle_scale
            self.vision_projection = nnx.Linear(
                config.vision_dim * ps * ps,
                config.d_model,
                use_bias=False,
                rngs=rngs,
            )

        if config.use_action_head:
            self.action_head = nnx.Linear(
                config.d_model,
                config.action_vocab_size,
                use_bias=False,
                rngs=rngs,
            )

    def _fuse_modalities(
        self,
        text_hidden: jax.Array,
        pixel_values: jax.Array | None,
    ) -> tuple[jax.Array, int]:
        """
        Returns fused hidden states and text token count.

        The returned text token count lets us slice the sequence back to text
        positions before the LM head.
        """
        text_token_count = int(text_hidden.shape[1])

        if not self.config.use_vision or pixel_values is None:
            return text_hidden, text_token_count

        # Encode: returns global summary (CLS) and per-patch spatial features.
        # h_patches / w_patches are needed to correctly perform pixel shuffle.
        _summary, spatial, h_patches, w_patches = self.vision_encoder(pixel_values)

        # Pixel-shuffle 4× downsampling: merge 2×2 patch blocks into one token.
        # Reduces token count by 4 and widens channels by 4.
        spatial = pixel_shuffle_down(
            spatial, h_patches, w_patches, scale=self.config.pixel_shuffle_scale
        )

        # Project vision tokens to the LLM's hidden dimension.
        vision_tokens = self.vision_projection(spatial)

        if self.config.vision_fusion == "prepend":
            fused = jnp.concatenate([vision_tokens, text_hidden], axis=1)
        else:
            fused = jnp.concatenate([text_hidden, vision_tokens], axis=1)

        return fused, text_token_count

    def _extract_text_hidden(
        self,
        fused_hidden: jax.Array,
        text_token_count: int,
    ) -> jax.Array:
        if (
            self.config.use_vision
            and self.config.vision_fusion == "prepend"
            and fused_hidden.shape[1] > text_token_count
        ):
            return fused_hidden[:, -text_token_count:, :]

        return fused_hidden[:, :text_token_count, :]

    def _select_action_hidden(
        self,
        text_hidden: jax.Array,
        action_positions: jax.Array | None,
    ) -> jax.Array:
        """
        Selects one hidden state per sample for action prediction.

        - If action_positions is provided, gather those indices.
        - Otherwise, default to the final text token.
        """
        if action_positions is None:
            return text_hidden[:, -1, :]

        if action_positions.ndim != 1:
            raise ValueError("action_positions must have shape (batch,)")

        seq_len = text_hidden.shape[1]
        clipped = jnp.clip(action_positions, 0, seq_len - 1)
        gather_idx = clipped[:, None, None]
        gather_idx = jnp.tile(gather_idx, (1, 1, text_hidden.shape[-1]))
        gathered = jnp.take_along_axis(text_hidden, gather_idx, axis=1)
        return gathered[:, 0, :]

    def __call__(
        self,
        token_ids: jax.Array,
        pixel_values: jax.Array | None = None,
        action_positions: jax.Array | None = None,
        return_dict: bool = False,
        return_action_logits: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array] | dict[str, jax.Array]:
        text_hidden = self.embedding(token_ids)
        fused_hidden, text_token_count = self._fuse_modalities(
            text_hidden=text_hidden,
            pixel_values=pixel_values,
        )

        x = fused_hidden
        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        text_hidden_out = self._extract_text_hidden(
            fused_hidden=x,
            text_token_count=text_token_count,
        )
        text_logits = self.lm_head(text_hidden_out)

        if not self.config.use_action_head:
            if return_dict:
                return {"text": text_logits}
            return text_logits

        action_hidden = self._select_action_hidden(
            text_hidden=text_hidden_out,
            action_positions=action_positions,
        )
        action_logits = self.action_head(action_hidden)

        if return_dict:
            return {
                "text": text_logits,
                "action": action_logits,
            }

        if not return_action_logits:
            return text_logits

        return text_logits, action_logits
