"""
Minimal vision token encoder for multimodal Nemotron experiments.

This intentionally avoids large external dependencies so it can run in the
same environment as the existing text model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


class VisionPatchEncoder(nnx.Module):
    """
    Converts images into patch embeddings.

    Input supports either channel-last (B, H, W, C) or channel-first
    (B, C, H, W). Output shape is (B, num_patches, vision_dim).
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        image_size: int,
        patch_size: int,
        in_channels: int,
        vision_dim: int,
    ):
        if image_size <= 0:
            raise ValueError("image_size must be > 0")
        if patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if in_channels <= 0:
            raise ValueError("in_channels must be > 0")
        if vision_dim <= 0:
            raise ValueError("vision_dim must be > 0")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.vision_dim = vision_dim

        patch_area = patch_size * patch_size * in_channels
        self.patch_projection = nnx.Linear(
            patch_area,
            vision_dim,
            use_bias=False,
            rngs=rngs,
        )

    def _to_channel_last(self, pixel_values: jax.Array) -> jax.Array:
        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape (B, H, W, C) or (B, C, H, W)"
            )

        # Heuristic: if axis 1 matches channel count and axis -1 does not,
        # treat input as channel-first and transpose.
        if (
            pixel_values.shape[1] == self.in_channels
            and pixel_values.shape[-1] != self.in_channels
        ):
            return jnp.transpose(pixel_values, (0, 2, 3, 1))

        if pixel_values.shape[-1] == self.in_channels:
            return pixel_values

        raise ValueError(
            "Could not infer channel dimension from pixel_values shape "
            f"{pixel_values.shape}. Expected channels={self.in_channels}."
        )

    def __call__(self, pixel_values: jax.Array) -> jax.Array:
        image = self._to_channel_last(pixel_values)
        batch_size, height, width, channels = image.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {channels}"
            )
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                "pixel_values spatial shape does not match configured image_size: "
                f"expected ({self.image_size}, {self.image_size}), got ({height}, {width})"
            )

        p = self.patch_size
        h_blocks = height // p
        w_blocks = width // p

        # Reshape into non-overlapping patches then flatten each patch.
        patches = jnp.reshape(image, (batch_size, h_blocks, p, w_blocks, p, channels))
        patches = jnp.transpose(patches, (0, 1, 3, 2, 4, 5))
        patches = jnp.reshape(
            patches,
            (batch_size, h_blocks * w_blocks, p * p * channels),
        )

        return self.patch_projection(patches)
