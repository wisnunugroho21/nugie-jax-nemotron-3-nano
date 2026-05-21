"""
Simple, minimal, and explainable Nemotron app.

This script shows a full small workflow:
1) Load a real dataset (roneneldan/TinyStories)
2) Train a tiny Nemotron language model
3) Evaluate it with validation loss + perplexity
4) Chat with it in the terminal

Design goals:
- Keep code easy to read and modify.
- Keep control flow explicit.
- Prefer clarity over speed/optimization.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from nemotron import NemotronConfig, NemotronNanoBlock
from nemotron_multimodal import NemotronMultimodal, NemotronMultimodalConfig
from task_router import HybridBatch, compute_hybrid_loss

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


CHECKPOINT_FORMAT_VERSION = 2

# =============================================================================
# Tokenizer
# =============================================================================


def load_hf_tokenizer(
    tokenizer_name: str,
    cache_dir: str | None = None,
) -> "PreTrainedTokenizerBase":
    """
    Loads a tokenizer from Hugging Face and guarantees core special tokens.

    We ensure PAD/BOS/EOS IDs exist because batching and generation rely on them.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Hugging Face tokenizer support requires the 'transformers' package. "
            "Install it with: pip install transformers"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            use_fast=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Hugging Face tokenizer '{tokenizer_name}'."
        ) from exc

    # Generation depends on EOS.
    if tokenizer.eos_token_id is None:
        if tokenizer.sep_token is not None:
            tokenizer.eos_token = tokenizer.sep_token
        else:
            tokenizer.add_special_tokens({"eos_token": "<eos>"})

    # We use BOS as a light conversation delimiter.
    if tokenizer.bos_token_id is None:
        if tokenizer.cls_token is not None:
            tokenizer.bos_token = tokenizer.cls_token
        else:
            tokenizer.add_special_tokens({"bos_token": "<bos>"})

    # Left padding is used for fixed-length context windows.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer


def _get_special_token_ids(
    tokenizer: "PreTrainedTokenizerBase",
) -> tuple[int, int, int]:
    """Returns guaranteed integer IDs for PAD/BOS/EOS tokens."""
    if (
        tokenizer.pad_token_id is None
        or tokenizer.bos_token_id is None
        or tokenizer.eos_token_id is None
    ):
        raise ValueError(
            "Tokenizer must define pad_token_id, bos_token_id, eos_token_id"
        )

    return (
        int(tokenizer.pad_token_id),  # type: ignore[arg-type]
        int(tokenizer.bos_token_id),  # type: ignore[arg-type]
        int(tokenizer.eos_token_id),  # type: ignore[arg-type]
    )


def encode_text(
    tokenizer: "PreTrainedTokenizerBase",
    text: str,
    add_bos: bool = False,
    add_eos: bool = False,
) -> list[int]:
    """Encodes text and optionally prepends/appends BOS/EOS."""
    _, bos_id, eos_id = _get_special_token_ids(tokenizer)

    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if add_bos:
        ids = [bos_id] + ids
    if add_eos:
        ids = ids + [eos_id]
    return ids


def _extract_story_text(example: dict[str, object]) -> str:
    """
    Reads one TinyStories row and returns its text.

    We check a few keys defensively so this stays easy to understand even if
    the upstream schema changes slightly.
    """
    for key in ("text", "story", "content"):
        value = example.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
    return ""


def _encode_segment(tokenizer: "PreTrainedTokenizerBase", text: str) -> list[int]:
    """Tokenizes raw text without adding any special tokens."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def encode_text_with_assistant_mask(
    tokenizer: "PreTrainedTokenizerBase",
    text: str,
    user_role_tag: str,
    assistant_role_tag: str,
) -> tuple[list[int], list[float]]:
    """
    Encodes one role-tagged conversation and returns per-token loss mask.

    Mask semantics:
    - 1.0 for assistant content tokens
    - 0.0 for user content tokens and role tag tokens
    """
    token_ids: list[int] = []
    loss_mask: list[float] = []

    if not user_role_tag or not assistant_role_tag:
        raise ValueError("user_role_tag and assistant_role_tag must be non-empty")

    cursor = 0
    current_role: str | None = None

    while cursor < len(text):
        next_user = text.find(user_role_tag, cursor)
        next_assistant = text.find(assistant_role_tag, cursor)

        candidates = [idx for idx in (next_user, next_assistant) if idx != -1]
        next_tag_index = min(candidates) if candidates else -1

        if next_tag_index == -1:
            chunk = text[cursor:]
            if chunk:
                chunk_ids = _encode_segment(tokenizer, chunk)
                mask_value = 1.0 if current_role == "assistant" else 0.0
                token_ids.extend(chunk_ids)
                loss_mask.extend([mask_value] * len(chunk_ids))
            break

        if next_tag_index > cursor:
            chunk = text[cursor:next_tag_index]
            if chunk:
                chunk_ids = _encode_segment(tokenizer, chunk)
                mask_value = 1.0 if current_role == "assistant" else 0.0
                token_ids.extend(chunk_ids)
                loss_mask.extend([mask_value] * len(chunk_ids))

        if text.startswith(user_role_tag, next_tag_index):
            tag_text = user_role_tag
            current_role = "user"
        else:
            tag_text = assistant_role_tag
            current_role = "assistant"

        tag_ids = _encode_segment(tokenizer, tag_text)
        token_ids.extend(tag_ids)
        loss_mask.extend([0.0] * len(tag_ids))

        cursor = next_tag_index + len(tag_text)

    return token_ids, loss_mask


# =============================================================================
# Dataset
# =============================================================================


def load_tinystories_texts(
    max_stories: int,
    split: str = "train",
    cache_dir: str | None = None,
) -> list[str]:
    """
    Loads a bounded number of stories from roneneldan/TinyStories.

    Why a bounded subset?
    - Keeps the demo easy to run locally.
    - Keeps the data flow simple and inspectable.
    - Streams rows and stops early at `max_stories`.
    """
    if max_stories < 2:
        raise ValueError("max_stories must be at least 2")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "TinyStories training requires the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    try:
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            cache_dir=cache_dir,
            streaming=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load roneneldan/TinyStories. "
            "Check your internet connection and cache permissions."
        ) from exc

    stories: list[str] = []

    # Intentionally iterate in plain Python for readability.
    for example in dataset:
        story = _extract_story_text(example)
        if story:
            stories.append(story)
        if len(stories) >= max_stories:
            break

    if len(stories) < 2:
        raise ValueError("TinyStories did not return enough non-empty stories")

    return stories


def split_train_val_texts(
    stories: list[str],
    train_ratio: float,
) -> tuple[list[str], list[str]]:
    """
    Splits stories into train and validation lists.

    This uses a deterministic ordered split to keep behavior simple and
    reproducible.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(stories) < 2:
        raise ValueError("Need at least 2 stories for train/validation split")

    split_index = int(len(stories) * train_ratio)
    split_index = max(1, split_index)
    split_index = min(split_index, len(stories) - 1)

    train_texts = stories[:split_index]
    val_texts = stories[split_index:]
    return train_texts, val_texts


@dataclass
class MultimodalExample:
    """One preprocessed multimodal training example."""

    token_ids: np.ndarray
    text_labels: np.ndarray
    text_loss_mask: np.ndarray
    pixel_values: np.ndarray
    action_label: int | None


def _encode_text_for_multimodal_example(
    tokenizer: "PreTrainedTokenizerBase",
    text: str,
    text_seq_len: int,
    assistant_only_loss: bool,
    user_role_tag: str,
    assistant_role_tag: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts one text sample to fixed-length next-token arrays.

    Output arrays all use length `text_seq_len`:
    - token_ids      : input IDs
    - text_labels    : shifted labels
    - text_loss_mask : shifted supervision mask
    """
    if text_seq_len <= 0:
        raise ValueError("text_seq_len must be > 0")

    pad_id, bos_id, eos_id = _get_special_token_ids(tokenizer)
    if assistant_only_loss:
        ids, mask = encode_text_with_assistant_mask(
            tokenizer=tokenizer,
            text=text,
            user_role_tag=user_role_tag,
            assistant_role_tag=assistant_role_tag,
        )
        ids = [bos_id] + ids + [eos_id]
        mask = [0.0] + mask + [0.0]
    else:
        ids = encode_text(tokenizer, text, add_bos=True, add_eos=True)
        mask = [1.0] * len(ids)

    target_len = text_seq_len + 1
    if len(ids) >= target_len:
        ids = ids[-target_len:]
        mask = mask[-target_len:]
    else:
        pad_count = target_len - len(ids)
        ids = [pad_id] * pad_count + ids
        mask = [0.0] * pad_count + mask

    ids_arr = np.asarray(ids, dtype=np.int32)
    mask_arr = np.asarray(mask, dtype=np.float32)

    token_ids = ids_arr[:-1]
    text_labels = ids_arr[1:]
    text_loss_mask = mask_arr[1:]
    return token_ids, text_labels, text_loss_mask


def _load_image_array(
    image_path: Path,
    image_size: int,
    image_channels: int,
) -> np.ndarray:
    """Loads and normalizes one image to (H, W, C), float32 in [-1, 1]."""
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        image = np.load(image_path)
    else:
        try:
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Image loading for non-.npy files requires Pillow. "
                "Install it with: pip install pillow"
            ) from exc

        with Image.open(image_path) as img:
            if image_channels == 3:
                img = img.convert("RGB")
            elif image_channels == 1:
                img = img.convert("L")
            else:
                raise ValueError("Only image_channels=1 or 3 are currently supported")
            img = img.resize((image_size, image_size))
            image = np.asarray(img)

    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError(f"Image must decode to rank-3 array, got shape {image.shape}")

    if image.shape[-1] != image_channels:
        raise ValueError(
            f"Image channel mismatch for {image_path}: "
            f"expected {image_channels}, got {image.shape[-1]}"
        )

    if image.shape[0] != image_size or image.shape[1] != image_size:
        # npy inputs are not resized automatically to keep behavior explicit.
        raise ValueError(
            f"Image spatial size mismatch for {image_path}: expected "
            f"({image_size}, {image_size}), got ({image.shape[0]}, {image.shape[1]})"
        )

    image_f32 = image.astype(np.float32)
    if image_f32.max() > 1.0 or image_f32.min() < -1.0:
        image_f32 = (image_f32 / 127.5) - 1.0

    return image_f32


def load_multimodal_jsonl_examples(
    jsonl_path: str,
    max_records: int,
    tokenizer: "PreTrainedTokenizerBase",
    text_seq_len: int,
    image_size: int,
    image_channels: int,
    text_key: str,
    image_key: str,
    action_key: str,
    image_root: str | None,
    assistant_only_loss: bool,
    user_role_tag: str,
    assistant_role_tag: str,
    require_action: bool,
) -> list[MultimodalExample]:
    """Loads real multimodal JSONL data into preprocessed examples."""
    if max_records <= 0:
        raise ValueError("max_records must be > 0")

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Multimodal JSONL file not found: {jsonl_path}")

    root_path = Path(image_root) if image_root else path.parent

    examples: list[MultimodalExample] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if len(examples) >= max_records:
                break

            stripped = line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}"
                ) from exc

            if not isinstance(obj, dict):
                continue

            raw_text = obj.get(text_key)
            raw_image = obj.get(image_key)
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue
            if not isinstance(raw_image, str) or not raw_image.strip():
                continue

            image_path = Path(raw_image)
            if not image_path.is_absolute():
                image_path = (root_path / image_path).resolve()
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image file not found at line {line_number}: {image_path}"
                )

            action_label: int | None = None
            if action_key in obj and obj.get(action_key) is not None:
                raw_action = obj.get(action_key)
                if isinstance(raw_action, (int, np.integer)):
                    action_label = int(raw_action)
                elif isinstance(raw_action, str) and raw_action.strip().isdigit():
                    action_label = int(raw_action.strip())
                else:
                    raise ValueError(
                        f"Invalid action value at line {line_number}: {raw_action!r}"
                    )

            if require_action and action_label is None:
                raise ValueError(
                    f"Missing required action label '{action_key}' at line {line_number}"
                )

            token_ids, text_labels, text_loss_mask = _encode_text_for_multimodal_example(
                tokenizer=tokenizer,
                text=raw_text,
                text_seq_len=text_seq_len,
                assistant_only_loss=assistant_only_loss,
                user_role_tag=user_role_tag,
                assistant_role_tag=assistant_role_tag,
            )
            pixel_values = _load_image_array(
                image_path=image_path,
                image_size=image_size,
                image_channels=image_channels,
            )
            examples.append(
                MultimodalExample(
                    token_ids=token_ids,
                    text_labels=text_labels,
                    text_loss_mask=text_loss_mask,
                    pixel_values=pixel_values,
                    action_label=action_label,
                )
            )

    if len(examples) < 2:
        raise ValueError(
            f"Need at least 2 valid multimodal records in {jsonl_path}; got {len(examples)}"
        )

    return examples


def validate_multimodal_jsonl_integrity(
    jsonl_path: str,
    image_size: int,
    image_channels: int,
    text_key: str,
    image_key: str,
    action_key: str,
    image_root: str | None,
    require_action: bool,
    max_reported_errors: int = 8,
) -> dict[str, int]:
    """
    Validates multimodal JSONL records before training.

    Checks include:
    - JSON parse validity and object structure
    - Required text/image keys and non-empty values
    - Required action label (for VLA) and integer parseability
    - Image file existence and shape/channel compatibility
    """
    if max_reported_errors <= 0:
        raise ValueError("max_reported_errors must be > 0")

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Multimodal JSONL file not found: {jsonl_path}")

    root_path = Path(image_root) if image_root else path.parent

    total_lines = 0
    valid_records = 0
    skipped_blank = 0
    errors: list[str] = []

    def record_error(msg: str) -> None:
        if len(errors) < max_reported_errors:
            errors.append(msg)

    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            total_lines += 1
            stripped = line.strip()
            if not stripped:
                skipped_blank += 1
                continue

            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                record_error(f"line {line_number}: invalid JSON ({exc.msg})")
                continue

            if not isinstance(obj, dict):
                record_error(f"line {line_number}: expected JSON object")
                continue

            raw_text = obj.get(text_key)
            if not isinstance(raw_text, str) or not raw_text.strip():
                record_error(
                    f"line {line_number}: missing/empty text key '{text_key}'"
                )
                continue

            raw_image = obj.get(image_key)
            if not isinstance(raw_image, str) or not raw_image.strip():
                record_error(
                    f"line {line_number}: missing/empty image key '{image_key}'"
                )
                continue

            action_value = obj.get(action_key)
            if require_action and action_value is None:
                record_error(
                    f"line {line_number}: missing required action key '{action_key}'"
                )
                continue

            if action_value is not None:
                is_int = isinstance(action_value, (int, np.integer))
                is_digit_string = (
                    isinstance(action_value, str) and action_value.strip().isdigit()
                )
                if not (is_int or is_digit_string):
                    record_error(
                        f"line {line_number}: invalid action value for '{action_key}'"
                    )
                    continue

            image_path = Path(raw_image)
            if not image_path.is_absolute():
                image_path = (root_path / image_path).resolve()

            if not image_path.exists():
                record_error(f"line {line_number}: image file missing at {image_path}")
                continue

            try:
                _load_image_array(
                    image_path=image_path,
                    image_size=image_size,
                    image_channels=image_channels,
                )
            except Exception as exc:
                record_error(f"line {line_number}: image validation failed ({exc})")
                continue

            valid_records += 1

    if errors:
        summary = (
            f"Multimodal JSONL integrity check failed for {jsonl_path}. "
            f"valid_records={valid_records}, total_lines={total_lines}, "
            f"blank_lines={skipped_blank}."
        )
        details = "\n".join(f"- {item}" for item in errors)
        raise ValueError(f"{summary}\n{details}")

    return {
        "total_lines": total_lines,
        "blank_lines": skipped_blank,
        "valid_records": valid_records,
    }


def load_jsonl_texts(
    jsonl_path: str,
    max_records: int,
    text_key: str = "serialized_text",
) -> list[str]:
    """
    Loads text samples from a JSONL file.

    Each non-empty line must contain a JSON object with a string field at
    `text_key` (default: serialized conversation text).
    """
    if max_records <= 0:
        raise ValueError("max_records must be > 0")

    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    texts: list[str] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if len(texts) >= max_records:
                break

            stripped = line.strip()
            if not stripped:
                continue

            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}"
                ) from exc

            value = obj.get(text_key) if isinstance(obj, dict) else None
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    texts.append(cleaned)

    if len(texts) < 2:
        raise ValueError(
            f"Need at least 2 non-empty records in {jsonl_path} for training/eval"
        )

    return texts


def _ensure_min_length(tokens: jax.Array, min_length: int) -> jax.Array:
    """
    Repeats tokens until we have enough positions for batching.

    This keeps the data pipeline simple for very tiny demo corpora.
    """
    if int(tokens.shape[0]) >= min_length:
        return tokens

    repeat_count = int(math.ceil(min_length / int(tokens.shape[0])))
    return jnp.tile(tokens, repeat_count)


def prepare_datasets(
    tokenizer: "PreTrainedTokenizerBase",
    train_texts: list[str],
    val_texts: list[str],
    seq_len: int,
    assistant_only_loss: bool = False,
    user_role_tag: str = "<|user|>",
    assistant_role_tag: str = "<|assistant|>",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Creates train/val token streams from a list of text samples.

    Works with both plain text (TinyStories) and role-tagged chat text (JSONL).
    When assistant_only_loss=True, also computes a per-token loss mask that
    marks only assistant content as supervised.

    Output format:
    - train_tokens: shape (num_train_tokens,)
    - val_tokens: shape (num_val_tokens,)
    - train_loss_mask: shape (num_train_tokens,)  — all 1.0 when masking is off
    - val_loss_mask: shape (num_val_tokens,)      — all 1.0 when masking is off
    """
    if not train_texts or not val_texts:
        raise ValueError("train_texts and val_texts must both be non-empty")

    def build_stream(texts: list[str]) -> tuple[list[int], list[float]]:
        if not assistant_only_loss:
            joined = "\n\n".join(texts)
            ids = encode_text(tokenizer, joined, add_bos=True, add_eos=True)
            return ids, [1.0] * len(ids)

        _, bos_id, eos_id = _get_special_token_ids(tokenizer)
        sep_ids = _encode_segment(tokenizer, "\n\n")

        all_ids: list[int] = [bos_id]
        all_mask: list[float] = [0.0]

        for text_index, text in enumerate(texts):
            ids, mask = encode_text_with_assistant_mask(
                tokenizer=tokenizer,
                text=text,
                user_role_tag=user_role_tag,
                assistant_role_tag=assistant_role_tag,
            )
            all_ids.extend(ids)
            all_mask.extend(mask)

            if text_index < len(texts) - 1:
                all_ids.extend(sep_ids)
                all_mask.extend([0.0] * len(sep_ids))

        all_ids.append(eos_id)
        all_mask.append(0.0)

        return all_ids, all_mask

    train_ids, train_mask = build_stream(train_texts)
    val_ids, val_mask = build_stream(val_texts)

    train_tokens = jnp.array(train_ids, dtype=jnp.int32)
    val_tokens = jnp.array(val_ids, dtype=jnp.int32)
    train_loss_mask = jnp.array(train_mask, dtype=jnp.float32)
    val_loss_mask = jnp.array(val_mask, dtype=jnp.float32)

    # Ensure both splits are large enough to sample (x, y) windows.
    min_stream_len = seq_len + 2
    train_tokens = _ensure_min_length(train_tokens, min_stream_len)
    val_tokens = _ensure_min_length(val_tokens, min_stream_len)
    train_loss_mask = _ensure_min_length(train_loss_mask, min_stream_len)
    val_loss_mask = _ensure_min_length(val_loss_mask, min_stream_len)

    return train_tokens, val_tokens, train_loss_mask, val_loss_mask


# =============================================================================
# Optimization
# =============================================================================


def cross_entropy_loss(
    logits: jax.Array,
    labels: jax.Array,
    loss_mask: jax.Array | None = None,
) -> jax.Array:
    """Language-model cross-entropy with optional token-level masking."""
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    losses = optax.softmax_cross_entropy(logits, one_hot)

    if loss_mask is None:
        return losses.mean()

    mask = loss_mask.astype(losses.dtype)
    masked_sum = jnp.sum(losses * mask)
    denom = jnp.sum(mask)
    return jnp.where(denom > 0, masked_sum / denom, losses.mean())


def create_lr_schedule(
    max_steps: int, warmup_steps: int, peak_lr: float
) -> optax.Schedule:
    """
    Creates a two-phase learning rate schedule:
      Phase 1 (steps 0 .. warmup_steps):        linear ramp  0 → peak_lr
      Phase 2 (steps warmup_steps .. max_steps): cosine decay peak_lr → 0

    This avoids early training instability (warmup) while allowing the
    optimizer to fine-tune at smaller learning rates later (cosine decay).
    """
    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps,
    )
    decay = optax.cosine_decay_schedule(
        init_value=peak_lr,
        decay_steps=max(max_steps - warmup_steps, 1),
    )
    return optax.join_schedules(
        schedules=[warmup, decay],
        boundaries=[warmup_steps],
    )


def _update_all_expert_biases(model: NemotronNanoBlock | NemotronMultimodal) -> None:
    """
    Update expert biases across every MoE layer after a training step.

    This is the aux-loss-free load balancing step (Wang et al. 2024, §2.4).
    Each MoE layer stored the top-k indices from its last forward pass in
    `moe.last_topk_indices`. We read those indices here and call
    `update_expert_bias`, which nudges each expert's bias by +/-bias_update_rate
    depending on whether that expert was over- or under-utilized.

    IMPORTANT: Call this AFTER optimizer.update, outside the gradient computation.
    The expert_bias is an nnx.Variable (not nnx.Param) so the optimizer does
    not touch it — it is updated only here.
    """
    for block in model.blocks:
        block.moe.update_expert_bias(block.moe.last_topk_indices[...])


def sample_lm_batch(
    token_stream: jax.Array,
    loss_mask_stream: jax.Array,
    batch_size: int,
    seq_len: int,
    rng_key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Samples random contiguous windows for next-token prediction.

    For each sampled window of length (seq_len + 1):
    - x = first seq_len tokens
    - y = next seq_len tokens
    """
    max_start = int(token_stream.shape[0]) - (seq_len + 1)
    if max_start < 0:
        raise ValueError("token_stream is shorter than seq_len + 1")

    starts = jax.random.randint(rng_key, (batch_size,), 0, max_start + 1)

    x_list: list[jax.Array] = []
    y_list: list[jax.Array] = []
    m_list: list[jax.Array] = []
    for start in starts.tolist():
        token_window = token_stream[start : start + seq_len + 1]
        mask_window = loss_mask_stream[start : start + seq_len + 1]
        x_list.append(token_window[:-1])
        y_list.append(token_window[1:])
        m_list.append(mask_window[1:])

    x = jnp.stack(x_list, axis=0)
    y = jnp.stack(y_list, axis=0)
    y_mask = jnp.stack(m_list, axis=0)
    return x, y, y_mask


def sample_multimodal_batch(
    examples: list[MultimodalExample],
    batch_size: int,
    rng_key: jax.Array,
    task_type: str,
) -> HybridBatch:
    """
    Samples a random multimodal minibatch from preloaded examples.
    """
    if task_type not in {"vlm", "vla"}:
        raise ValueError("task_type must be 'vlm' or 'vla'")
    if len(examples) == 0:
        raise ValueError("examples must be non-empty")

    indices = jax.random.randint(
        rng_key,
        shape=(batch_size,),
        minval=0,
        maxval=len(examples),
    )
    selected = [examples[int(i)] for i in indices.tolist()]

    token_ids = jnp.asarray(np.stack([ex.token_ids for ex in selected], axis=0))
    text_labels = jnp.asarray(np.stack([ex.text_labels for ex in selected], axis=0))
    text_loss_mask = jnp.asarray(
        np.stack([ex.text_loss_mask for ex in selected], axis=0)
    )
    pixel_values = jnp.asarray(np.stack([ex.pixel_values for ex in selected], axis=0))

    action_labels: jax.Array | None
    if task_type == "vla":
        maybe_actions = [ex.action_label for ex in selected]
        if any(action is None for action in maybe_actions):
            raise ValueError("VLA batch contains missing action_label values")
        action_labels = jnp.asarray(maybe_actions, dtype=jnp.int32)
    else:
        action_labels = None

    return HybridBatch(
        token_ids=token_ids,
        text_labels=text_labels,
        text_loss_mask=text_loss_mask,
        pixel_values=pixel_values,
        action_labels=action_labels,
        action_positions=None,
        task_type=task_type,
    )


def train_model(
    model: NemotronNanoBlock,
    optimizer: nnx.Optimizer,
    train_tokens: jax.Array,
    train_loss_mask: jax.Array,
    steps: int,
    batch_size: int,
    seq_len: int,
    rng_key: jax.Array,
    debug_mask_ratio: bool = False,
    debug_mask_every: int = 1,
) -> jax.Array:
    """Runs a tiny training loop and prints readable metrics."""

    if debug_mask_every <= 0:
        raise ValueError("debug_mask_every must be > 0")

    mask_ratios: list[float] = []

    @nnx.jit
    def train_step(
        model: NemotronNanoBlock,
        optimizer: nnx.Optimizer,
        x_batch: jax.Array,
        y_batch: jax.Array,
        y_mask_batch: jax.Array,
    ) -> jax.Array:
        def loss_fn(model: NemotronNanoBlock) -> jax.Array:
            logits_local = model(x_batch)
            return cross_entropy_loss(logits_local, y_batch, loss_mask=y_mask_batch)

        total_loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

        # Aux-loss-free load balancing: update expert biases AFTER the gradient step.
        # Uses the top-k indices stored by each SparseMoE during the forward pass.
        _update_all_expert_biases(model)

        return total_loss

    print("\nTraining:")
    for step in range(steps):
        rng_key, batch_key = jax.random.split(rng_key)
        x_batch, y_batch, y_mask_batch = sample_lm_batch(
            train_tokens,
            train_loss_mask,
            batch_size,
            seq_len,
            batch_key,
        )
        total_loss = train_step(model, optimizer, x_batch, y_batch, y_mask_batch)

        print(f"  step {step + 1:>3}/{steps} | ce={float(total_loss):.4f}")
        if debug_mask_ratio:
            supervised_tokens = float(jnp.sum(y_mask_batch))
            total_tokens = int(y_mask_batch.size)
            ratio = supervised_tokens / max(total_tokens, 1)
            mask_ratios.append(ratio)
            
            if ((step + 1) % debug_mask_every == 0):
                print(
                    "    mask-debug "
                    f"supervised={supervised_tokens:.1f}/{total_tokens} "
                    f"ratio={ratio:.4f}"
                )

    if debug_mask_ratio and mask_ratios:
        ratio_min = min(mask_ratios)
        ratio_max = max(mask_ratios)
        ratio_mean = sum(mask_ratios) / len(mask_ratios)
        print(
            "Mask ratio summary "
            f"min={ratio_min:.4f} "
            f"max={ratio_max:.4f} "
            f"mean={ratio_mean:.4f}"
        )

    return rng_key


def train_model_multimodal(
    model: NemotronMultimodal,
    optimizer: nnx.Optimizer,
    train_examples: list[MultimodalExample],
    steps: int,
    batch_size: int,
    task_type: str,
    rng_key: jax.Array,
) -> jax.Array:
    """Runs multimodal training with text-only or text+action loss."""

    @nnx.jit
    def train_step(
        model: NemotronMultimodal,
        optimizer: nnx.Optimizer,
        token_ids: jax.Array,
        text_labels: jax.Array,
        text_loss_mask: jax.Array,
        pixel_values: jax.Array,
        action_labels: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        def loss_fn(model: NemotronMultimodal) -> tuple[jax.Array, jax.Array]:
            outputs = model(
                token_ids,
                pixel_values=pixel_values,
                return_dict=True,
                return_action_logits=(task_type == "vla"),
            )
            if not isinstance(outputs, dict):
                raise TypeError("Expected dict outputs in multimodal train_step")
            text_logits = outputs["text"]
            action_logits = outputs.get("action")

            total_loss, metrics = compute_hybrid_loss(
                text_logits=text_logits,
                text_labels=text_labels,
                text_loss_mask=text_loss_mask,
                action_logits=action_logits,
                action_labels=action_labels,
            )
            action_metric = metrics.get("action_loss", jnp.array(0.0))
            return total_loss, action_metric

        (total_loss, action_loss), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
            model
        )
        optimizer.update(model, grads)
        _update_all_expert_biases(model)
        return total_loss, action_loss

    print("\nTraining:")
    for step in range(steps):
        rng_key, batch_key = jax.random.split(rng_key)
        batch = sample_multimodal_batch(
            examples=train_examples,
            batch_size=batch_size,
            rng_key=batch_key,
            task_type=task_type,
        )
        total_loss, action_loss = train_step(
            model=model,
            optimizer=optimizer,
            token_ids=batch.token_ids,
            text_labels=batch.text_labels,
            text_loss_mask=batch.text_loss_mask
            if batch.text_loss_mask is not None
            else jnp.ones_like(batch.text_labels, dtype=jnp.float32),
            pixel_values=batch.pixel_values
            if batch.pixel_values is not None
            else jnp.zeros((batch_size, 1, 1, 3), dtype=jnp.float32),
            action_labels=batch.action_labels,
        )

        if task_type == "vla":
            print(
                f"  step {step + 1:>3}/{steps} | total={float(total_loss):.4f} "
                f"| action={float(action_loss):.4f}"
            )
        else:
            print(f"  step {step + 1:>3}/{steps} | ce={float(total_loss):.4f}")

    return rng_key


def evaluate_model(
    model: NemotronNanoBlock,
    val_tokens: jax.Array,
    val_loss_mask: jax.Array,
    batch_size: int,
    seq_len: int,
    eval_batches: int,
    rng_key: jax.Array,
) -> tuple[float, float, jax.Array]:
    """
    Evaluates model on validation batches.

    Returns:
    - mean_ce_loss
    - perplexity = exp(mean_ce_loss)
    - updated rng_key
    """
    ce_losses: list[jax.Array] = []

    for _ in range(eval_batches):
        rng_key, batch_key = jax.random.split(rng_key)
        x_batch, y_batch, y_mask_batch = sample_lm_batch(
            val_tokens,
            val_loss_mask,
            batch_size,
            seq_len,
            batch_key,
        )
        logits = model(x_batch)
        ce_losses.append(cross_entropy_loss(logits, y_batch, loss_mask=y_mask_batch))

    mean_ce = jnp.mean(jnp.stack(ce_losses))
    ppl = jnp.exp(mean_ce)

    return float(mean_ce), float(ppl), rng_key


def evaluate_model_multimodal(
    model: NemotronMultimodal,
    val_examples: list[MultimodalExample],
    batch_size: int,
    eval_batches: int,
    task_type: str,
    rng_key: jax.Array,
) -> tuple[float, float, jax.Array]:
    """
    Evaluates multimodal model.

    Returns mean total loss and exp(mean total loss) for parity with text mode.
    """
    losses: list[jax.Array] = []

    for _ in range(eval_batches):
        rng_key, batch_key = jax.random.split(rng_key)
        batch = sample_multimodal_batch(
            examples=val_examples,
            batch_size=batch_size,
            rng_key=batch_key,
            task_type=task_type,
        )

        outputs = model(
            batch.token_ids,
            pixel_values=batch.pixel_values,
            return_dict=True,
            return_action_logits=(task_type == "vla"),
        )
        if not isinstance(outputs, dict):
            raise TypeError("Expected dict outputs in multimodal evaluate")
        total_loss, _ = compute_hybrid_loss(
            text_logits=outputs["text"],
            text_labels=batch.text_labels,
            text_loss_mask=batch.text_loss_mask,
            action_logits=outputs.get("action"),
            action_labels=batch.action_labels,
        )
        losses.append(total_loss)

    mean_loss = jnp.mean(jnp.stack(losses))
    pseudo_ppl = jnp.exp(mean_loss)
    return float(mean_loss), float(pseudo_ppl), rng_key


# =============================================================================
# Chat
# =============================================================================


def pad_or_trim_context(token_ids: list[int], seq_len: int, pad_id: int) -> jax.Array:
    """
    Makes context length exactly `seq_len` so Mamba chunking constraints hold.

    - If context is too long, keep the most recent tokens.
    - If context is too short, left-pad with <pad>.
    """
    if len(token_ids) >= seq_len:
        fixed = token_ids[-seq_len:]
    else:
        pad_count = seq_len - len(token_ids)
        fixed = [pad_id] * pad_count + token_ids

    return jnp.array([fixed], dtype=jnp.int32)


def sample_next_token(
    logits: jax.Array,
    temperature: float,
    rng_key: jax.Array,
) -> int:
    """
    Picks the next token.

    - temperature <= 0: greedy argmax
    - temperature > 0: categorical sampling after temperature scaling
    """
    if temperature <= 0.0:
        return int(jnp.argmax(logits))

    safe_temp = max(temperature, 1e-6)
    scaled = logits / safe_temp
    return int(jax.random.categorical(rng_key, scaled))


def generate_reply(
    model: NemotronNanoBlock | NemotronMultimodal,
    tokenizer: "PreTrainedTokenizerBase",
    prompt_text: str,
    seq_len: int,
    max_new_tokens: int,
    temperature: float,
    rng_key: jax.Array,
) -> tuple[str, jax.Array]:
    """Autoregressively generates assistant text from a prompt."""
    pad_id, bos_id, eos_id = _get_special_token_ids(tokenizer)

    context_ids = encode_text(tokenizer, prompt_text, add_bos=True, add_eos=False)
    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        model_input = pad_or_trim_context(context_ids, seq_len, pad_id)
        logits = model(model_input)
        if isinstance(logits, tuple):
            logits = logits[0]
        elif isinstance(logits, dict):
            logits = logits["text"]

        next_logits = logits[0, -1]

        # Avoid sampling control tokens during response generation.
        # Some tokenizers reuse eos as pad, so guard against masking eos by accident.
        if pad_id != eos_id:
            next_logits = next_logits.at[pad_id].set(-1e9)
        if bos_id != eos_id and bos_id != pad_id:
            next_logits = next_logits.at[bos_id].set(-1e9)

        rng_key, sample_key = jax.random.split(rng_key)
        next_id = sample_next_token(next_logits, temperature, sample_key)

        if next_id == eos_id:
            break

        context_ids.append(next_id)
        generated_ids.append(next_id)

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return str(decoded), rng_key


def build_chat_prompt(
    turns: list[tuple[str, str]],
    user_text: str,
    user_role_tag: str,
    assistant_role_tag: str,
) -> str:
    """
    Builds a role-tagged multi-turn prompt from chat history and latest user text.

    Format:
      <|user|>...<|assistant|>...<|user|>...<|assistant|>
    """
    parts: list[str] = []
    for prev_user, prev_assistant in turns:
        parts.append(f"{user_role_tag}\n{prev_user.strip()}\n")
        parts.append(f"{assistant_role_tag}\n{prev_assistant.strip()}\n")

    parts.append(f"{user_role_tag}\n{user_text.strip()}\n")
    parts.append(f"{assistant_role_tag}\n")
    return "".join(parts)


def chat_loop(
    model: NemotronNanoBlock | NemotronMultimodal,
    tokenizer: "PreTrainedTokenizerBase",
    seq_len: int,
    temperature: float,
    max_new_tokens: int,
    rng_key: jax.Array,
    user_role_tag: str,
    assistant_role_tag: str,
    history_turns: int,
) -> None:
    """Runs an interactive terminal chatbot with short conversation memory."""
    if history_turns <= 0:
        raise ValueError("history_turns must be > 0")

    print("\nChatbot is ready. Commands: /help, /reset, /exit")
    print(f"Using up to {history_turns} previous turns as prompt context.")

    turns: list[tuple[str, str]] = []

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: bye.")
            break

        lower_text = user_text.lower()
        if lower_text in {"quit", "exit", "/exit"}:
            print("Bot: bye.")
            break
        if not user_text:
            continue
        if lower_text == "/help":
            print("Bot: /reset clears memory, /exit ends chat.")
            continue
        if lower_text == "/reset":
            turns.clear()
            print("Bot: conversation memory cleared.")
            continue

        prompt_text = build_chat_prompt(
            turns=turns[-history_turns:],
            user_text=user_text,
            user_role_tag=user_role_tag,
            assistant_role_tag=assistant_role_tag,
        )

        reply, rng_key = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            seq_len=seq_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            rng_key=rng_key,
        )

        reply = reply.strip() or "..."
        turns.append((user_text, reply))
        print(f"Bot: {reply}")


def _model_kind(model: NemotronNanoBlock | NemotronMultimodal) -> str:
    if isinstance(model, NemotronMultimodal):
        if model.config.use_action_head:
            return "vla"
        if model.config.use_vision:
            return "vlm"
    return "text"


def _collect_checkpoint_metadata(
    model: NemotronNanoBlock | NemotronMultimodal,
) -> dict[str, Any]:
    param_state = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(param_state)
    shapes = [list(jnp.asarray(leaf).shape) for leaf in leaves]

    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_kind": _model_kind(model),
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "param_count": len(leaves),
        "param_shapes": shapes,
    }


def save_checkpoint_from_model(
    model: NemotronNanoBlock | NemotronMultimodal,
    checkpoint_path: str,
) -> int:
    """
    Saves model parameters in .npz format with metadata.

    - Parameter leaves keep the legacy key format: param_0, param_1, ...
    - Metadata is stored at key: __metadata_json__
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    param_state = nnx.state(model, nnx.Param)
    param_leaves = jax.tree_util.tree_leaves(param_state)

    payload: dict[str, np.ndarray] = {}
    for leaf_index, leaf in enumerate(param_leaves):
        payload[f"param_{leaf_index}"] = np.asarray(leaf)

    metadata = _collect_checkpoint_metadata(model)
    payload["__metadata_json__"] = np.asarray(json.dumps(metadata), dtype=np.str_)

    np.savez(str(path), **payload)  # type: ignore[arg-type]
    return len(param_leaves)


def _read_checkpoint_metadata(ckpt: np.lib.npyio.NpzFile) -> dict[str, Any] | None:
    if "__metadata_json__" not in ckpt.files:
        return None

    raw = ckpt["__metadata_json__"]
    try:
        if getattr(raw, "ndim", 0) == 0:
            metadata_text = str(raw.item())
        else:
            metadata_text = str(raw.tolist())
        metadata = json.loads(metadata_text)
    except Exception as exc:
        raise ValueError("Checkpoint metadata is present but could not be parsed") from exc

    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a JSON object")

    return metadata


def load_checkpoint_into_model(
    model: NemotronNanoBlock | NemotronMultimodal,
    checkpoint_path: str,
    strict: bool = True,
) -> int:
    """
    Loads model parameter leaves from a legacy .npz checkpoint.

    Supported checkpoint formats:
    - Legacy v1: keys param_0..param_N only
    - v2+: param_0..param_N plus __metadata_json__ (versioned metadata)
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    param_state = nnx.state(model, nnx.Param)
    param_leaves = jax.tree_util.tree_leaves(param_state)
    tree_def = jax.tree_util.tree_structure(param_state)

    with np.load(path, allow_pickle=False) as ckpt:
        keys = list(ckpt.files)
        if not keys:
            raise ValueError(f"Checkpoint is empty: {checkpoint_path}")

        metadata = _read_checkpoint_metadata(ckpt)
        param_keys = [key for key in keys if key.startswith("param_")]
        if not param_keys:
            raise ValueError(
                "Checkpoint format mismatch: expected parameter keys like 'param_0', 'param_1', ..."
            )

        non_param_keys = [key for key in keys if not key.startswith("param_")]
        if non_param_keys and non_param_keys != ["__metadata_json__"]:
            raise ValueError(
                "Checkpoint format mismatch: unsupported non-parameter keys "
                f"{non_param_keys}"
            )

        try:
            ordered_keys = sorted(
                param_keys,
                key=lambda name: int(name.split("_", 1)[1]),
            )
        except ValueError as exc:
            raise ValueError(
                "Checkpoint keys must follow 'param_<index>' format with integer index"
            ) from exc

        ckpt_leaves = [jnp.asarray(ckpt[key]) for key in ordered_keys]

    if metadata is not None:
        version = metadata.get("format_version")
        if not isinstance(version, int):
            raise ValueError("Checkpoint metadata missing integer format_version")
        if version > CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                "Checkpoint format version is newer than this loader supports: "
                f"{version} > {CHECKPOINT_FORMAT_VERSION}"
            )

        meta_count = metadata.get("param_count")
        if isinstance(meta_count, int) and strict and meta_count != len(param_leaves):
            raise ValueError(
                "Checkpoint metadata param_count mismatch: "
                f"checkpoint={meta_count}, model={len(param_leaves)}"
            )
    else:
        print(
            "Checkpoint metadata not found (legacy format). "
            "Consider re-saving with --save-checkpoint-path for versioned metadata."
        )

    if strict and len(ckpt_leaves) != len(param_leaves):
        raise ValueError(
            "Checkpoint parameter count mismatch: "
            f"checkpoint has {len(ckpt_leaves)} leaves, "
            f"model expects {len(param_leaves)} leaves"
        )

    restored_leaves: list[jax.Array] = [jnp.asarray(leaf) for leaf in param_leaves]
    pair_count = min(len(param_leaves), len(ckpt_leaves))

    restored_count = 0
    for leaf_index in range(pair_count):
        target_arr = jnp.asarray(param_leaves[leaf_index])
        source_leaf = ckpt_leaves[leaf_index]

        if target_arr.shape != source_leaf.shape:
            if strict:
                raise ValueError(
                    "Checkpoint shape mismatch at leaf "
                    f"{leaf_index}: checkpoint {source_leaf.shape} vs model {target_arr.shape}"
                )
            continue

        if source_leaf.dtype != target_arr.dtype:
            source_leaf = source_leaf.astype(target_arr.dtype)

        restored_leaves[leaf_index] = source_leaf
        restored_count += 1

    restored_params = jax.tree_util.tree_unflatten(tree_def, restored_leaves)
    nnx.update(model, restored_params)
    return restored_count


# =============================================================================
# Main
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI arguments kept intentionally small and beginner-friendly."""
    parser = argparse.ArgumentParser(description="Minimal Nemotron train/eval/chat app")
    parser.add_argument("--preset", type=str, default="tiny", help="Nemotron preset")
    parser.add_argument(
        "--model-mode",
        type=str,
        default="text",
        choices=["text", "vlm", "vla"],
        help="Run mode: text-only Nemotron, vision-language (vlm), or vision-language-action (vla)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="Input image size for multimodal mode",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="Vision patch size for multimodal mode",
    )
    parser.add_argument(
        "--vision-dim",
        type=int,
        default=256,
        help="Intermediate vision embedding dimension before projection to d_model",
    )
    parser.add_argument(
        "--vision-in-channels",
        type=int,
        default=3,
        help="Input image channel count for multimodal mode",
    )
    parser.add_argument(
        "--vision-fusion",
        type=str,
        default="prepend",
        choices=["prepend", "append"],
        help="How vision tokens are fused with text tokens in multimodal mode",
    )
    parser.add_argument(
        "--action-vocab-size",
        type=int,
        default=256,
        help="Action vocabulary size used when --model-mode=vla",
    )
    parser.add_argument("--steps", type=int, default=80, help="Training steps")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Train/eval batch size"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length (must be divisible by Mamba chunk size)",
    )
    parser.add_argument(
        "--eval-batches", type=int, default=10, help="Validation batches"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 for greedy decoding, >0 for sampling",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=80, help="Max tokens per reply"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--tinystories-max-stories",
        type=int,
        default=5000,
        help="How many TinyStories stories to load for this run",
    )
    parser.add_argument(
        "--tinystories-train-ratio",
        type=float,
        default=0.9,
        help="Fraction of stories used for training (rest for validation)",
    )
    parser.add_argument(
        "--tinystories-split",
        type=str,
        default="train",
        help="Hugging Face split to read from TinyStories",
    )
    parser.add_argument(
        "--tinystories-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="tinystories",
        choices=["tinystories", "jsonl", "multimodal_jsonl"],
        help="Dataset source format: TinyStories streaming or local JSONL files",
    )
    parser.add_argument(
        "--train-jsonl",
        type=str,
        default=None,
        help="Path to train JSONL file (used when --dataset-format=jsonl)",
    )
    parser.add_argument(
        "--val-jsonl",
        type=str,
        default=None,
        help="Path to validation JSONL file (used when --dataset-format=jsonl)",
    )
    parser.add_argument(
        "--jsonl-text-key",
        type=str,
        default="serialized_text",
        help="JSON key to read text from in JSONL records",
    )
    parser.add_argument(
        "--jsonl-max-train-records",
        type=int,
        default=50000,
        help="Max train records to read from train JSONL",
    )
    parser.add_argument(
        "--jsonl-max-val-records",
        type=int,
        default=5000,
        help="Max validation records to read from val JSONL",
    )
    parser.add_argument(
        "--multimodal-text-key",
        type=str,
        default="serialized_text",
        help="JSON key for text in multimodal_jsonl records",
    )
    parser.add_argument(
        "--multimodal-image-key",
        type=str,
        default="image_path",
        help="JSON key for image path in multimodal_jsonl records",
    )
    parser.add_argument(
        "--multimodal-action-key",
        type=str,
        default="action_id",
        help="JSON key for action label in multimodal_jsonl records",
    )
    parser.add_argument(
        "--multimodal-image-root",
        type=str,
        default=None,
        help="Optional root directory used to resolve relative image paths in multimodal_jsonl",
    )
    parser.add_argument(
        "--validate-multimodal-jsonl",
        action="store_true",
        help="Run multimodal JSONL integrity checks (keys, image files, shapes) before training",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run validation checks and exit without training/evaluation",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="google/byt5-small",
        help="Hugging Face tokenizer name or local path",
    )
    parser.add_argument(
        "--tokenizer-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache directory for tokenizer files",
    )
    parser.add_argument(
        "--preview-first-story",
        action="store_true",
        help="Print a short preview of the first loaded TinyStories sample",
    )
    parser.add_argument(
        "--skip-chat",
        action="store_true",
        help="Run train+eval only (useful for non-interactive testing)",
    )
    parser.add_argument(
        "--chat-only",
        action="store_true",
        help="Skip train/eval and start chat after loading --checkpoint-path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to .npz checkpoint with model parameter leaves",
    )
    parser.add_argument(
        "--checkpoint-non-strict",
        action="store_true",
        help="Allow partial checkpoint loading by skipping mismatched parameter leaves",
    )
    parser.add_argument(
        "--save-checkpoint-path",
        type=str,
        default=None,
        help="Optional output path to save a versioned checkpoint after training",
    )
    parser.add_argument(
        "--assistant-only-loss",
        action="store_true",
        help=(
            "Mask loss to assistant content tokens only using role tags in text "
            "(recommended with --dataset-format=jsonl chat data)"
        ),
    )
    parser.add_argument(
        "--user-role-tag",
        type=str,
        default="<|user|>",
        help="Role tag used to mark user turns in serialized chat text",
    )
    parser.add_argument(
        "--assistant-role-tag",
        type=str,
        default="<|assistant|>",
        help="Role tag used to mark assistant turns in serialized chat text",
    )
    parser.add_argument(
        "--debug-mask-ratio",
        action="store_true",
        help="Print supervised-token mask ratio during training",
    )
    parser.add_argument(
        "--debug-mask-every",
        type=int,
        default=1,
        help="Print mask-debug line every N training steps",
    )
    parser.add_argument(
        "--chat-history-turns",
        type=int,
        default=6,
        help="How many previous user-assistant turns to include in chat context",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be > 0")
    if args.chat_history_turns <= 0:
        raise ValueError("--chat-history-turns must be > 0")
    if args.chat_only and not args.checkpoint_path:
        raise ValueError("--checkpoint-path is required when --chat-only is set")
    if args.chat_only and args.skip_chat:
        raise ValueError("--chat-only and --skip-chat cannot be used together")
    if args.vision_in_channels <= 0:
        raise ValueError("--vision-in-channels must be > 0")
    if args.model_mode == "text" and args.dataset_format == "multimodal_jsonl":
        raise ValueError(
            "--dataset-format=multimodal_jsonl requires --model-mode=vlm or vla"
        )
    if args.model_mode in {"vlm", "vla"} and args.dataset_format != "multimodal_jsonl":
        raise ValueError(
            "--model-mode=vlm/vla requires --dataset-format=multimodal_jsonl"
        )
    if args.validate_multimodal_jsonl and args.dataset_format != "multimodal_jsonl":
        raise ValueError(
            "--validate-multimodal-jsonl requires --dataset-format=multimodal_jsonl"
        )
    if args.validate_only and not args.validate_multimodal_jsonl:
        raise ValueError("--validate-only requires --validate-multimodal-jsonl")

    print("Initializing minimal Nemotron app...")

    if args.assistant_only_loss and args.dataset_format != "jsonl":
        print(
            "Note: --assistant-only-loss is most useful with role-tagged chat JSONL data."
        )

    # 1) Load Hugging Face tokenizer.
    tokenizer = load_hf_tokenizer(
        tokenizer_name=args.tokenizer_name,
        cache_dir=args.tokenizer_cache_dir,
    )

    # 2) Build config/model.
    base_config = NemotronConfig.from_preset(args.preset)
    # len(tokenizer) includes any runtime-added special tokens.
    base_config.vocab_size = len(tokenizer)

    if args.seq_len % base_config.mamba_chunk_size != 0:
        raise ValueError(
            f"seq_len must be divisible by mamba_chunk_size ({base_config.mamba_chunk_size})"
        )

    print(
        "Model setup: "
        f"mode={args.model_mode}, preset={args.preset}, tokenizer={args.tokenizer_name}, "
        f"vocab_size={base_config.vocab_size}, "
        f"seq_len={args.seq_len}, chunk_size={base_config.mamba_chunk_size}"
    )

    rngs = nnx.Rngs(args.seed)
    model: NemotronNanoBlock | NemotronMultimodal
    train_text_seq_len = args.seq_len
    task_type: str = "text"
    active_mm_config: NemotronMultimodalConfig | None = None

    if args.model_mode == "text":
        model = NemotronNanoBlock(rngs=rngs, config=base_config)
    else:
        mm_config = NemotronMultimodalConfig(
            **vars(base_config),
            use_vision=True,
            image_size=args.image_size,
            patch_size=args.patch_size,
            vision_in_channels=args.vision_in_channels,
            vision_dim=args.vision_dim,
            vision_fusion=args.vision_fusion,
            use_action_head=(args.model_mode == "vla"),
            action_vocab_size=args.action_vocab_size,
        )

        vision_tokens = mm_config.num_vision_tokens
        if vision_tokens >= args.seq_len:
            raise ValueError(
                "For multimodal mode, seq_len must be larger than vision token count: "
                f"seq_len={args.seq_len}, vision_tokens={vision_tokens}"
            )

        # Keep total fused sequence length equal to seq_len so Mamba chunk
        # constraints stay predictable.
        train_text_seq_len = args.seq_len - vision_tokens
        model = NemotronMultimodal(rngs=rngs, config=mm_config)
        active_mm_config = mm_config
        task_type = args.model_mode

        print(
            "Multimodal adapter enabled: "
            f"vision={mm_config.use_vision}, "
            f"fusion={mm_config.vision_fusion}, "
            f"action_head={mm_config.use_action_head}, "
            f"vision_tokens={vision_tokens}, "
            f"text_seq_len={train_text_seq_len}"
        )

    # Chat-only mode: load checkpoint and jump straight to interactive chat.
    if args.chat_only:
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required for chat-only mode")

        restored_count = load_checkpoint_into_model(
            model=model,
            checkpoint_path=args.checkpoint_path,
            strict=not args.checkpoint_non_strict,
        )
        print(
            f"Loaded checkpoint '{args.checkpoint_path}' "
            f"({restored_count} parameter leaves restored)."
        )

        chat_loop(
            model=model,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            rng_key=jax.random.PRNGKey(args.seed + 1),
            user_role_tag=args.user_role_tag,
            assistant_role_tag=args.assistant_role_tag,
            history_turns=args.chat_history_turns,
        )
        return

    # Optional warm-start for train/eval runs.
    if args.checkpoint_path:
        restored_count = load_checkpoint_into_model(
            model=model,
            checkpoint_path=args.checkpoint_path,
            strict=not args.checkpoint_non_strict,
        )
        print(
            f"Loaded checkpoint '{args.checkpoint_path}' "
            f"({restored_count} parameter leaves restored)."
        )

    # 3) Load data from selected dataset source.
    train_texts: list[str] = []
    val_texts: list[str] = []
    train_examples_mm: list[MultimodalExample] | None = None
    val_examples_mm: list[MultimodalExample] | None = None

    if args.dataset_format == "tinystories":
        all_stories = load_tinystories_texts(
            max_stories=args.tinystories_max_stories,
            split=args.tinystories_split,
            cache_dir=args.tinystories_cache_dir,
        )
        train_texts, val_texts = split_train_val_texts(
            stories=all_stories,
            train_ratio=args.tinystories_train_ratio,
        )

        print(
            "Dataset setup: "
            "source=tinystories, "
            f"total_stories={len(all_stories)}, "
            f"train_stories={len(train_texts)}, "
            f"val_stories={len(val_texts)}"
        )

        # Optional preview: helps beginners see real input text before tokenization.
        if args.preview_first_story:
            preview = all_stories[0].replace("\n", " ").strip()
            preview_limit = 240
            if len(preview) > preview_limit:
                preview = preview[:preview_limit] + "..."

            print("\nTinyStories preview (first loaded sample):")
            print(f"  {preview}")
    elif args.dataset_format == "jsonl":
        if not args.train_jsonl or not args.val_jsonl:
            raise ValueError(
                "--train-jsonl and --val-jsonl are required when --dataset-format=jsonl"
            )

        train_texts = load_jsonl_texts(
            jsonl_path=args.train_jsonl,
            max_records=args.jsonl_max_train_records,
            text_key=args.jsonl_text_key,
        )
        val_texts = load_jsonl_texts(
            jsonl_path=args.val_jsonl,
            max_records=args.jsonl_max_val_records,
            text_key=args.jsonl_text_key,
        )

        print(
            "Dataset setup: "
            "source=jsonl, "
            f"train_records={len(train_texts)}, "
            f"val_records={len(val_texts)}, "
            f"text_key={args.jsonl_text_key}"
        )

        if args.preview_first_story:
            preview = train_texts[0].replace("\n", " ").strip()
            preview_limit = 240
            if len(preview) > preview_limit:
                preview = preview[:preview_limit] + "..."

            print("\nJSONL preview (first loaded sample):")
            print(f"  {preview}")
    else:
        if not args.train_jsonl or not args.val_jsonl:
            raise ValueError(
                "--train-jsonl and --val-jsonl are required when --dataset-format=multimodal_jsonl"
            )
        if active_mm_config is None:
            raise RuntimeError("Internal error: multimodal config missing")

        if args.validate_multimodal_jsonl:
            train_stats = validate_multimodal_jsonl_integrity(
                jsonl_path=args.train_jsonl,
                image_size=active_mm_config.image_size,
                image_channels=active_mm_config.vision_in_channels,
                text_key=args.multimodal_text_key,
                image_key=args.multimodal_image_key,
                action_key=args.multimodal_action_key,
                image_root=args.multimodal_image_root,
                require_action=(task_type == "vla"),
            )
            val_stats = validate_multimodal_jsonl_integrity(
                jsonl_path=args.val_jsonl,
                image_size=active_mm_config.image_size,
                image_channels=active_mm_config.vision_in_channels,
                text_key=args.multimodal_text_key,
                image_key=args.multimodal_image_key,
                action_key=args.multimodal_action_key,
                image_root=args.multimodal_image_root,
                require_action=(task_type == "vla"),
            )
            print(
                "Multimodal JSONL validation passed: "
                f"train_valid={train_stats['valid_records']}, "
                f"val_valid={val_stats['valid_records']}"
            )
            if args.validate_only:
                print("Validation-only mode complete. Exiting before training.")
                return

        train_examples_mm = load_multimodal_jsonl_examples(
            jsonl_path=args.train_jsonl,
            max_records=args.jsonl_max_train_records,
            tokenizer=tokenizer,
            text_seq_len=train_text_seq_len,
            image_size=active_mm_config.image_size,
            image_channels=active_mm_config.vision_in_channels,
            text_key=args.multimodal_text_key,
            image_key=args.multimodal_image_key,
            action_key=args.multimodal_action_key,
            image_root=args.multimodal_image_root,
            assistant_only_loss=args.assistant_only_loss,
            user_role_tag=args.user_role_tag,
            assistant_role_tag=args.assistant_role_tag,
            require_action=(task_type == "vla"),
        )
        val_examples_mm = load_multimodal_jsonl_examples(
            jsonl_path=args.val_jsonl,
            max_records=args.jsonl_max_val_records,
            tokenizer=tokenizer,
            text_seq_len=train_text_seq_len,
            image_size=active_mm_config.image_size,
            image_channels=active_mm_config.vision_in_channels,
            text_key=args.multimodal_text_key,
            image_key=args.multimodal_image_key,
            action_key=args.multimodal_action_key,
            image_root=args.multimodal_image_root,
            assistant_only_loss=args.assistant_only_loss,
            user_role_tag=args.user_role_tag,
            assistant_role_tag=args.assistant_role_tag,
            require_action=(task_type == "vla"),
        )
        print(
            "Dataset setup: "
            "source=multimodal_jsonl, "
            f"train_records={len(train_examples_mm)}, "
            f"val_records={len(val_examples_mm)}, "
            f"text_key={args.multimodal_text_key}, "
            f"image_key={args.multimodal_image_key}, "
            f"action_key={args.multimodal_action_key}"
        )

    # Build optimizer: AdamW with warmup + cosine decay
    max_steps = max(args.steps, 1)
    warmup_steps = max(1, max_steps // 10)
    lr_schedule = create_lr_schedule(
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        peak_lr=3e-4,
    )
    gradient_transformation = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=0.1),
    )
    optimizer = nnx.Optimizer(model, gradient_transformation, wrt=nnx.Param)

    # Use a separate key stream for data/generation randomness.
    rng_key = jax.random.PRNGKey(args.seed + 1)

    train_tokens: jax.Array | None = None
    val_tokens: jax.Array | None = None
    train_loss_mask: jax.Array | None = None
    val_loss_mask: jax.Array | None = None

    if args.dataset_format != "multimodal_jsonl":
        # 4) Prepare token streams from train/validation text samples.
        train_tokens, val_tokens, train_loss_mask, val_loss_mask = prepare_datasets(
            tokenizer=tokenizer,
            train_texts=train_texts,
            val_texts=val_texts,
            seq_len=train_text_seq_len,
            assistant_only_loss=args.assistant_only_loss,
            user_role_tag=args.user_role_tag,
            assistant_role_tag=args.assistant_role_tag,
        )

    # 5) Train.
    if isinstance(model, NemotronMultimodal):
        if active_mm_config is None:
            raise RuntimeError("Internal error: multimodal config missing")
        if train_examples_mm is None:
            raise RuntimeError("Multimodal training examples are missing")

        rng_key = train_model_multimodal(
            model=model,
            optimizer=optimizer,
            train_examples=train_examples_mm,
            steps=args.steps,
            batch_size=args.batch_size,
            task_type=task_type,
            rng_key=rng_key,
        )
    else:
        if train_tokens is None or train_loss_mask is None:
            raise RuntimeError("Text training streams are missing")
        rng_key = train_model(
            model=model,
            optimizer=optimizer,
            train_tokens=train_tokens,
            train_loss_mask=train_loss_mask,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            rng_key=rng_key,
            debug_mask_ratio=args.debug_mask_ratio,
            debug_mask_every=args.debug_mask_every,
        )

    # 6) Evaluate.
    if isinstance(model, NemotronMultimodal):
        if active_mm_config is None:
            raise RuntimeError("Internal error: multimodal config missing")
        if val_examples_mm is None:
            raise RuntimeError("Multimodal validation examples are missing")

        mean_ce, perplexity, rng_key = evaluate_model_multimodal(
            model=model,
            val_examples=val_examples_mm,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            task_type=task_type,
            rng_key=rng_key,
        )
    else:
        if val_tokens is None or val_loss_mask is None:
            raise RuntimeError("Text validation streams are missing")
        mean_ce, perplexity, rng_key = evaluate_model(
            model=model,
            val_tokens=val_tokens,
            val_loss_mask=val_loss_mask,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_batches=args.eval_batches,
            rng_key=rng_key,
        )

    print("\nEvaluation:")
    print(f"  mean CE loss: {mean_ce:.4f}")
    print(f"  perplexity:   {perplexity:.4f}")

    if args.save_checkpoint_path:
        saved_count = save_checkpoint_from_model(
            model=model,
            checkpoint_path=args.save_checkpoint_path,
        )
        print(
            f"Saved checkpoint '{args.save_checkpoint_path}' "
            f"({saved_count} parameter leaves, format v{CHECKPOINT_FORMAT_VERSION})."
        )

    # 7) Chat.
    if not args.skip_chat:
        chat_loop(
            model=model,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            rng_key=rng_key,
            user_role_tag=args.user_role_tag,
            assistant_role_tag=args.assistant_role_tag,
            history_turns=args.chat_history_turns,
        )


if __name__ == "__main__":
    main()
