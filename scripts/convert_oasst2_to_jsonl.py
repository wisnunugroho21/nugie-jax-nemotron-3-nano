"""
Convert OpenAssistant OASST2 into role-formatted JSONL files.

Output schema per line:
{
  "id": str,
  "source": "OpenAssistant/oasst2",
  "split": "train" | "val",
  "lang": str,
  "turns": [{"role": "user"|"assistant", "text": str}, ...],
  "num_turns": int,
  "serialized_text": str,
  "stats": {
    "char_len": int,
    "token_estimate": int,
    "has_system": bool
  }
}

Design goals:
- Keep conversion explicit and easy to inspect.
- Build root-to-leaf conversation paths from message trees.
- Produce deterministic train/val split without extra dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ALLOWED_ROLES = {ROLE_USER, ROLE_ASSISTANT}


def _normalize_text(text: str) -> str:
    """Trim and collapse excessive blank lines for stable serialization."""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _serialize_turns(turns: list[dict[str, str]]) -> str:
    """Serialize turns into role-tagged plain text for LM training."""
    chunks: list[str] = []
    for turn in turns:
        role = turn["role"]
        text = turn["text"]
        chunks.append(f"<|{role}|>\n{text}\n")
    return "".join(chunks)


def _stable_hash_fraction(text: str) -> float:
    """Map a string deterministically to [0, 1) for split assignment."""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    # Divide by 2**32 so the range is [0, 1) — 0xFFFFFFFF+1 ensures the
    # maximum 32-bit value never maps to exactly 1.0.
    return value / 0x100000000


def _get_field(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return first existing key from a row, else None."""
    for key in keys:
        if key in row:
            return row[key]
    return None


def _map_role(raw_role: Any) -> str | None:
    """Map dataset-specific role labels to user/assistant."""
    if not isinstance(raw_role, str):
        return None

    role = raw_role.strip().lower()

    if role in {"prompter", "user", "human"}:
        return ROLE_USER
    if role in {"assistant", "gpt", "model"}:
        return ROLE_ASSISTANT

    return None


def _iter_paths_from_roots(
    root_ids: list[str],
    children_by_parent: dict[str, list[str]],
) -> list[list[str]]:
    """Enumerate all root-to-leaf message paths."""
    all_paths: list[list[str]] = []

    def dfs(node_id: str, path: list[str]) -> None:
        children = children_by_parent.get(node_id, [])
        if not children:
            all_paths.append(path.copy())
            return
        for child_id in children:
            path.append(child_id)
            dfs(child_id, path)
            path.pop()

    for root_id in root_ids:
        dfs(root_id, [root_id])

    return all_paths


def _passes_quality_filters(
    turns: list[dict[str, str]],
    min_turns: int,
    max_turns: int,
    min_chars: int,
    max_chars: int,
) -> bool:
    """Simple quality gates to keep low-noise conversations."""
    if not (min_turns <= len(turns) <= max_turns):
        return False

    serialized = _serialize_turns(turns)
    char_len = len(serialized)
    if not (min_chars <= char_len <= max_chars):
        return False

    return True


def convert_oasst2(
    out_dir: Path,
    lang: str,
    val_ratio: float,
    min_turns: int,
    max_turns: int,
    min_chars: int,
    max_chars: int,
    max_samples_train: int,
    max_samples_val: int,
) -> None:
    """Convert OASST2 trees into JSONL train/val files."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "This converter requires the 'datasets' package. Install with: pip install datasets"
        ) from exc

    dataset = load_dataset("OpenAssistant/oasst2", split="train")

    node_by_id: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    root_ids: list[str] = []

    for row in dataset:
        if not isinstance(row, dict):
            continue

        row_lang = _get_field(row, ("lang", "language"))
        if isinstance(row_lang, str) and row_lang.lower() != lang.lower():
            continue

        node_id_raw = _get_field(row, ("message_id", "id"))
        parent_id_raw = _get_field(row, ("parent_id", "parent_message_id"))

        if node_id_raw is None:
            continue

        node_id = str(node_id_raw)
        parent_id = str(parent_id_raw) if parent_id_raw is not None else None

        node_by_id[node_id] = row
        if parent_id is None or parent_id == "None":
            root_ids.append(node_id)
        else:
            children_by_parent[parent_id].append(node_id)

    for parent_id in list(children_by_parent.keys()):
        children_by_parent[parent_id].sort()
    root_ids = sorted(set(root_ids))

    paths = _iter_paths_from_roots(root_ids, children_by_parent)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    num_train = 0
    num_val = 0
    skipped = 0

    with train_path.open("w", encoding="utf-8") as train_file, val_path.open(
        "w", encoding="utf-8"
    ) as val_file:
        for path_idx, path_node_ids in enumerate(paths):
            turns: list[dict[str, str]] = []

            for node_id in path_node_ids:
                row = node_by_id.get(node_id)
                if row is None:
                    continue

                mapped_role = _map_role(_get_field(row, ("role", "speaker")))
                if mapped_role not in ALLOWED_ROLES:
                    continue

                text_raw = _get_field(row, ("text", "message", "content"))
                if not isinstance(text_raw, str):
                    continue

                text = _normalize_text(text_raw)
                if not text:
                    continue

                if turns and turns[-1]["role"] == mapped_role:
                    # Keep alternating role flow by merging same-role fragments.
                    turns[-1]["text"] = f"{turns[-1]['text']}\n\n{text}"
                else:
                    turns.append({"role": mapped_role, "text": text})

            if not turns:
                skipped += 1
                continue

            # Ensure conversations start with user and end with assistant.
            while turns and turns[0]["role"] != ROLE_USER:
                turns.pop(0)
            while turns and turns[-1]["role"] != ROLE_ASSISTANT:
                turns.pop()

            if not turns:
                skipped += 1
                continue

            if not _passes_quality_filters(
                turns=turns,
                min_turns=min_turns,
                max_turns=max_turns,
                min_chars=min_chars,
                max_chars=max_chars,
            ):
                skipped += 1
                continue

            serialized_text = _serialize_turns(turns)
            char_len = len(serialized_text)

            base_id = f"oasst2_path_{path_idx:08d}"
            split_fraction = _stable_hash_fraction(base_id)
            split = "val" if split_fraction < val_ratio else "train"

            if split == "train" and num_train >= max_samples_train:
                continue
            if split == "val" and num_val >= max_samples_val:
                continue

            record = {
                "id": base_id,
                "source": "OpenAssistant/oasst2",
                "split": split,
                "lang": lang,
                "turns": turns,
                "num_turns": len(turns),
                "serialized_text": serialized_text,
                "stats": {
                    "char_len": char_len,
                    "token_estimate": max(1, char_len // 4),
                    "has_system": False,
                },
            }

            target_file = val_file if split == "val" else train_file
            target_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if split == "val":
                num_val += 1
            else:
                num_train += 1

    print("OASST2 conversion complete")
    print(f"  train records: {num_train}")
    print(f"  val records:   {num_val}")
    print(f"  skipped paths: {skipped}")
    print(f"  output dir:    {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert OASST2 to JSONL chat corpus")
    parser.add_argument("--out-dir", type=str, default="data/oasst2")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--min-chars", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--max-samples-train", type=int, default=200000)
    parser.add_argument("--max-samples-val", type=int, default=10000)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1")

    convert_oasst2(
        out_dir=Path(args.out_dir),
        lang=args.lang,
        val_ratio=args.val_ratio,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_samples_train=args.max_samples_train,
        max_samples_val=args.max_samples_val,
    )


if __name__ == "__main__":
    main()
