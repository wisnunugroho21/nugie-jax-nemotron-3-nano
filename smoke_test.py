"""
Project smoke tests for quick sanity checks.

Run:
    python smoke_test.py

Optional:
    python smoke_test.py --skip-train-step
"""

from __future__ import annotations

import argparse
import importlib
import py_compile
import traceback


PROJECT_FILES = [
    "attention.py",
    "mamba_2.py",
    "latent_moe.py",
    "multi_token_prediction.py",
    "moe.py",
    "nemotron.py",
    "pretrained.py",
]

PROJECT_MODULES = [
    "attention",
    "mamba_2",
    "latent_moe",
    "multi_token_prediction",
    "moe",
    "nemotron",
    "pretrained",
]


def _run_step(name: str, fn) -> None:
    print(f"[smoke] {name} ...")
    fn()
    print(f"[smoke] {name}: ok")


def _check_python_compile() -> None:
    for file_path in PROJECT_FILES:
        py_compile.compile(file_path, doraise=True)


def _check_imports() -> None:
    for module_name in PROJECT_MODULES:
        importlib.import_module(module_name)


def _check_model_shapes() -> None:
    import jax.numpy as jnp
    from flax import nnx

    from nemotron import (
        NemotronConfig,
        NemotronNanoBlock,
        NemotronSuperConfig,
        NemotronSuperBlock,
    )

    for preset in ("tiny", "kaggle", "paper_close"):
        NemotronConfig.from_preset(preset).validate()

    for preset in ("tiny_super", "kaggle_super", "paper_super"):
        NemotronSuperConfig.from_preset(preset).validate()

    nano_cfg = NemotronConfig.from_preset("tiny")
    nano = NemotronNanoBlock(rngs=nnx.Rngs(0), config=nano_cfg)

    super_cfg = NemotronSuperConfig.from_preset("tiny_super")
    super_model = NemotronSuperBlock(rngs=nnx.Rngs(1), config=super_cfg)

    tokens = jnp.arange(0, 128, dtype=jnp.int32).reshape(2, 64)
    nano_logits = nano(tokens)
    super_logits = super_model(tokens)

    assert nano_logits.shape == (2, 64, nano_cfg.vocab_size)
    assert super_logits.shape == (2, 64, super_cfg.vocab_size)

    extended = (
        jnp.arange(
            0,
            2 * (64 + super_cfg.num_mtp_heads + 1),
            dtype=jnp.int32,
        ).reshape(2, 64 + super_cfg.num_mtp_heads + 1)
        % super_cfg.vocab_size
    )
    main_logits, main_labels, mtp_outputs = super_model.forward_train(extended)

    assert main_logits.shape == (2, 64, super_cfg.vocab_size)
    assert main_labels.shape == (2, 64)
    assert len(mtp_outputs) == super_cfg.num_mtp_heads


def _check_train_step() -> None:
    import jax.numpy as jnp
    from flax import nnx

    from pretrained import (
        build_model,
        collect_moe_layers,
        make_gradient_transform_optimizer,
        train_step,
        update_moe_biases,
    )

    model = build_model(seed=0)
    optimizer = nnx.Optimizer(
        model,
        make_gradient_transform_optimizer(
            max_steps=20,
            warmup_steps=5,
            peak_lr=3e-4,
        ),
        wrt=nnx.Param,
    )

    seq_len = 64
    num_mtp = model.config.num_mtp_heads
    batch = (
        jnp.arange(0, 2 * (seq_len + num_mtp + 1), dtype=jnp.int32)
        .reshape(2, seq_len + num_mtp + 1)
        % model.config.vocab_size
    )

    loss = train_step(model, optimizer, batch)
    update_moe_biases(collect_moe_layers(model))

    assert bool(jnp.isfinite(loss))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run project smoke tests.")
    parser.add_argument(
        "--skip-train-step",
        action="store_true",
        help="Skip the optimizer/train-step smoke check.",
    )
    args = parser.parse_args()

    checks = [
        ("py_compile", _check_python_compile),
        ("imports", _check_imports),
        ("model_shapes", _check_model_shapes),
    ]
    if not args.skip_train_step:
        checks.append(("train_step", _check_train_step))

    try:
        for name, fn in checks:
            _run_step(name, fn)
    except Exception:
        print("[smoke] FAILED")
        traceback.print_exc()
        return 1

    print("[smoke] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
