# Nemotron 3 – JAX Implementation (Nano & Super)

A **simple, minimalistic, and explainable** implementation of both **Nemotron 3 Nano** and **Nemotron 3 Super** in JAX/Flax NNX.

Both models are efficient hybrid Mamba-Transformer architectures with Mixture-of-Experts (MoE), designed for agentic reasoning. This codebase prioritizes **clarity and educational value** over performance optimization, making it ideal for understanding how modern hybrid architectures work.

- **Nemotron 3 Nano**: Original hybrid with Sparse MoE and standard next-token training.
- **Nemotron 3 Super**: Extends Nano with **Latent MoE** (experts computed in compressed space) and **Multi-Token Prediction** (joint training for multiple future tokens).

---

## 🎯 Project Goals

- **Explainability**: Every design choice is documented with clear comments.
- **Minimalism**: Unnecessary abstractions and optimizations are removed; only the essential concepts remain.
- **Reproducibility**: Small default dimensions allow full training on CPU/GPU without enterprise infrastructure.
- **Educational**: Serve as a reference for understanding Nemotron 3 Nano and hybrid LLM architectures.

---

## 🏗️ Architecture Overview

### Hybrid Stack Pattern

Both models alternate between two types of mixer blocks:

- **Mamba 2 Blocks**: State-space models (SSMs) with linear-time complexity
- **Grouped-Query Attention (GQA)**: Efficient causal self-attention with fewer KV heads

Each mixer is followed by a **MoE** layer (Sparse MoE for Nano, Latent MoE for Super).

### Key Components

#### 1. **Mamba 2 Blocks** (`mamba_2.py`)
- State-space model layer with selective scanning (SSD algorithm)
- Processes sequences efficiently in O(n) time
- Uses input-dependent gating and a D skip connection for selective computation
- Chunked SSD algorithm keeps memory usage bounded for long sequences

#### 2. **Grouped-Query Attention** (`attention.py`)
- Causal masking (decoder-only) for language modeling
- Multiple query heads but shared KV heads (reduces parameters & memory)
- No positional embeddings, dropout, or bias on projections

#### 3. **Sparse Mixture-of-Experts** (`moe.py`) — *Nano only*
- **Routed Experts**: Fine-grained expert specialization via granularity factors (DeepSeekMoE style)
- **Shared Experts**: Always-on experts for stable, universal computation
- **Sigmoid Gating**: Independent gate scores per expert (not softmax); top-k scores are renormalized
- **Squared-ReLU**: Stronger nonlinearity in expert FFNs
- **Bias-based Load Balancing**: Avoids auxiliary loss; expert biases are nudged with a simple sign rule after each step

#### 4. **Latent Mixture-of-Experts** (`latent_moe.py`) — *Super only*
- Extends SparseMoE by routing in a **compressed latent space** of size ℓ < d\_model (compression ratio α = d\_model / ℓ, typically 4)
- `down_proj` (d→ℓ) and `up_proj` (ℓ→d) wrap the routed expert FFNs; the router still operates in full d-space
- Allows α× more experts at the same FLOPs per token, enabling the scale-up from Nano's 4 experts to Super's 512
- Same sigmoid gating and bias-based load balancing interface as `SparseMoE`

#### 5. **Multi-Token Prediction** (`multi_token_prediction.py`) — *Super only*
- A single shared `MTPHead` is applied iteratively for each prediction depth
- Each head: RMSNorm → concat previous hidden state + next-token embedding → Linear → Mamba2Block → LM head
- Jointly trains the model to predict `num_mtp_heads` (default 2) tokens beyond the standard next-token target
- `mtp_loss(outputs, scale=0.3)` computes the weighted auxiliary loss; training batches must be `num_mtp_heads + 1` tokens longer than the main sequence length

#### 6. **Hybrid Model** (`nemotron.py`)
- **`NemotronNanoBlock`**: Configurable layer pattern via `patterns` list (e.g., `mamba_moe` and `mamba_attention_moe` blocks), pre-norm RMSNorm residuals, token embedding → N hybrid blocks → RMSNorm → LM head
- **`NemotronSuperBlock`**: Identical structure but uses `latent_mamba_moe` and `latent_mamba_attention_moe` blocks (with `LatentMoE`); exposes `forward_train(batch)` → `(main_logits, main_labels, mtp_outputs)` for joint next-token + MTP training

---

## 📦 Installation

### Requirements
- Python 3.9+
- JAX (`jax[cpu]` or `jax[cuda]`)
- Flax
- Optax (for optimization)
- Orbax (for checkpointing)
- Datasets (for streaming FineWeb-Edu)
- Transformers (for Hugging Face tokenizers)

### Setup

```bash
# Clone or navigate to the project
cd nugie-jax-nemotron-3-nano

# Create and activate virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install "jax[cpu]" flax optax orbax-checkpoint datasets transformers
```

---

## 🚀 Usage

### Jupyter Notebook and Google Colab

Use the ready notebook at `notebooks/pretrain_nemotron.ipynb`.

Local Jupyter:

```bash
jupyter notebook notebooks/pretrain_nemotron.ipynb
```

Google Colab:

1. Open Colab and upload `notebooks/pretrain_nemotron.ipynb`.
2. Run all cells from top to bottom.

### Pretraining on FineWeb-Edu

`pretrained.py` implements the full pretraining workflow for **Nemotron 3 Super**:

1. **Tokenization**: Loads the `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` tokenizer from Hugging Face.
2. **Dataset**: Streams text from `HuggingFaceFW/fineweb-edu` (no full download required). Each packed chunk is `SEQ_LEN + NUM_MTP_HEADS + 1` tokens wide to supply both the main next-token targets and the MTP auxiliary targets.
3. **Training**: Combined next-token + MTP auxiliary + MoE load-balancing losses with AdamW + warmup-stable-minus-sqrt (WSD-style) LR schedule.
4. **Checkpointing**: Saves model weights via Orbax every `CHECKPOINT_EVERY` steps; resumes from latest checkpoint automatically.
5. **Evaluation**: Reports validation loss and perplexity (main next-token loss only) after training.
6. **Interactive Chat**: Launches a terminal chat loop after training completes.

```bash
python pretrained.py
```

### Smoke Tests (One Command)

Run a quick automated sanity pass across project syntax, imports, model
construction, forward shapes, and a single optimizer/train-step update:

```bash
python smoke_test.py
```

If you want a faster check that skips the train-step/JIT compilation path:

```bash
python smoke_test.py --skip-train-step
```

Key hyperparameters are constants at the top of `pretrained.py`:

```python
VOCAB_SIZE       = 131072   # Nemotron tokenizer vocabulary size
SEQ_LEN          = 256      # Tokens per training sample (must be divisible by CHUNK_SIZE)
CHUNK_SIZE       = 64       # Mamba SSD chunk size
NUM_MTP_HEADS    = 2        # Extra prediction depths (must match NemotronSuperConfig.num_mtp_heads)
BATCH_SIZE       = 2
LEARNING_RATE    = 3e-4
CHECKPOINT_EVERY = 200      # Save a checkpoint every N steps
MAX_TRAIN_STEPS  = 10000
WARMUP_STEPS     = 1000     # Linear warmup for the first N steps
VAL_STEPS        = 50       # Batches averaged for validation
MAX_GEN_TOKENS   = 200      # Max new tokens per chat response
MAX_CTX_LEN      = 512      # Rolling context window during generation
```

### Checkpointing

Model weights are saved using Orbax in `checkpoints/`. The training loop automatically resumes from the latest checkpoint if one exists:

```
checkpoints/
└── <step>/        # Orbax checkpoint directory per step
```

---

## 📂 Project Structure

```
nugie-jax-nemotron-3-nano/
├── pretrained.py              # Pretraining loop, evaluation, and interactive chat (Super)
├── nemotron.py                # Model assembly: NemotronNanoBlock + NemotronSuperBlock
├── attention.py               # Grouped-Query Attention (GQA) implementation
├── mamba_2.py                 # Mamba 2 State-Space Model blocks (SSD algorithm)
├── moe.py                     # Sparse MoE — used by NemotronNanoBlock
├── latent_moe.py              # Latent MoE — used by NemotronSuperBlock
├── multi_token_prediction.py  # Multi-Token Prediction (MTP) heads and loss
├── notebooks/
│   └── pretrain_nemotron.ipynb  # Jupyter / Google Colab notebook
├── checkpoints/               # Orbax checkpoint directories (created at runtime)
├── LICENSE                    # Apache 2.0
└── README.md                  # This file
```

---

## 🔧 Configuration

### Nemotron 3 Nano — `NemotronConfig`

Three named presets are available through `NemotronConfig.from_preset()`:

| Preset | `d_model` | Layers | Notes |
|---|---|---|---|
| `tiny` *(default)* | 128 | 10 | Fits on any CPU; good for quick local tests |
| `kaggle` / `colab` | 256 | 13 | Medium size; fits a Kaggle/Colab GPU |
| `paper_close` | 2048 | 26 | Closest to the published Nemotron 3 Nano style |

```python
from nemotron import NemotronConfig, NemotronNanoBlock
from flax import nnx

config = NemotronConfig.from_preset("tiny")  # or "kaggle", "paper_close"
config.vocab_size = 131072

model = NemotronNanoBlock(rngs=nnx.Rngs(0), config=config)
```

Full list of `NemotronConfig` fields:

```python
NemotronConfig(
    vocab_size=1000,              # Vocabulary size (set from tokenizer)
    d_model=128,                  # Embedding / hidden dimension

    # Layer pattern: list of (block_type, repeats)
    # block_type ∈ {"mamba_moe", "mamba_attention_moe"}
    patterns=[("mamba_moe", 2), ("mamba_attention_moe", 1), ...],

    # Attention (GQA)
    num_attention_heads=4,        # Query heads
    num_kv_heads=1,               # KV heads (num_attention_heads % num_kv_heads == 0)
    attention_head_dim=32,        # num_attention_heads * attention_head_dim == d_model

    # Mamba-2 SSM
    mamba_d_state=64,             # SSM state dimension
    mamba_d_conv=4,               # Causal conv kernel width
    mamba_expand=2,               # Inner dim = mamba_expand * d_model
    mamba_headdim=64,             # Dimension per Mamba head
    mamba_ngroups=1,              # B/C groups (like GQA for Mamba)
    mamba_chunk_size=64,          # SSD chunk size (seq_len must be divisible)

    # Sparse MoE
    num_experts=4,                # Routed (base) expert count
    num_shared_experts=1,         # Always-on shared experts
    top_k=2,                      # Top-k routed experts per token
    expert_hidden_dim=256,        # Expert FFN hidden dimension
    granularity_factor=1,         # Splits each expert into finer sub-experts
    scale_top_k_with_granularity=True,  # Scale top_k by granularity_factor

    rms_norm_eps=1e-6,            # RMSNorm epsilon
)
```

`NemotronConfig.validate()` checks all shape constraints and raises an `AssertionError` with a descriptive message if any constraint is violated.

---

### Nemotron 3 Super — `NemotronSuperConfig`

Three named presets are available through `NemotronSuperConfig.from_preset()`:

| Preset | `d_model` | Layers | `latent_size` | `num_experts` | `top_k` | Notes |
|---|---|---|---|---|---|---|
| `tiny_super` *(default)* | 128 | 10 | 32 | 8 | 2 | Fits on any CPU |
| `kaggle_super` | 256 | 17 | 64 | 32 | 4 | Kaggle/Colab T4/P100 |
| `paper_super` | 4096 | 88 | 1024 | 512 | 22 | Paper-faithful (Table 1) |

```python
from nemotron import NemotronSuperConfig, NemotronSuperBlock
from flax import nnx

config = NemotronSuperConfig.from_preset("tiny_super")  # or "kaggle_super", "paper_super"
config.vocab_size = 131072

model = NemotronSuperBlock(rngs=nnx.Rngs(0), config=config)

# Inference
logits = model(input_ids)                              # (B, T, vocab_size)

# Training (returns main logits, main labels, and MTP outputs)
main_logits, main_labels, mtp_outputs = model.forward_train(batch)
```

Full list of `NemotronSuperConfig` fields (fields shared with `NemotronConfig` have the same meaning):

```python
NemotronSuperConfig(
    vocab_size=1000,
    d_model=128,

    # Layer pattern: list of (block_type, repeats)
    # block_type ∈ {"latent_mamba_moe", "latent_mamba_attention_moe"}
    patterns=[("latent_mamba_moe", 2), ("latent_mamba_attention_moe", 1), ...],

    # Attention (GQA) — same fields as NemotronConfig
    num_attention_heads=4,
    num_kv_heads=1,
    attention_head_dim=32,

    # Mamba-2 SSM — same fields as NemotronConfig
    mamba_d_state=64,
    mamba_d_conv=4,
    mamba_expand=2,
    mamba_headdim=64,
    mamba_ngroups=1,
    mamba_chunk_size=64,

    # Latent MoE  (replaces Sparse MoE; no granularity_factor)
    latent_size=32,               # Compressed dimension ℓ; α = d_model / latent_size
    num_experts=8,                # Routed experts (should be α× the Nano count)
    num_shared_experts=1,         # Always-on shared experts (operate in full d-space)
    top_k=2,                      # Active routed experts per token (α-scaled)
    expert_hidden_dim=256,        # Routed expert FFN intermediate dim
    shared_expert_hidden_dim=512, # Shared expert FFN intermediate dim

    # Multi-Token Prediction
    num_mtp_heads=2,              # Extra prediction depths (2 for Super)
    mtp_loss_scale=0.3,           # Auxiliary MTP loss weight

    rms_norm_eps=1e-6,
)
```

---

## 📚 References

This implementation is inspired by:

1. **Nemotron 3 Nano Paper**: "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning"  
   [arXiv:2512.20848](https://arxiv.org/abs/2512.20848)

2. **Nemotron 3 Super Paper**: "Nemotron 3 Super: An Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning"  
   [arXiv:2601.18089](https://arxiv.org/abs/2601.18089)

3. **Multi-Token Prediction (MTP)**: "DeepSeek-V3 Technical Report" — §2.5 Multi-Token Prediction  
   [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

4. **Mamba 2 / SSD**: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (Dao & Gu, 2024)  
   [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

5. **Mamba**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"  
   [arXiv:2312.08636](https://arxiv.org/abs/2312.08636)

6. **Attention Is All You Need**: "Attention Is All You Need"  
   [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

7. **MoE Designs**: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"  
   [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)

---

## 📝 License

Apache License 2.0 – See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

This is primarily an educational project. Feel free to:
- Open issues for bugs or clarifications
- Submit PRs with improvements or additional documentation
- Fork and adapt for your own experiments

---

## ⚠️ Status

**In Progress** – Core architecture is implemented and functional. Ongoing work includes:
- [ ] Performance benchmarking
- [ ] Longer sequence length testing
- [ ] Scaling to larger model sizes
- [ ] Advanced evaluation metrics
- [ ] Notebook update for Nemotron 3 Super workflow

---

## 💡 Tips for Experimentation

1. **Start small**: Use the `tiny_super` preset to verify correctness locally before scaling up
2. **Monitor loss**: Watch for training instability in early steps; warmup helps
3. **Ablations**: Swap `latent_mamba_attention_moe` blocks for `latent_mamba_moe` to measure attention's contribution
4. **Latent size**: Reduce `latent_size` (increase α) to use more experts at the same FLOPs; increase it to trade parameter efficiency for capacity
5. **MTP depth**: Reduce `num_mtp_heads` to 1 (and set `NUM_MTP_HEADS = 1` in `pretrained.py`) if GPU memory is tight
6. **Text generation**: Use the interactive chat after training to qualitatively assess learned patterns
7. **Chunk size**: `SEQ_LEN` and `MAX_CTX_LEN` must both be divisible by `CHUNK_SIZE`

---

**Questions or suggestions?** Refer to inline code comments for detailed explanations of each component.
