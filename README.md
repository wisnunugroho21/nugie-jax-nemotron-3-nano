# Nemotron 3 Nano – JAX Implementation

A **simple, minimalistic, and explainable** implementation of Nemotron 3 Nano in JAX/Flax NNX.

Nemotron 3 Nano is an efficient hybrid Mamba-Transformer model with Mixture-of-Experts (MoE), designed for agentic reasoning. This codebase prioritizes **clarity and educational value** over performance optimization, making it ideal for understanding how modern hybrid architectures work.

---

## 🎯 Project Goals

- **Explainability**: Every design choice is documented with clear comments.
- **Minimalism**: Unnecessary abstractions and optimizations are removed; only the essential concepts remain.
- **Reproducibility**: Small default dimensions allow full training on CPU/GPU without enterprise infrastructure.
- **Educational**: Serve as a reference for understanding Nemotron 3 Nano and hybrid LLM architectures.

---

## 🏗️ Architecture Overview

### Hybrid Stack Pattern

The model alternates between two types of mixer blocks:

- **Mamba 2 Blocks**: State-space models (SSMs) with linear-time complexity
- **Grouped-Query Attention (GQA)**: Efficient causal self-attention with fewer KV heads

Each mixer is followed by a **Sparse Mixture-of-Experts (MoE)** layer.

### Key Components

#### 1. **Mamba 2 Blocks** (`mamba_2.py`)
- State-space model layer with selective scanning
- Processes sequences efficiently in O(n) time
- Uses input-dependent gating for selective computation

#### 2. **Grouped-Query Attention** (`attention.py`)
- Causal masking (decoder-only) for language modeling
- Rotary Position Embeddings (RoPE) for relative position awareness
- Multiple query heads but shared KV heads (reduces parameters & memory)

#### 3. **Sparse Mixture-of-Experts** (`moe.py`)
- **Routed Experts**: Fine-grained expert specialization via granularity factors
- **Shared Experts**: Always-on experts for stable computation
- **Sigmoid Gating**: Independent gate scores per expert (not softmax)
- **Squared-ReLU**: Stronger nonlinearity in expert FFNs
- **Bias-based Load Balancing**: Avoids auxiliary loss engineering

#### 4. **Hybrid Model** (`nemotron.py`)
- Configurable layer pattern (e.g., Mamba→MoE→Attention→MoE)
- Pre-norm RMSNorm residual connections
- Token embedding + rotating position embeddings

---

## 📦 Installation

### Requirements
- Python 3.9+
- JAX/Jax-cpu or Jax-gpu
- Flax
- Optax (for optimization)
- Datasets (for TinyStories dataset)
- Transformers (for Hugging Face tokenizers)

### Setup

```bash
# Clone or navigate to the project
cd nugie-jax-nemotron

# Create and activate virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # If available
# Or manually:
pip install jax flax optax datasets transformers
```

---

## 🚀 Usage

### Training & Evaluation

Run the full pipeline (load data → train → evaluate → chat):

```bash
python app.py \
   --steps 80 \
   --batch-size 8 \
   --seq-len 64 \
   --tokenizer-name google/byt5-small
```

**What happens in `app.py`:**

1. **Data Loading**: Loads TinyStories dataset
2. **Tokenization**: Uses a Hugging Face tokenizer (default: `google/byt5-small`)
3. **Model Training**: Trains the Nemotron model with validation loss tracking
4. **Evaluation**: Computes validation perplexity
5. **Interactive Chat**: Generates text from prompts in the terminal

### Checkpointing

Trained model weights are saved as `.npz` files in `checkpoints/`:

```
checkpoints/
├── step_3.npz        # Model weights after N steps
└── ...
```

### Hugging Face Tokenizer

The project now uses a tokenizer loaded from Hugging Face (`AutoTokenizer` in `app.py`).

- Default tokenizer: `google/byt5-small`
- Override with: `--tokenizer-name <model-or-path>`
- Optional cache: `--tokenizer-cache-dir <path>`

Special token behavior is normalized in code so batching and generation always
have PAD/BOS/EOS IDs available.

---

## 📂 Project Structure

```
nugie-jax-nemotron/
├── app.py              # Training loop, evaluation, interactive chat
├── nemotron.py         # Main model architecture (config + hybrid layers)
├── attention.py        # Grouped-Query Attention (GQA) implementation
├── mamba_2.py          # Mamba 2 State-Space Model blocks
├── moe.py              # Sparse Mixture-of-Experts implementation
├── checkpoints/        # Saved model weights (.npz files)
├── data/               # Training datasets (TinyStories)
├── LICENSE             # Apache 2.0
└── README.md           # This file
```

---

## 🔧 Configuration

Edit hyperparameters in `app.py` or pass via CLI arguments:

```python
# Key config variables (from nemotron.NemotronConfig)
config = NemotronConfig(
   vocab_size=...,              # Set from len(hf_tokenizer)
    max_seq_len=256,             # Maximum sequence length
    d_model=128,                 # Embedding dimension
    n_layers=7,                  # Number of hybrid blocks
    layer_pattern=[...],         # Mamba/Attention/MoE scheduling
    mamba_d_state=16,            # Mamba state size
    num_heads=4,                 # Attention heads
    num_kv_heads=2,              # KV heads (grouped query)
    moe_num_experts=4,           # Number of routed experts
    moe_top_k=2,                 # Select top-2 experts
    moe_expert_hidden_dim=256,   # Expert FFN hidden size
)
```

---

## 📚 References

This implementation is inspired by:

1. **Nemotron 3 Nano Paper**: "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning"  
   [arXiv:2512.20848](https://arxiv.org/abs/2512.20848)

2. **Mamba**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"  
   [arXiv:2312.08636](https://arxiv.org/abs/2312.08636)

3. **Attention Is All You Need**: "Attention Is All You Need"  
   [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

4. **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"  
   [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

5. **MoE Designs**: DeepSeekMoE, Mixture of Experts (GShard, Switch Transformers)

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

---

## 💡 Tips for Experimentation

1. **Start small**: Use default tiny dimensions to verify correctness locally
2. **Monitor loss**: Watch for training instability in early epochs
3. **Ablations**: Try disabling MoE or Mamba layers to understand their contribution
4. **Text generation**: Use the interactive chat to qualitatively assess learned patterns
5. **Checkpoint**: Save model weights frequently during training

---

**Questions or suggestions?** Refer to inline code comments for detailed explanations of each component.
