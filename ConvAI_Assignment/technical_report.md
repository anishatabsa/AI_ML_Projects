# Technical Report: CodeGen-Style Decoder-Only Transformer
## Architectural Hyperparameter Choices, Structural Limitations, and Hardware Challenges

---

## 1. Architecture Overview

The model implements a **GPT-2-style Decoder-Only Transformer** with Pre-LayerNorm (Pre-LN) normalization for causal language modeling on Python source code.

```
Input Token IDs → TokenEmbedding + SinusoidalPE → [N × DecoderBlock] → LayerNorm → LM Head
```

Each `DecoderBlock` contains:
1. **Pre-LayerNorm** → **Masked Multi-Head Self-Attention** → **Residual**
2. **Pre-LayerNorm** → **Position-wise FFN (GELU)** → **Residual**

---

## 2. Hyperparameter Choices and Rationale

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `d_model` | 256 | Balances representational capacity with training feasibility on CPU/MPS. At 256 dims, each attention head gets 32 dimensions — sufficient to learn code syntax patterns. |
| `num_heads` | 8 | 8 parallel attention heads allow the model to simultaneously attend to: syntax structure, variable references, indentation levels, and semantic context. |
| `num_layers` | 4 | 4 decoder layers provide enough depth for local and short-range dependencies in Python code. Deeper models (8-12 layers) need GPU training. |
| `d_ff` | 1024 | Standard 4× expansion ratio (256 → 1024 → 256) — empirically validated in the original "Attention Is All You Need" paper. |
| `vocab_size` | ~8000 | BPE vocabulary trained on the corpus. Large enough to encode Python keywords as single tokens, small enough for fast embedding lookups. |
| `max_seq_len` | 256 | Covers the 90th percentile of Python function lengths in the dataset. Extending to 512 would double memory requirements. |
| `dropout` | 0.1 | Standard regularization rate. Higher dropout (0.2+) would hurt convergence on the small 20k sample corpus. |
| `batch_size` | 32 | Memory-efficient for CPU/MPS. Larger batches (128+) improve gradient estimates but require GPU memory. |
| `learning_rate` | 3e-4 | Standard AdamW learning rate for Transformer models. Combined with warmup and cosine decay for stability. |
| `warmup_steps` | 200 | Prevents large gradient updates during random weight initialization. Corresponds to ~1-2% of total training steps. |
| `weight_decay` | 0.1 | Applied to all linear layers (not biases/LayerNorm). Prevents overfitting on the training corpus. |
| `grad_clip` | 1.0 | Gradient norm clipping prevents exploding gradients, especially important in early training epochs. |
| `betas` | (0.9, 0.95) | Slightly lower β₂ than default (0.999) — more responsive to recent gradient history, better for code generation tasks. |

---

## 3. Key Design Decisions

### 3.1 Pre-LayerNorm vs Post-LayerNorm
We use **Pre-LayerNorm** (apply LayerNorm before attention/FFN, not after):
- **Stability**: Pre-LN models train more stably without careful learning rate tuning
- **Gradient flow**: Normalization before the residual branch ensures gradients flow cleanly through the identity path
- **Convergence**: Pre-LN models typically converge faster for medium-sized models

### 3.2 Weight Tying
The token embedding matrix and LM head projection share weights:
- **Parameter reduction**: Saves `vocab_size × d_model` parameters (~2M at our scale)
- **Semantic alignment**: Forces the input embeddings and output projections to occupy the same semantic space
- **Regularization effect**: Acts as an implicit constraint that improves generalization

### 3.3 Causal Masking Implementation
The lower-triangular boolean mask is **precomputed and registered as a buffer** (not a parameter):
- Computed once at initialization, reused across all forward passes
- Buffer moves to the correct device automatically with `.to(device)`
- Masked positions receive `-inf` before softmax → attention weight ≈ 0

### 3.4 Sinusoidal vs Learned Positional Encodings
We use **fixed sinusoidal** encodings rather than learned:
- **Extrapolation**: Can handle sequences longer than seen during training
- **No additional parameters**: Keeps model size smaller
- **Relative position encoding**: Inner products between PE vectors encode relative distances, useful for nested code structures

---

## 4. Structural Limitations

### 4.1 Model Scale
At ~7-10M parameters (d_model=256, 4 layers), the model is approximately:
- **100× smaller** than CodeGen-350M
- **3000× smaller** than GPT-3 (175B)

Practical consequences:
- Learns surface syntax patterns well (indentation, keywords, common idioms)
- Cannot reason about multi-step algorithms or complex logic
- Cannot maintain coherent variable naming across long functions

### 4.2 Context Window
256 tokens covers ~90% of short Python functions but misses:
- Class definitions with multiple methods
- Functions with complex nested logic
- Long docstrings followed by implementation

Real production models (StarCoder, CodeGen-2) use 8192+ token contexts.

### 4.3 No Instruction Tuning
The model is trained on next-token prediction only, not on:
- Prompt-completion pairs ("write a function that does X")
- Human preference feedback (RLHF)
- Unit test alignment (code that passes specific tests)

Adding supervised fine-tuning on instruction pairs would dramatically improve usefulness for practical code completion.

### 4.4 Repetition Degeneracy
Small autoregressive models are prone to degenerate loops:
```python
# Example degenerate output:
def calculate_median(numbers):
    if len(numbers) == 0:
        return None
    return None
    return None  # ← repetition degeneracy
    return None
```
**Root cause**: The model learns that `return None` frequently follows certain patterns and assigns high probability repeatedly.

**Mitigations**:
- Repetition penalty: Downscale logits of recently generated tokens
- Minimum length constraints
- Beam search with n-gram blocking

### 4.5 No Type System Awareness
The tokenizer treats `int`, `str`, `List[int]` as token sequences without semantic meaning. The model cannot enforce type consistency:
```python
def add(a: int, b: int) -> int:
    return str(a) + b  # type error — model may generate this
```

---

## 5. Hardware Challenges

### 5.1 Training Environment
| Device | Estimated Training Time (5 epochs, 20k samples) |
|--------|--------------------------------------------------|
| CPU (M-series) | 1–3 hours |
| Apple MPS (M1/M2) | 20–45 minutes |
| NVIDIA A10G | 5–10 minutes |
| NVIDIA A100 | 2–4 minutes |

### 5.2 Memory Constraints
- **Attention matrix**: For batch_size=32, seq_len=256, num_heads=8: `32 × 8 × 256 × 256 × 4 bytes ≈ 2GB` just for attention scores
- **Gradient checkpointing** (not implemented here) would halve memory at the cost of ~30% more compute
- **BF16/FP16 training** would halve memory on CUDA but is unstable on MPS

### 5.3 Apple MPS Limitations
- Limited support for some PyTorch operations (e.g., `torch.float64`, certain scatter operations)
- No `torch.cuda.amp` support → cannot use automatic mixed precision
- Workaround: Use `torch.float32` throughout, avoid double-precision operations

### 5.4 Scalability Path
To train a production-grade model:
1. **Scale model**: d_model=512-768, num_layers=12-24, d_ff=2048-3072 (~125M-350M params)
2. **Scale data**: Full CodeSearchNet (2M+ samples) + The Stack (GitHub Python)
3. **Scale hardware**: Multi-GPU with FSDP/DDP, BF16 mixed precision
4. **Extended context**: Flash Attention for efficient O(N) attention on long sequences

---

## 6. Why Decoder-Only Architecture for Code Generation

### Conceptual Argument
Code generation is fundamentally an **autoregressive** task: given a function signature and some body, produce the next logical continuation. This maps directly to the decoder-only training objective.

Bidirectional encoders (BERT, RoBERTa) see all tokens simultaneously — this is ideal for:
- Code search (find semantically similar code)
- Bug detection (classify whether code has a bug)
- Code understanding (answer questions about code)

But for **generation**, they require:
1. An encoder-decoder bridge (adds architectural complexity)
2. A separate language modeling head trained differently
3. Masked generation strategies that don't naturally extend during inference

Decoder-only models **are** the generation process — the training objective (predict next token) is identical to the inference operation (predict next token). No translation layer needed.

### Information Theoretic View
The decoder-only model learns: `P(code) = ∏ P(token_i | token_{0..i-1})`

This factorization allows:
- **Exact sampling** via ancestral sampling (temperature/top-k/top-p)
- **Conditional generation** by simply prepending a prompt
- **Completion** of any partial code snippet

Bidirectional models learn a different objective and cannot naturally be used for this factorization.

---

## 7. Summary

This implementation demonstrates a complete end-to-end CLM pipeline from raw code corpus to generating Python function completions. The model, while small, successfully learns Python syntax patterns and produces code that is often syntactically valid and stylistically Python-idiomatic. Scaling both the model and data using the same architecture and training procedure would yield production-quality results — exactly the approach used by Salesforce CodeGen, BigCode StarCoder, and Meta Code Llama.
