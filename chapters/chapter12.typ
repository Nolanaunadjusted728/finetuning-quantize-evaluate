#import "../template.typ": *

#let Chapter12() = [
== 12 - Unsloth Framework

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-18-unsloth/unsloth_practical.ipynb")[`unsloth_practical.ipynb`]
  *Benchmark vs HF:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/tree/main/LLM%20Fine-Tuning-unsloth-vs-hf")[`LLM Fine-Tuning-unsloth-vs-hf/`]
]

=== 12.1 What Is Unsloth

Unsloth is a high-performance fine-tuning framework that claims *2–3× faster training* and *50–80% less GPU memory* compared to standard Hugging Face training, with *no accuracy loss* (exact math, no approximation).

=== 12.2 How Unsloth Achieves Performance Gains

#figure(
  table(
    columns: 2,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Optimization*], [*Description*]),
    [*Custom CUDA & Triton kernels*], [Replaces standard PyTorch kernels with optimized GPU code. CUDA is Nvidia's parallel computing platform and API; Triton is OpenAI's open GPU kernel language],
    [*Fused attention + MLP operations*], [Merges the two main transformer blocks (attention and feed-forward network) into a single operation],
    [*Optimized forward/backward propagation*], [Optimizations at the loss function and optimizer level],
    [*Smart gradient checkpointing*], [More efficient checkpoint storage in memory],
    [*Flash Attention compatibility*], [IO-aware exact attention - efficiently loads attention operations into memory],
    [*Manual backpropagation engine*], [Does NOT use PyTorch autograd's directed acyclic graph; uses custom backpropagation logic],
    [*Automatic sequence packing*], [Combines multiple short sequences into a single batch entry to avoid wasted padding],
  ),
  caption: [Unsloth Performance Optimizations],
  kind: table,
)

=== 12.3 Long Context Training

Unsloth's most impressive feature - handles up to *300K token* training sequences:

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*GPU VRAM*], [*Max Context (Unsloth)*], [*Max Context (Standard HF)*]),
    [8 GB], [~3,000 tokens], [OOM error],
    [12 GB], [~21,000 tokens], [OOM error],
    [16 GB], [~40,000 tokens], [OOM error],
    [24 GB], [~78,000 tokens], [OOM error],
    [80 GB], [~340,000 tokens], [~28,000 tokens],
  ),
  caption: [Unsloth Long Context Support],
  kind: table,
)

=== 12.4 Practical Implementation

The following code demonstrates Unsloth's optimized workflow: loading a 4-bit quantized model, applying LoRA adapters with Unsloth's custom kernels, and training with TRL's `SFTTrainer`. This entire pipeline runs on a free Google Colab T4 GPU.

#figure(
```python
from unsloth import FastLanguageModel

# Load model with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama-bnb-4bit",
    max_seq_length=4096,
    dtype=None,          # Auto-detect
    load_in_4bit=True    # 4-bit quantization
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

# Training uses SFTTrainer from TRL
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir="outputs",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
    ),
)
trainer.train()
```
, caption: [Unsloth FastLanguageModel setup with LoRA and SFTTrainer for optimized training],
)

#pagebreak(weak: true)

*Configuration best practices:*

- Set random seeds for reproducibility: `torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`, `random.seed(42)`, `np.random.seed(42)`
- Enable TF32 for faster matrix multiplication: `torch.backends.cuda.matmul.allow_tf32 = True`
- Set high float precision: `torch.set_float32_matmul_precision("high")`

=== 12.5 Worked Example: Legal Contract Analysis with Long Context

*Scenario:* A law firm wants to fine-tune a model to analyze 20-50 page contracts - identifying risk clauses, summarizing obligations, and flagging non-standard terms. Standard Hugging Face training on a free Colab T4 (16GB VRAM) can only handle ~1,024 tokens per sample. Unsloth enables ~40,000 tokens - enough for full contracts.

*Training data (Alpaca format with long-context input):*

#figure(
```json
[
  {
    "instruction": "Identify all indemnification clauses and assess risk level for the buyer.",
    "input": "[Full 15-page SaaS agreement text - ~12,000 tokens covering service terms, SLAs, data processing, indemnification, limitation of liability, termination...]",
    "output": "## Indemnification Analysis\n\n### Clause 1 (Section 8.1 - Vendor Indemnification)\n**Risk: LOW**\nVendor indemnifies Buyer against third-party IP infringement claims arising from use of the Service. This is standard and favorable to Buyer.\n\n### Clause 2 (Section 8.2 - Buyer Indemnification)\n**Risk: HIGH**\nBuyer indemnifies Vendor against \"any and all claims arising from Buyer's use of the Service.\" This language is overly broad - it could include claims caused by Vendor's own negligence. Recommend narrowing to \"claims arising from Buyer's breach of this Agreement or misuse of the Service.\"\n\n### Clause 3 (Section 8.3 - Indemnification Cap)\n**Risk: MEDIUM**\nIndemnification is capped at 12 months of fees paid. Industry standard is typically 24 months or uncapped for IP indemnification. Recommend negotiating to 24 months."
  },
  {
    "instruction": "Summarize the termination provisions and identify any lock-in risks.",
    "input": "[Full 22-page enterprise software license - ~18,000 tokens...]",
    "output": "## Termination Summary\n\n**Term:** 36-month initial term with automatic 12-month renewals.\n**Termination for convenience:** Only permitted at end of current term with 90-day written notice. No mid-term termination without cause.\n**Termination for cause:** 30-day cure period for material breach.\n\n### Lock-in Risks\n1. **Data portability (HIGH RISK):** No contractual obligation for Vendor to provide data export in standard format upon termination. Recommend adding a data portability clause.\n2. **Auto-renewal (MEDIUM RISK):** 90-day notice window is aggressive - calendar reminders recommended.\n3. **Early termination fee (HIGH RISK):** Remaining contract value is due upon early termination. On a $500K/year deal, this could mean $1M+ liability."
  }
]
```
, caption: [Long-context Alpaca training data for legal contract risk analysis],
)

*Why Unsloth is essential here:* Each contract is 12,000-18,000 tokens. Standard HF training on a T4 GPU would produce an OOM error at ~1,024 tokens. Unsloth's memory optimizations enable processing 40,000-token sequences on the same hardware - making full-contract analysis possible without expensive A100 GPUs.

*Inference after training — testing the contract analysis model:*

#figure(
```python
from unsloth import FastLanguageModel

# Load trained model (Unsloth requires its own inference mode switch)
model, tokenizer = FastLanguageModel.from_pretrained("./outputs/contract-analyzer")
FastLanguageModel.for_inference(model)  # REQUIRED: switches from training to inference mode

# Analyze a new contract
contract_text = "[Full 18-page vendor agreement text...]"
prompt = f"""Below is an instruction that describes a task, paired with an input.

### Instruction:
Identify all indemnification clauses and assess risk level for the buyer.

### Input:
{contract_text}

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[-1].strip())
```
, caption: [Unsloth long-context contract analysis inference with for_inference mode switch],
)

#blockquote[
  *Unsloth inference requirement:* Always call `FastLanguageModel.for_inference(model)` before generating — this disables LoRA training mode and enables optimized inference kernels. Forgetting this step results in significantly slower generation.
]

*Performance comparison on this task:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Metric*], [*Standard HF (T4 16GB)*], [*Unsloth (T4 16GB)*]),
    [Max sequence length], [~1,024 tokens (OOM beyond)], [~40,000 tokens],
    [Training time (100 examples, 3 epochs)], [N/A (OOM)], [~45 minutes],
    [VRAM usage], [\>16GB (crash)], [~11GB],
  ),
  caption: [Performance Comparison: Unsloth vs HuggingFace],
  kind: table,
)

#pagebreak(weak: true)

=== 12.6 Head-to-Head Benchmark: Unsloth vs HuggingFace

The repo includes a controlled benchmark (`LLM Fine-Tuning-unsloth-vs-hf/`) with identical setups - same dataset (yahma/alpaca-cleaned, 200 samples), same model (TinyLlama 1.1B), same hyperparameters (50 steps, batch_size=2, gradient_accumulation=4, lr=2e-4).

*Key differences in setup:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Aspect*], [*HuggingFace*], [*Unsloth*]),
    [Model loading], [`AutoModelForCausalLM` + `BitsAndBytesConfig`], [`FastLanguageModel.from_pretrained` (pre-quantized)],
    [LoRA targets], [`q_proj`, `v_proj` (2 modules)], [`q_proj`, `k_proj`, `v_proj`, `o_proj` (4 modules)],
    [Pre-inference step], [None], [`FastLanguageModel.for_inference(model)`],
    [Import order], [Any order], [`import unsloth` MUST be first import],
  ),
  caption: [Key Setup Differences: Unsloth vs HuggingFace],
  kind: table,
)

*Key observations:*

- Unsloth targets 4 attention projections by default (vs HF's typical 2), yet still uses less memory due to kernel optimizations
- Unsloth uses pre-quantized model checkpoints (`unsloth/tinyllama-bnb-4bit`), eliminating runtime quantization overhead
- VRAM measurement: `torch.cuda.reset_peak_memory_stats()` + `torch.cuda.max_memory_reserved()` for accurate peak tracking

=== 12.7 Model Support

Unsloth supports almost all model types:

- *Text-to-Text* (text generation, chat models): Mistral, DeepSeek, Qwen, Gemma, Phi, LLaMA
- *Multimodal models* (text + image + audio)
- *Image-to-Text* (Optical Character Recognition (OCR), vision-language models)
- *Text-to-Speech* (TTS): Orpheus, Bark, XTTS
- *Speech-to-Text* (STT): Whisper, Whisper Large, Wav2Vec2
- *Vision*: Qwen3-VL
- *Classical language models* like BERT

=== 12.8 Training Types Supported

- Full Fine-Tuning
- LoRA
- QLoRA (4-bit and 8-bit)
- 16-bit LoRA
- FP8 Training (on compatible hardware)
- Reinforcement Learning: Reward Modeling, KTO, PPO, GRPO, GSPO, DPO, ORPO

=== 12.9 Why Unsloth Is Fast (Internal Optimizations)

Unsloth's speed comes from deep internal optimizations, not configuration tricks:

- Custom CUDA & Triton kernels (not standard PyTorch autograd)
- Fused attention and MLP operations
- Optimized forward and backward pass
- Smart gradient checkpointing
- Flash-Attention compatibility
- Manual backpropagation engine (not PyTorch autograd)
- Automatic sequence packing

Result: *Exact math* (no approximation), practically zero accuracy loss, 2-3x faster, 50-80% less VRAM.

#blockquote[
  *Benchmark context:* Unsloth's performance claims are model- and hardware-dependent. The head-to-head benchmark in Section 12.6 uses only 50 training steps on 200 samples of TinyLlama 1.1B — a small test that demonstrates the speedup pattern but is not a rigorous large-scale benchmark. Performance gains may vary with model size, sequence length, GPU architecture, and batch size. The claims are well-supported for the tested configurations but should be validated on your specific setup.
]

*Technology Stack:*

#figure(
  image("../diagrams/15-unsloth-tech-stack.png", width: 55%),
  caption: [Unsloth technology stack: user code through CUDA/Triton GPU kernels],
)

=== 12.10 Inference Export Targets

Models trained with Unsloth can be exported to:

- llama.cpp (GGUF format)
- Ollama
- vLLM
- SGLang
- Hugging Face Hub
- Open WebUI

Note: These are inference/deployment tools, NOT training tools.

=== 12.11 Embedding Fine-Tuning with Unsloth

Unsloth now supports embedding model fine-tuning through #link("https://unsloth.ai/docs/new/embedding-finetuning")[SentenceTransformers integration], bringing its speed optimizations to retrieval and RAG use cases (see Section 18 for embedding fine-tuning fundamentals).

*Supported models:* EmbeddingGemma-300M, Qwen3-Embedding (0.6B/4B), BGE-M3, All-MiniLM-L6-v2, ModernBERT, E5-large, MPNet, DistilBERT, and most encoder-only models with `modules.json` files.

*Training modes:* LoRA, QLoRA (4-bit), 16-bit LoRA, and full fine-tuning - all with no pipeline rewrites needed. Cross-encoder training is also supported.

*Performance:* 1.8-3.3x faster training with 20% less memory compared to standard Flash Attention 2 implementations. EmbeddingGemma-300M with QLoRA requires just 3GB VRAM.

#figure(
```python
from unsloth import FastSentenceTransformer

# Load embedding model with Unsloth optimizations
model = FastSentenceTransformer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    for_inference=True,
)

# Encode and compute similarity
query_emb = model.encode_query("What is LoRA?")
doc_emb = model.encode_document("LoRA is a parameter-efficient fine-tuning technique...")
score = model.similarity(query_emb, doc_emb)
```
, caption: [Unsloth FastSentenceTransformer for optimized embedding encoding and similarity],
)

*Key advantages over standard Sentence Transformers:* Automatic pooling defaults, gradient checkpointing patches for DistilBERT/MPNet, and universal deployment (works with transformers, LangChain, Ollama, vLLM, llama.cpp) with no vendor lock-in or accuracy degradation.

=== 12.12 Faster MoE Training with Split LoRA

Unsloth provides #link("https://unsloth.ai/docs/new/faster-moe")[optimized training for Mixture of Experts (MoE) models], delivering up to 12x faster training with 35%+ less VRAM through custom Triton kernels and a technique called *Split LoRA*.

*Supported MoE models:* Qwen3 (30B-A3B, 235B, VL, Coder), GPT-OSS (Open-Source Series) (20B, 120B), DeepSeek (R1, V3, V3.1, V3.2), GLM (4.6, 4.7, Flash).

*Split LoRA approach:* Instead of materializing full LoRA deltas across all experts, Unsloth reorders matrix operations using associativity to avoid peak memory expansion. This becomes advantageous when sequence length exceeds ~16K tokens.

*Performance benchmarks:*

- GPT-OSS (Open-Source Series) BF16: 7x faster training, 36% VRAM reduction (B200, 16K context)
- Qwen3-30B-A3B: 1.7x speedup, ~35% better memory efficiency
- GLM 4.7 Flash: 2.6x faster throughput, \>15% less VRAM
- Enables ~6x longer context compared to standard approaches

#figure(
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",
)
```
, caption: [Unsloth Split LoRA configuration for efficient MoE model training],
)

*Automatic backend selection:* Unsloth chooses the optimal implementation - `grouped_mm` (default, broad compatibility), `unsloth_triton` (2.5x speedup on A100), or `native_torch` (fallback). Works across T4 through B200 and consumer GPUs (RTX 3090).

=== 12.13 GRPO Long Context Training

Unsloth enables #link("https://unsloth.ai/docs/new/grpo-long-context")[~7x longer context for GRPO reinforcement learning] through flattened sequence chunking and activation offloading for log softmax computation - with no accuracy or speed degradation.

*Context length achievements (single GPU):*

- 380K context: GPT-OSS (Open-Source Series) QLoRA on 192GB B200
- 110K context: Qwen3-8B GRPO on 80GB H100 with vLLM + QLoRA
- 65K context: GPT-OSS (Open-Source Series) with BF16 LoRA
- 20-32K context: 24GB VRAM setups

*Supported models:* GPT-OSS (Open-Source Series), Qwen3-8B, Qwen3-VL-8B, Llama, Gemma, and auto-supported models.

*Key parameters for auto-tuning:*

- `unsloth_grpo_mini_batch`: Controls batch dimension chunking
- `unsloth_logit_chunk_multiplier`: Handles sequence dimension chunking (defaults to `max(4, context_length // 4096)`)

*Additional capabilities:* Integration with vLLM for 11x faster generation, weight-sharing between training and generation, Flex Attention support, and Float8 training compatibility.

#pagebreak(weak: true)

=== 12.14 Tutorial: Training a Reasoning Model with GRPO

Unsloth provides a #link("https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo")[step-by-step tutorial] for training your own reasoning model using GRPO, with pre-configured Colab notebooks for quick setup.

*Base models:* GPT-OSS (Open-Source Series)-20b, Qwen3 (4B), DeepSeek-R1, Llama 3.2. VRAM requirement is approximately 1GB per billion parameters.

*Step 1 - Define a structured reasoning prompt:*

#figure(
```python
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>...</reasoning>
<answer>...</answer>
"""
```
, caption: [GRPO structured reasoning system prompt with reasoning and answer tags],
)

*Step 2 - Prepare reward functions (verifiers):*

Reward functions tell the model whether its outputs are good or bad. Approaches include:
- Rule-based scoring (e.g., +1 for correct answer format, -1 for excessive length)
- Pre-built verifiers (e.g., GSM8K math answer checking)
- LLM-based evaluation using a separate model to judge quality

*Step 3 - Configure and train with GRPOConfig:*

Key hyperparameters:
- `use_vllm`: Enable vLLM for fast inference during generation
- `num_generations`: Number of completions per prompt (for group comparison)
- `max_steps`: Total training iterations (minimum ~300 steps / ~30 minutes)
- Loss type variants: `grpo`, `bnpo`, `dr_grpo`, `dapo`

*Step 4 - Evaluate and export:*

After training, the model develops increasingly sophisticated reasoning chains. Export options include 16-bit merged weights, GGUF format, or push directly to Hugging Face Hub.

*What makes GRPO effective for reasoning:* Unlike SFT which requires curated reasoning examples, GRPO lets the model discover its own reasoning strategies through trial and reward. The model generates multiple completions per prompt, compares them within the group, and reinforces strategies that lead to correct answers.

=== 12.15 Section Summary

Unsloth provides 2-3x faster training and 50-80% VRAM reduction through custom CUDA/Triton kernels, fused operations, flash attention, and manual backpropagation - all with zero accuracy loss. It enables training on free Colab GPUs that would otherwise produce OOM errors. Unsloth also extends its optimizations to embedding fine-tuning for RAG (Section 12.11), MoE models with Split LoRA for up to 12x speedup (Section 12.12), GRPO long-context RL training supporting up to 380K context on a single GPU (Section 12.13), and a complete tutorial for training reasoning models with GRPO (Section 12.14).

#line(length: 100%, stroke: 0.5pt + luma(200))
]
