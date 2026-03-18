#import "../template.typ": *

#let Chapter3() = [
== 3 - HuggingFace Ecosystem

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-08-Huggingface/huggingface_crash_course.ipynb")[`huggingface_crash_course.ipynb`]
]

=== 3.1 Authentication Methods

HuggingFace requires authentication to access gated models and private repositories. There are four methods available, ranging from CLI-based login to programmatic token usage.

#figure(
```python
# Method 1: CLI login
# !huggingface-cli login

# Method 2: Programmatic login
from huggingface_hub import login
login(token="hf_...")

# Method 3: Notebook widget
from huggingface_hub import notebook_login
notebook_login()

# Method 4: HfApi
from huggingface_hub import HfApi
api = HfApi(token="hf_...")
```
, caption: [HuggingFace authentication: four methods from CLI login to HfApi],
)

=== 3.2 Datasets Library

The `datasets` library provides efficient data loading with key operations:

#figure(
```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("imdb")

# Key operations
dataset["train"].shuffle(seed=42)           # Randomize order
dataset["train"].select(range(100))          # Take first 100 samples
dataset["train"].filter(lambda x: x["label"] == 1)  # Filter by condition
dataset["train"].map(tokenize_function, batched=True)  # Transform
dataset["train"].train_test_split(test_size=0.2)       # Split

# Streaming mode (avoids downloading terabytes to disk)
streaming_dataset = load_dataset("HuggingFaceFW/fineweb", streaming=True)
for example in streaming_dataset["train"]:
    process(example)
    break  # Process one at a time, no disk storage needed
```
, caption: [HuggingFace datasets: load, shuffle, filter, map, split, and stream operations],
)

*Streaming mode* is critical for large datasets like FineWeb (15TB), C4 (800GB), and OpenWebText - it loads data lazily from the network without downloading the full dataset.

=== 3.3 Tokenizers: Fast vs Slow

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Feature*], [*Slow Tokenizer (Python)*], [*Fast Tokenizer (Rust)*]),
    [Backend], [Pure Python], [Rust (via `tokenizers` library)],
    [Speed], [~1x], [~10-20x faster],
    [Offset mapping], [No], [Yes (`return_offsets_mapping=True`)],
    [Batch processing], [Slow], [Optimized],
  ),
  caption: [Fast vs Slow Tokenizers],
  kind: table,
)

*Training a BPE tokenizer from scratch:*

#figure(
```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=100, special_tokens=["[PAD]", "[UNK]"])
tokenizer.train(["corpus.txt"], trainer)

# Wrap as HuggingFace PreTrainedTokenizerFast
from transformers import PreTrainedTokenizerFast
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```
, caption: [Training a BPE tokenizer from scratch and wrapping as PreTrainedTokenizerFast],
)

=== 3.4 Model Loading Patterns

The `transformers` library provides multiple ways to load models depending on whether you need pre-trained weights, a blank architecture for training from scratch, or a full local copy of the model files.

#figure(
```python
from transformers import AutoModelForCausalLM, AutoConfig

# Load pre-trained (with weights)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

# Load config only (random initialization - for training from scratch)
config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_config(config)  # Random weights!

# Download full model snapshot locally
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-3.1-8B", local_dir="./llama-local")
```
, caption: [Model loading patterns: from_pretrained, from_config, and snapshot_download],
)

*Critical distinction:* `from_pretrained()` loads trained weights. `from_config()` creates a model with the same architecture but *random weights* - useful for pre-training from scratch, never for fine-tuning.

=== 3.5 Pipeline API

The `pipeline` API provides single-line inference for common tasks:

#figure(
```python
from transformers import pipeline

# Sentiment analysis
classifier = pipeline("sentiment-analysis")
classifier("I love this product!")  # → [{"label": "POSITIVE", "score": 0.99}]

# Zero-shot classification (no training needed)
classifier = pipeline("zero-shot-classification")
classifier("This is a cooking tutorial", candidate_labels=["sports", "cooking", "politics"])

# Text generation
generator = pipeline("text-generation", model="gpt2")
generator("The future of AI is", max_length=50)

# Summarization
summarizer = pipeline("summarization", model="google/long-t5-tglobal-base")
summarizer(long_text, max_length=100)

# Question answering (extractive)
qa = pipeline("question-answering")
qa(question="What is LoRA?", context="LoRA is a parameter-efficient fine-tuning technique...")
```
, caption: [HuggingFace Pipeline API for sentiment analysis, generation, summarization, and QA],
)

#pagebreak(weak: true)

=== 3.6 LangChain + HuggingFace Integration

LangChain can use HuggingFace models as its LLM backend, either through the remote Inference API or by running a quantized model locally. This is useful for building RAG pipelines and agentic workflows on top of open-source models.

#figure(
```python
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline

# Remote inference via HF Inference API
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1", task="text-generation")

# Local inference with quantized model
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)
pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta",
                model_kwargs={"quantization_config": bnb_config})
llm = HuggingFacePipeline(pipeline=pipe)
```
, caption: [LangChain integration with HuggingFace for remote and local quantized inference],
)

#pagebreak(weak: true)

=== 3.7 Trainer vs Manual Training Loop

HuggingFace provides two approaches for training: the high-level `Trainer` API and writing a manual PyTorch training loop. The choice depends on how much customization you need.

*HuggingFace Trainer* handles gradient accumulation, mixed precision, logging, checkpointing, distributed training, and evaluation scheduling out of the box. For standard fine-tuning tasks (classification, QA, summarization), Trainer is almost always the right choice because it eliminates boilerplate and encodes best practices.

*Manual PyTorch loop* gives full control over every aspect of training but requires you to implement gradient accumulation, mixed precision contexts, checkpoint saving, metric logging, and learning rate scheduling yourself.

*Code comparison:*

#figure(
```python
# ── Trainer approach (recommended for standard tasks) ──
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,    # Simulates batch size of 32 on limited VRAM
    fp16=True,                        # Mixed precision → ~2x memory savings
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-5,
    warmup_ratio=0.1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)
trainer.train()
```
, caption: [HuggingFace Trainer approach with gradient accumulation and mixed precision],
)

#figure(
```python
# ── Manual PyTorch loop (for non-standard requirements) ──
from torch.amp import autocast, GradScaler  # Modern API (PyTorch 2.0+)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scaler = GradScaler("cuda")
accumulation_steps = 4

model.train()
for epoch in range(3):
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast("cuda"):
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if (step + 1) % 500 == 0:
            # Manual evaluation, logging, checkpointing...
            pass
```
, caption: [Manual PyTorch training loop with gradient accumulation and GradScaler],
)

*Key Trainer features:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Feature*], [*Argument*], [*Effect*]),
    [Gradient accumulation], [`gradient_accumulation_steps=4`], [Simulates larger batch sizes when VRAM is limited],
    [Mixed precision], [`fp16=True` or `bf16=True`], [~2x memory savings; bf16 preferred on Ampere+ GPUs],
    [Periodic evaluation], [`evaluation_strategy="steps"`], [Run eval every N steps or every epoch],
    [Custom callbacks], [`TrainerCallback` subclass], [Custom logging, early stopping, metric tracking],
    [Checkpoint management], [`save_total_limit=3`], [Keeps only the N best/latest checkpoints],
  ),
  caption: [Key Trainer features],
  kind: table,
)

*When to choose a manual training loop:*

- *Custom loss functions* — e.g., contrastive loss, multi-objective losses, or losses that combine multiple model outputs in non-standard ways
- *Non-standard architectures* — models that do not follow the standard `forward() → loss` pattern expected by Trainer
- *Multi-task learning with dynamic task weighting* — where task sampling probabilities or loss weights change during training based on per-task performance
- *Research experimentation* — when you need to modify gradient computation, implement custom regularization, or test novel optimization strategies

#blockquote[
  *Rule of thumb:* Start with Trainer. Only drop to a manual loop when Trainer's callback system cannot express your training logic.
]

=== 3.8 Section Summary

This section provides a practitioner's walkthrough of the HuggingFace ecosystem, covering authentication, the `datasets` library (including streaming mode for terabyte-scale corpora), fast Rust-backed tokenizers, model loading patterns (`from_pretrained` vs `from_config`), and the high-level `pipeline` API for inference. A substantial portion (3.6) is dedicated to evaluation: rule-based metrics (BLEU, ROUGE, METEOR, BERTScore), human evaluation with inter-rater agreement (Cohen's/Fleiss' Kappa), LLM-as-a-Judge with position-bias mitigation, factuality scoring via claim decomposition, and standard benchmarks run through EleutherAI's lm-evaluation-harness. The section concludes with debugging fine-tuning runs, reproducibility practices, LangChain integration for RAG pipelines, and a comparison of HuggingFace's Trainer API versus manual PyTorch training loops.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
