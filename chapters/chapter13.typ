#import "../template.typ": *

#let Chapter13() = [
== 13 - Axolotl Framework

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-19-Axolotl/axolotl_final_code.ipynb")[`axolotl_final_code.ipynb`]
  *Config examples:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/tree/main/LLM%20Fine-Tuning-19-Axolotl/axolotal-config")[`axolotal-config/`]
  *Docker setup:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-19-Axolotl/axolotl-docker-setup-steps.md")[`axolotl-docker-setup-steps.md`]
]

=== 13.1 What Is Axolotl

Axolotl is a config-driven fine-tuning framework that simplifies complex training setups. Unlike LLaMA Factory (which provides a UI), Axolotl is designed for engineers who want maximum control through YAML/Python configuration.

=== 13.2 Axolotl vs LLaMA Factory vs Unsloth

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Feature*], [*LLaMA Factory*], [*Unsloth*], [*Axolotl*]),
    [*Interface*], [Web UI + CLI], [Python API], [YAML config + CLI],
    [*Primary strength*], [Beginner-friendly], [Speed/memory optimization], [Flexibility/configurability],
    [*Docker support*], [Limited], [No], [First-class (official Docker image)],
    [*Multi-GPU*], [Via HF Accelerate], [Single GPU focus], [Fully Sharded Data Parallel (FSDP), DeepSpeed native support],
    [*DPO support*], [Yes], [Yes], [Yes (separate config)],
    [*Sample packing*], [Yes], [Yes (automatic)], [Yes],
    [*Custom optimizers*], [Limited], [No], [`paged_adamw_8bit`, `adamw_torch`, others],
  ),
  caption: [Axolotl vs LLaMA Factory vs Unsloth],
  kind: table,
)

#pagebreak(weak: true)

=== 13.3 Configuration-Driven Training

Axolotl accepts configuration either as a Python dictionary or as a YAML file. Both approaches specify the same parameters - base model, adapter type, quantization, dataset paths, and training hyperparameters.

*Programmatic config (Python):*

#figure(
```python
from axolotl.utils.config import DictDefault

cfg = DictDefault({
    "base_model": "Qwen/Qwen2.5-3B-Instruct",
    "load_in_4bit": True,
    "adapter": "qlora",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "down_proj", "up_proj"],
    "datasets": [{"path": "bpHigh/pirate-ultrachat-10k", "type": "chat_template"}],
    "chat_template": "qwen3",
    "optimizer": "paged_adamw_8bit",
    "lr_scheduler": "cosine",
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "sample_packing": True,
    "max_grad_norm": 0.1,
    "fp16": True,
})
```
, caption: [Axolotl programmatic configuration with DictDefault for QLoRA fine-tuning],
)

#pagebreak(weak: true)

*YAML config (`base_sft_lora.yaml`):*

#figure(
```yaml
base_model: meta-llama/Llama-2-7b-hf
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
datasets:
  - path: timdettmers/openassistant-guanaco
    type: chat_template
sequence_len: 2048
micro_batch_size: 1
gradient_accumulation_steps: 8
optimizer: adamw_torch
learning_rate: 2e-4
fp16: true
gradient_checkpointing: true
```
, caption: [Axolotl YAML config base_sft_lora.yaml for Llama 2 LoRA fine-tuning],
)

=== 13.4 Training and Inference

Axolotl's Python API loads a YAML or dictionary configuration, prepares the datasets, and runs training in a single call. After training, inference uses the standard HuggingFace tokenizer and model generation pipeline.

#figure(
```python
from axolotl.cli.args import load_cfg
from axolotl.common.datasets import load_datasets
from axolotl.train import train

# Load config, prepare data, train
cfg = load_cfg(cfg)
dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)
model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

# Inference with chat template
messages = [{"role": "user", "content": "Explain LoRA in one paragraph."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
, caption: [Axolotl Python API: load config, prepare datasets, train, and run inference],
)

=== 13.5 DPO with Axolotl

DPO is configured as a separate YAML that overrides the base SFT config:

#figure(
```yaml
training_type: dpo
dpo_beta: 0.1
datasets:
  - path: argilla/ultrafeedback-binarized
    type: preference
```
, caption: [Axolotl DPO configuration with beta and preference dataset],
)

=== 13.6 Docker Workflow

Axolotl provides official Docker images with all dependencies pre-installed, which avoids CUDA version conflicts and simplifies multi-GPU setups. The container supports training, inference, and LoRA merging through simple CLI commands.

#figure(
```bash
# Launch container with GPU access
docker run --gpus all --rm -it axolotlai/axolotl:main-latest

# Train
axolotl train my_config.yaml

# Inference (with optional Gradio UI)
axolotl inference --lora-model-dir outputs/checkpoint-100 --gradio

# Merge LoRA into base model for standalone deployment
axolotl merge-lora --lora-model-dir outputs/checkpoint-100
```
, caption: [Axolotl Docker workflow: launch container, train, infer, and merge LoRA],
)

=== 13.7 Framework Comparison: HF vs Unsloth vs LLaMA-Factory vs Axolotl

#figure(
  table(
    columns: 5,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Dimension*], [*Hugging Face*], [*Unsloth*], [*LLaMA-Factory*], [*Axolotl*]),
    [Core Purpose], [Base ML framework for model training & inference], [Performance engine that accelerates HF], [End-to-end fine-tuning platform with UI], [Training orchestrator for scalable experiments],
    [Performance], [Baseline speed, high VRAM], [2-3x faster, 50-80% less VRAM (verified)], [Same as HF (no kernel optimization)], [Same as HF (no kernel optimization)],
    [GPU-level Optimizations], [None (standard PyTorch ops)], [Custom CUDA + Triton kernels, manual backprop], [None], [None],
    [Workflow & Scale], [Code-heavy, manual setup], [Simple code, single-GPU focused], [UI + CLI driven, structured but less scalable], [YAML-driven, reproducible, multi-GPU, DeepSpeed/FSDP],
  ),
  caption: [Framework Comparison: HF vs Unsloth vs LLaMA-Factory vs Axolotl],
  kind: table,
)

=== 13.8 HF vs Axolotl: Performance Optimization Comparison

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Optimization*], [*Hugging Face Alone*], [*Axolotl*]),
    [Multipacking], [Manual custom collator], [Built-in],
    [Flash Attention], [Manual enable + version issues], [Auto-handled],
    [xFormers], [Manual install + config], [Plug & play],
    [Liger Kernel], [Not native], [Integrated],
    [Cut Cross Entropy], [Custom loss], [Built-in],
    [Sequence Parallelism (SP)], [Very complex], [Partial / advanced],
    [LoRA optimizations], [PEFT tuning needed], [Optimized defaults],
    [Multi-GPU (FSDP1/FSDP2)], [Manual painful], [One-flag],
    [DeepSpeed], [JSON + code glue], [Stable configs],
    [Multi-node training], [torchrun setup], [torchrun / Ray ready],
  ),
  caption: [HF vs Axolotl Performance Optimization Comparison],
  kind: table,
)

=== 13.9 Notable Configuration Options

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Parameter*], [*Purpose*], [*Axolotl Default*]),
    [`sample_packing`], [Combine short sequences into one batch entry], [true],
    [`gradient_checkpointing`], [Trade compute for memory], [true],
    [`max_grad_norm`], [Gradient clipping threshold], [0.1 (aggressive)],
    [`paged_adamw_8bit`], [Memory-efficient optimizer], [Recommended for QLoRA],
    [`cosine` lr_scheduler], [Cosine annealing learning rate], [Smoother than linear],
    [`embeddings_skip_upcast`], [Skip FP32 upcast for embeddings], [Saves memory],
  ),
  caption: [Axolotl Notable Configuration Options],
  kind: table,
)

=== 13.10 Section Summary

Axolotl is a configuration-driven fine-tuning framework that supports both YAML files and Python dictionaries for specifying training setups, including SFT, QLoRA, and DPO workflows. Compared to LLaMA Factory (UI-focused) and Unsloth (single-GPU speed optimization), Axolotl targets multi-GPU scalability with first-class FSDP and DeepSpeed support, official Docker images, and built-in performance features such as sample packing, Flash Attention, xFormers integration, and Liger Kernel. Notable defaults include aggressive gradient clipping (max_grad_norm 0.1), paged_adamw_8bit optimizer for QLoRA, and cosine learning rate scheduling.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
