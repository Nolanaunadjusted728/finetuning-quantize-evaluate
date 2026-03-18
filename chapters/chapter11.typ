#import "../template.typ": *

#let Chapter11() = [
== 11 - LLaMA Factory Framework

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-17-Llama-Factory/llamafactory.ipynb")[`llamafactory.ipynb`]
  *Config:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-17-Llama-Factory/train_gemma_qlora.yaml")[`train_gemma_qlora.yaml`]
]

=== 11.1 What Is LLaMA Factory

LLaMA Factory is an open-source fine-tuning project (GitHub: hiyouga/LLaMA-Factory, ~63K stars, ~7.7K forks) created by Yaowei Zheng. It provides both a *Web UI (LLaMA Board)* and a *CLI* for model fine-tuning.

*Key insight:* LLaMA Factory is built on top of Hugging Face libraries (transformers, peft, bitsandbytes, trl). It does not implement custom model code - it wraps HF libraries with custom training pipelines, UI, data templates, and dataset handling.

=== 11.2 Supported Training Methods

#figure(
  table(
    columns: 2,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Method*], [*Description*]),
    [SFT], [Supervised fine-tuning with LoRA/QLoRA],
    [DPO], [Direct Preference Optimization],
    [RLHF], [Reinforcement Learning from Human Feedback (reward modeling + PPO)],
    [RLAIF], [Reinforcement Learning from AI Feedback],
    [Full Fine-tuning], [Train all model weights],
  ),
  caption: [LLaMA Factory Supported Training Methods],
  kind: table,
)

=== 11.3 Supported Data Formats

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Format*], [*Columns*], [*Use Case*]),
    [*Alpaca*], [instruction, input, output], [Instruction fine-tuning],
    [*ShareGPT*], [conversations (from/value pairs)], [Conversational fine-tuning],
    [*DPO*], [prompt, chosen, rejected], [Preference alignment],
  ),
  caption: [LLaMA Factory Supported Data Formats],
  kind: table,
)

=== 11.4 Data Configuration

LLaMA Factory uses a central registry file to define how custom datasets should be parsed. Custom datasets require an entry in `data/dataset_info.json`:

#pagebreak(weak: true)

#figure(
```json
{
  "my_custom_data": {
    "file_name": "data/my_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```
, caption: [LLaMA Factory dataset_info.json entry for registering a custom Alpaca dataset],
)

For Hugging Face datasets, specify the HF repository URL instead of a local file path.

=== 11.5 CLI Training

LLaMA Factory supports fully configuration-driven training through YAML files, allowing you to launch training jobs from the command line without writing any Python code.

#figure(
```bash
# Create a YAML config file with all parameters
# Then run:
python -m llamafactory.cli train config.yaml
```
, caption: [LLaMA Factory CLI training command with YAML configuration file],
)

The YAML config includes: model name/path, stage (sft/dpo), fine-tuning type (lora), dataset, template, learning rate, epochs, batch size, output directory, quantization settings, etc.

==== 11.5.1 LLaMA Factory CLI Commands Reference

#figure(
  table(
    columns: 2,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Command*], [*Purpose*]),
    [`llamafactory-cli train config.yaml`], [Run a training/fine-tuning session (LoRA, QLoRA, Full FT)],
    [`llamafactory-cli chat config.yaml`], [Launch CLI chat after training/export],
    [`llamafactory-cli eval config.yaml`], [Evaluate model (perplexity/custom eval)],
    [`llamafactory-cli export config.yaml`], [Merge adapters and export final model],
    [`llamafactory-cli api config_api.yaml`], [Start an API server endpoint (OpenAI-style)],
    [`llamafactory-cli webui`], [Launch graphical interface (training, eval, chat, export)],
    [`llamafactory-cli webchat`], [Launch web-based chat UI],
    [`llamafactory-cli version`], [Show installed version],
  ),
  caption: [LLaMA Factory CLI Commands Reference],
  kind: table,
)

==== 11.5.2 Key YAML Configuration Parameters

*Model Download Source (`hub_name`):*

- `huggingface` - Default recommended source
- `modelscope` - China-based model hub for Chinese models
- `openmind` - Local folder or custom storage for offline loading

*Fine-Tuning Method (`finetuning_type`):*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Method*], [*Description*], [*VRAM*]),
    [`lora`], [Train only small adapters], [Low - best practical method],
    [`full`], [Train all weights], [Very high - maximum accuracy],
    [`freeze`], [Freeze some layers, train others], [Medium - stable training],
  ),
  caption: [Fine-Tuning Method Parameter Options],
  kind: table,
)

*Quantization Bit (`quantization_bit`):*

#figure(
  table(
    columns: 2,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Setting*], [*Effect*]),
    [`none`], [Full precision - highest VRAM],
    [`8`], [8-bit loading - moderate VRAM reduction],
    [`4`], [4-bit (QLoRA) - minimum VRAM, best choice for fine-tuning],
  ),
  caption: [Quantization Bit Configuration],
  kind: table,
)

*Quantization Method (`quantization_method`):*

- `bnb` (BitsAndBytes) - Stable, recommended default
- `hqq` (High Quality Quantization) - Better accuracy preservation, slightly heavier
- `etqq` - Experimental fast quantization, not stable for all models

*Training Speed Enhancer (`ster`):*

- `auto` - LLaMA Factory automatically selects fastest kernel
- `flash_attn` - Flash Attention for GPU memory and speed improvement
- `unsloth` - Ultra-fast training, halves VRAM, best for laptops/Colab
- `liger_kernel` - Advanced fused kernels, stable + faster training

*Compute Type:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Type*], [*Description*], [*Best For*]),
    [`bf16`], [Fast, low-memory, stable], [A100, H100, RTX 4090],
    [`fp16`], [Half precision, memory efficient], [Most GPUs],
    [`fp32`], [Full precision], [Maximum accuracy (slow)],
    [`pure_bf16`], [All operations in bf16], [Premium GPUs, big models],
  ),
  caption: [Compute Type Options],
  kind: table,
)

*RoPE Scaling (`rope_scaling`):* Extends context window beyond default

- `none` - Original context length
- `linear` - Basic long-context extension
- `dynamic` - Adaptive positional scaling, more stable
- `yarn` - SOTA method for 8K-\>32K-\>128K upgrades

*Other important parameters:*

- `learning_rate`: 5e-5 to 2e-4 recommended for QLoRA
- `cutoff_len`: Max input token length (common: 2048 or 4096)
- `max_grad_norm`: Gradient clipping to prevent gradient explosion (common: 1.0)
- `max_samples`: Limit dataset size for training speed/cost control

=== 11.6 Worked Example: E-Commerce Customer Support Bot

*Scenario:* An e-commerce company wants to fine-tune Llama 4 Scout 17B to handle customer support queries - returns, order tracking, product recommendations - using LLaMA Factory's UI instead of writing code.

*Training data (ShareGPT format, saved as `data/ecommerce_support.json`):*

#figure(
```json
[
  {
    "conversations": [
      {"from": "system", "value": "You are a helpful customer support agent for ShopMax, an online electronics retailer. Be concise, empathetic, and solution-oriented."},
      {"from": "user", "value": "I ordered a laptop 5 days ago and it still hasn't shipped. Order #SM-29471."},
      {"from": "assistant", "value": "I understand the frustration with the delay on order #SM-29471. Let me look into this for you.\n\nOur standard processing time is 1-2 business days, so a 5-day delay is unusual. This could be due to:\n1. Stock verification for high-value items (laptops require serial number logging)\n2. Payment verification hold\n3. Warehouse backlog\n\nI've flagged your order for priority processing. You should receive a shipping confirmation with tracking within 24 hours. If you don't, please reply to this thread and I'll escalate to our fulfillment team directly."}
    ]
  },
  {
    "conversations": [
      {"from": "system", "value": "You are a helpful customer support agent for ShopMax, an online electronics retailer. Be concise, empathetic, and solution-oriented."},
      {"from": "user", "value": "The monitor I received has a dead pixel in the center of the screen. Can I return it?"},
      {"from": "assistant", "value": "I'm sorry to hear about the dead pixel - that's definitely a defect covered under our quality guarantee.\n\nHere's how to proceed:\n1. **Return window:** You're within our 30-day return policy for defective items\n2. **Process:** Go to ShopMax.com/returns → enter your order number → select \"Defective item\" → print the prepaid return label\n3. **Replacement:** We'll ship a replacement within 1 business day of receiving your return, or you can opt for a full refund\n4. **No restocking fee:** Defective items are exempt from the 15% restocking fee\n\nWould you prefer a replacement or a refund?"}
    ]
  }
]
```
, caption: [ShareGPT training data for e-commerce customer support bot fine-tuning],
)

#pagebreak(weak: true)

*LLaMA Factory `dataset_info.json` entry:*

#figure(
```json
{
  "ecommerce_support": {
    "file_name": "data/ecommerce_support.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations"
    }
  }
}
```
, caption: [LLaMA Factory dataset_info.json entry for ShareGPT-formatted e-commerce data],
)

*LLaMA Factory YAML config (`configs/ecommerce_sft.yaml`):*

#figure(
```yaml
model_name_or_path: meta-llama/Llama-3.1-8B
stage: sft
finetuning_type: lora
dataset: ecommerce_support
template: llama3
lora_rank: 16
lora_alpha: 32
learning_rate: 2e-4
num_train_epochs: 3
per_device_train_batch_size: 4
quantization_bit: 4
output_dir: ./outputs/ecommerce-support-bot
```
, caption: [LLaMA Factory YAML training configuration for QLoRA SFT on Llama 3.1],
)

*What makes this example useful:* LLaMA Factory handles all the boilerplate - tokenization, prompt formatting, LoRA setup, quantization, trainer configuration - from a single YAML file. The same task done manually with Hugging Face (Sections 2-3) requires ~50 lines of Python. This is LLaMA Factory's value proposition.

*Inference after training — testing with LLaMA Factory CLI:*

#figure(
```bash
# Option 1: CLI chat (interactive)
llamafactory-cli chat configs/ecommerce_sft.yaml

# Option 2: Export merged model, then use standard HuggingFace inference
llamafactory-cli export configs/ecommerce_sft.yaml
```
, caption: [LLaMA Factory CLI chat and export commands for trained model deployment],
)

#figure(
```python
# Option 3: Python inference after export
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./outputs/ecommerce-support-bot/merged")
tokenizer = AutoTokenizer.from_pretrained("./outputs/ecommerce-support-bot/merged")

messages = [
    {"role": "system", "content": "You are a helpful customer support agent for ShopMax."},
    {"role": "user", "content": "I received a damaged laptop screen. What should I do?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
, caption: [Python inference with exported LLaMA Factory merged model using chat template],
)

=== 11.7 Section Summary

LLaMA Factory provides a beginner-friendly UI and a scriptable CLI for fine-tuning, built on Hugging Face libraries. It supports SFT, DPO, RLHF, and full fine-tuning with Alpaca, ShareGPT, and DPO data formats.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
