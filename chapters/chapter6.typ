#import "../template.typ": *

#let Chapter6() = [
== 6 - Non-Instructional Fine-Tuning

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-14-Train-LLMs-on-Your-PDF-Text-Data%20-Domain-Specific-Fine-Tuning-with-HuggingFace/non_Instruction_pretrain_llm_finetuning_on_domain_specific_data.ipynb")[`non_Instruction_pretrain_llm_finetuning_on_domain_specific_data.ipynb`]
]

=== 6.1 Data Preparation Pipeline

For non-instructional fine-tuning, the data pipeline follows these steps:

#figure(
  image("../diagrams/14-data-preparation-pipeline.png", width: 65%),
  caption: [Data Preparation Pipeline],
)

*Chunking strategies:*

- Paragraph-based splitting (using regex patterns like double newlines)
- Semantic chunking
- Token-length-based chunking (constrained by model context window)
- Hybrid approaches

*Context window reference:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Model*], [*Max Tokens*], [*Approx. Words*]),
    [GPT-1], [512], [~350],
    [GPT-2], [1,024], [~750],
    [GPT-3], [2,048], [~1,500],
    [GPT-3.5], [4,096], [~3,000],
    [GPT-5], [200,000], [~150,000],
    [Llama 4 Scout], [10,000,000], [~8,000,000],
    [Gemini 3 Pro], [1,000,000], [~800,000],
  ),
  caption: [Context Window Reference by Model],
  kind: table,
)

#pagebreak()

=== 6.2 Hugging Face Libraries Used

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Library*], [*Creator*], [*Purpose*]),
    [`transformers`], [Hugging Face], [Model loading, tokenization, training],
    [`datasets`], [Hugging Face], [Data loading and processing],
    [`accelerate`], [Hugging Face], [Multi-GPU setup (dependency)],
    [`bitsandbytes`], [Tim Dettmers], [Quantized model loading (4-bit, 8-bit)],
    [`peft`], [Hugging Face], [LoRA configuration, parameter-efficient fine-tuning],
    [`trl`], [Hugging Face], [Transformer Reinforcement Learning - SFT, DPO trainers],
    [`fitz` (PyMuPDF)], [-], [PDF text extraction],
  ),
  caption: [HuggingFace Libraries for Fine-Tuning],
  kind: table,
)

=== 6.3 Practical Implementation

*Step 1: Extract text from PDF*

#figure(
```python
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text.append(page.get_text())
    return text
```
, caption: [Extracting text from PDF pages using PyMuPDF (fitz)],
)

*Step 2: Split into paragraphs (chunking)*

#figure(
```python
import re

def split_paragraphs(pages):
    paragraphs = []
    for page in pages:
        chunks = re.split(r'\n\n+', page)  # Split on double newlines
        for chunk in chunks:
            if len(chunk) > 30:  # Only keep chunks with >30 characters (~8-12 tokens)
                paragraphs.append(chunk)
    return paragraphs
```
, caption: [Paragraph-based text chunking using regex double-newline splitting],
)

#pagebreak()

*Step 3: Convert to Hugging Face Dataset format*

#figure(
```python
from datasets import Dataset

data = [{"text": chunk} for chunk in paragraphs]
dataset = Dataset.from_list(data)
```
, caption: [Converting chunked text paragraphs into HuggingFace Dataset format],
)

The resulting dataset has a single `text` column with chunked text in multiple rows - matching the format of pre-built datasets like FineWeb, Pile PubMed, and OpenWebText.

#blockquote[
  *Sequence Packing (production optimization):* The paragraph-based chunking above is simple but wasteful — short chunks require padding, leaving GPU compute underutilized. In production CPT, *sequence packing* concatenates all documents end-to-end (separated by End of Sequence (`<EOS>`) tokens), then splits the result into fixed-length blocks of exactly `context_length` tokens (e.g., 4096). This achieves 0% padding and 100% compute utilization. Unsloth and Axolotl support this via `sample_packing: true` (see Sections 11, 12).
]

*Step 4: Tokenization*

#figure(
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token

def tokenize_function(example):
    tokens = tokenizer(example["text"], truncation=True, padding=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # Key step: labels = input_ids for causal LM
    return tokens

tokenized_data = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
```
, caption: [Tokenization for causal LM: setting labels equal to input_ids for next-token prediction],
)

*Why `labels = input_ids.copy()`:* This is the key step for causal (autoregressive) language modeling. By setting labels equal to input IDs, the model learns to predict the next token in the same sequence - the self-supervised training objective.

*Padding token explanation:* Pad tokens make all sequences in a batch the same length. If input_1 has 3 tokens ("Hello world .") and input_2 has 4 tokens ("Good morning everyone ."), we append a PAD token to input_1 to match lengths. The PAD token is a dedicated token (often `[PAD]` or set to the EOS token ID when no PAD token exists) that the model learns to ignore via attention masking.

*Step 5: Model loading and training*

Attempting full fine-tuning on a free Colab GPU produces an *out-of-memory error* because all model parameters are being retrained. The solution: #link("https://arxiv.org/abs/2106.09685")[LoRA]-based parameter-efficient fine-tuning.

#pagebreak()

*Step 6: LoRA configuration*

#figure(
```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Next-token prediction
    r=8,                            # Rank of the low-rank matrices
    lora_alpha=32,                  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which attention layers to adapt
    lora_dropout=0.05,
    bias="none"
)

# Load model in 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
model = get_peft_model(model, lora_config)
```
, caption: [LoRA configuration and 8-bit model loading for memory-efficient fine-tuning],
)

*Step 7: Training*

#figure(
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./tiny-llama-domain",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    report_to="none"
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_data)
trainer.train()
```
, caption: [TrainingArguments and Trainer setup for non-instructional domain fine-tuning],
)

#pagebreak(weak: true)

=== 6.4 Worked Example: Pharmaceutical Domain Adaptation

*Scenario:* You work at a pharmaceutical company and want the model to understand drug terminology, clinical trial language, and pharmacological concepts before teaching it to answer questions.

*Source data:* 50 PDF research papers on cardiovascular drugs (statins, ACE inhibitors, beta-blockers).

*Sample chunks after processing:*

#figure(
```
Row 1: "Atorvastatin calcium, marketed as Lipitor, is a synthetic lipid-lowering agent.
        It competitively inhibits HMG-CoA reductase, the rate-limiting enzyme in
        cholesterol biosynthesis. Peak plasma concentrations are achieved within 1-2
        hours of oral administration, with an absolute bioavailability of approximately 14%."

Row 2: "In the ASCOT-LLA trial (n=10,305), atorvastatin 10mg reduced the relative risk
        of primary coronary events by 36% (HR 0.64, 95% CI 0.50-0.83, p=0.0005) compared
        to placebo over a median follow-up of 3.3 years."

Row 3: "Adverse effects include myalgia (5-10% of patients), elevated hepatic transaminases,
        and rare cases of rhabdomyolysis. CYP3A4 inhibitors (e.g., clarithromycin, itraconazole)
        increase systemic exposure and should be co-administered with caution."
```
, caption: [Pharmaceutical domain training chunks: drug descriptions and clinical trial data],
)

*Before fine-tuning (base TinyLlama):*

#figure(
```
Prompt: "The pharmacokinetic profile of atorvastatin shows"
Output: "that the drug is effective and has been used by many people around the world
         for various health conditions..."  ← vague, generic, no domain knowledge
```
, caption: [Base TinyLlama before domain fine-tuning: vague generic continuation],
)

*After non-instructional fine-tuning:*

#figure(
```
Prompt: "The pharmacokinetic profile of atorvastatin shows"
Output: "an absolute bioavailability of approximately 14%, with peak plasma concentrations
         achieved within 1-2 hours of oral administration. The drug undergoes extensive
         first-pass metabolism via CYP3A4, with an elimination half-life of 14 hours..."
         ← fluent domain language, accurate terminology, coherent continuation
```
, caption: [TinyLlama after pharma domain fine-tuning: accurate pharmacokinetic terminology],
)

*What changed:* The model now "speaks pharma." It knows drug names, clinical trial terminology (HR, CI, p-values), pharmacokinetic concepts (bioavailability, CYP3A4), and adverse effect vocabulary. It cannot yet answer structured questions - that comes in Section 7.

#pagebreak()

*Inference after training — testing the domain-adapted model:*

#figure(
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model + trained LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, "pharma-domain-lora-checkpoint")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate domain-adapted text continuation
prompt = "The pharmacokinetic profile of atorvastatin shows"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    repetition_penalty=1.2,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Expected: fluent continuation using pharma terminology (bioavailability, CYP3A4, etc.)
```
, caption: [Loading domain-adapted LoRA model and generating pharma text continuation],
)

#pagebreak(weak: true)

=== 6.5 Mitigating Catastrophic Forgetting

Continued pre-training on domain text risks degrading the model's general capabilities — the model may "forget" how to perform non-domain tasks. This is one of the most important practical challenges in fine-tuning and deserves careful monitoring.

*How to detect catastrophic forgetting:*

Track *two perplexity curves* during training:
- *Domain perplexity* (on a held-out domain validation set) — should decrease steadily
- *General perplexity* (on a general benchmark like WikiText-2 or a sample of C4) — should remain roughly stable

#figure(
  image("../diagrams/04-perplexity-monitoring.png", width: 85%),
  caption: [Perplexity Monitoring During Fine-Tuning],
)

*Warning signs and thresholds:*

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Metric*], [*Healthy*], [*Warning*], [*Action Required*]),
    [General perplexity increase], [\<5% above baseline], [5-15% above baseline], [\>15% above baseline — stop training],
    [Domain perplexity], [Decreasing, plateauing], [Oscillating], [Rising — data quality issue],
    [General task accuracy (e.g., MMLU subset)], [\<2% drop], [2-5% drop], [\>5% drop — forgetting is severe],
  ),
  caption: [Warning signs and thresholds],
  kind: table,
)

#figure(
```python
# Monitoring setup: evaluate on both domain and general data during training
from datasets import load_dataset

# General benchmark for forgetting detection
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# Compute perplexity on general data every N steps
def compute_general_perplexity(model, tokenizer, dataset, max_samples=200):
    """Track this metric during training — if it rises >15%, stop."""
    model.eval()
    total_loss, total_tokens = 0, 0
    for text in dataset["text"][:max_samples]:
        if len(text.strip()) < 10:
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    return math.exp(total_loss / total_tokens)  # Perplexity
```
, caption: [Monitoring general perplexity during training to detect catastrophic forgetting],
)

*Mitigation strategies (in order of effectiveness):*

1. *Data mixing (most important):* Blend domain text with general-purpose data. A common ratio is 70% domain / 30% general, but the optimal ratio depends on how specialized your domain is. For highly specialized domains (e.g., organic chemistry), use 50/50 or even 40/60 to preserve more general capability. For domains closer to general text (e.g., customer support), 80/20 is sufficient. The general data can come from SlimPajama, RedPajama, or a filtered sample of the model's original pre-training data if available.

2. *Low learning rate:* Use 1e-5 to 5e-5 for continued pre-training (lower than the 1e-4 to 2e-4 typical for SFT). The intuition: smaller weight updates preserve more of the original learned representations.

3. *Short training with early stopping:* Domain adaptation often converges within 1-3 epochs. Monitor general perplexity and stop when it begins to rise — continued training past this point actively degrades the model.

4. *LoRA instead of full fine-tuning:* LoRA inherently limits forgetting because only a small parameter subspace is modified. The base model weights remain frozen, preserving general capabilities. This is one of LoRA's underappreciated benefits.

5. *Replay-based methods (advanced):* Periodically re-expose the model to examples from its original training distribution. This can be implemented by interleaving general-purpose examples into training batches at regular intervals.

#pagebreak()

#blockquote[
  *Note on Elastic Weight Consolidation (EWC):* EWC (Kirkpatrick et al., 2017) penalizes changes to weights that were important for previous tasks, measured via the Fisher Information Matrix. While theoretically elegant, it adds significant memory overhead (storing the Fisher diagonal for all parameters) and has seen limited adoption in LLM fine-tuning compared to the simpler strategies above. It is more commonly used in continual learning research settings.
]

=== 6.6 Section Summary

Non-instructional fine-tuning trains a base model on domain-specific plain text using the self-supervised next-token prediction objective. The process involves extracting text from documents, chunking, tokenizing, and training with LoRA for memory efficiency. The result is a domain-adapted model that understands the vocabulary and terminology of the target domain.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
