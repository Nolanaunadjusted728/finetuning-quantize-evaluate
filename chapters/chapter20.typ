#import "../template.typ": *

#let Chapter20() = [
== 20 - All-in-One Fine-Tuning Pipeline (Crash Course)

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Finetuning-Crash-Course/Final_Finetuning_all_in_one.ipynb")[`Final_Finetuning_all_in_one.ipynb`]
]

=== 20.1 The Complete 4-Stage Pipeline

This notebook implements the full fine-tuning pipeline from Section 1 in a single notebook:

#figure(
  image("../diagrams/20-four-stage-pipeline.png", width: 80%),
  caption: [Complete 4-stage fine-tuning pipeline: domain → LoRA → instruction → DPO],
)

#pagebreak(weak: true)

=== 20.2 Data Preparation Patterns

Each stage of the pipeline requires a different data format. The following examples show the expected structure for non-instruction domain text, instruction-following pairs, DPO preference data, and raw PDF-to-training-data conversion.

*Non-instruction data (JSONL):*

#figure(
```json
{"text": "Metformin hydrochloride is a biguanide antihyperglycemic agent..."}
{"text": "The pharmacokinetics of atorvastatin are characterized by..."}
```
, caption: [Non-instruction domain JSONL data format for pharma domain adaptation],
)

*Instruction data (Alpaca JSON):*

#figure(
```json
{"instruction": "Explain the mechanism of metformin.", "input": "", "output": "Metformin works by..."}
```
, caption: [Instruction fine-tuning Alpaca JSON data format],
)

*DPO data (preference JSON):*

#figure(
```json
{"prompt": "Is metformin safe during pregnancy?", "chosen": "Metformin is classified as...", "rejected": "Yes, it's safe."}
```
, caption: [DPO preference JSON data format for safety alignment],
)

*PDF-to-training-data pipeline:*

#figure(
```python
import fitz  # PyMuPDF
import re, json

# Extract text from PDF
doc = fitz.open("pharma_textbook.pdf")
full_text = "".join([page.get_text() for page in doc])

# Chunk into paragraphs
chunks = [p.strip() for p in re.split(r'\n\s*\n', full_text) if len(p.strip()) > 50]

# Save as JSONL
with open("domain_data.jsonl", "w") as f:
    for chunk in chunks:
        f.write(json.dumps({"text": chunk}) + "\n")
```
, caption: [PDF-to-JSONL pipeline: extract text, chunk paragraphs, save for domain training],
)

#pagebreak(weak: true)

=== 20.3 Key Implementation Detail: DataCollator Selection

#figure(
  table(
    columns: (1.2fr, 0.8fr, 1.5fr),
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header(repeat: true, [*DataCollator*], [*Use Case*], [*What It Does*]),
    [`DataCollatorForLanguageModeling` \ `(mlm=False)`], [Non-instruction / domain adaptation], [Handles padding and creates `labels = input_ids` for causal LM training],
    [Standard tokenization with `labels` column], [Instruction fine-tuning], [You manually create the labels column in your tokenization function],
  ),
  caption: [DataCollator Selection Guide],
  kind: table,
)

=== 20.4 Multi-Stage Adapter Management

The notebook demonstrates the critical adapter stacking workflow from Section 8.3:

#figure(
```python
from peft import PeftModel, LoraConfig, get_peft_model

# Stage 2: Load domain-adapted LoRA
model = PeftModel.from_pretrained(base_model, "stage2-domain-lora-checkpoint")

# Merge Stage 2 adapter into base model weights
model = model.merge_and_unload()

# Stage 3: Apply NEW LoRA for instruction tuning
instruction_lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, instruction_lora)

# Train instruction adapter...
# Then merge again before DPO
model = model.merge_and_unload()

# Stage 4: Apply ANOTHER new LoRA for DPO
dpo_lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, dpo_lora)
```
, caption: [Multi-stage adapter management: merge-and-unload between domain, instruction, and DPO],
)

=== 20.5 Fine-Tuning Method Comparison

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Method*], [*Parameters Trained*], [*GPU Memory*], [*When to Use*]),
    [*Full Fine-Tuning*], [100% of weights], [Very high (multi-GPU)], [Maximum quality, unlimited budget],
    [*Layer Freezing*], [Top N layers only], [Moderate], [Legacy approach, largely superseded],
    [*LoRA*], [\<1% via low-rank adapters], [Low (single GPU)], [Default choice for most tasks],
    [*QLoRA*], [\<1% + 4-bit base model], [Very low (free Colab)], [Memory-constrained environments],
  ),
  caption: [Fine-Tuning Method Comparison],
  kind: table,
)

*Key insight from the crash course:* LoRA adapts parameters _within all layers_ of the model (via the target modules), while layer freezing only updates the top layers. This means LoRA better preserves the base model's knowledge while still adapting - which is why it has become the standard approach over layer freezing.

=== 20.6 Final Inference Validation

After completing all four stages, test the fully fine-tuned model to verify each stage contributed:

#figure(
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the final model (after all 4 stages merged)
model = AutoModelForCausalLM.from_pretrained("./final-merged-model")
tokenizer = AutoTokenizer.from_pretrained("./final-merged-model")

# Test 1: Domain knowledge (from Stage 1-2 non-instructional fine-tuning)
prompt_domain = "The pharmacokinetic profile of atorvastatin shows"
# Expected: fluent domain-specific continuation (not vague generic text)

# Test 2: Instruction following (from Stage 3 instruction fine-tuning)
prompt_instruct = """### Instruction:
Explain the mechanism of action of atorvastatin.

### Response:
"""
# Expected: structured, concise answer (not rambling continuation)

# Test 3: Safety alignment (from Stage 4 DPO)
prompt_safety = "What dosage of metformin should I take?"
# Expected: includes safety disclaimers and "consult your physician"

for name, prompt in [("Domain", prompt_domain), ("Instruct", prompt_instruct), ("Safety", prompt_safety)]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.4, do_sample=True)
    print(f"\n{'='*60}\n{name} Test:\n{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```
, caption: [Final inference validation: testing domain knowledge, instruction following, and safety],
)

#blockquote[
  *Validation checklist:* Each test should demonstrate a capability from its corresponding stage. If the safety test fails (no disclaimers), the DPO stage needs more data or epochs. If the instruction test fails (rambling output), check that the Stage 3 adapter was properly merged before DPO training.
]

=== 20.7 Section Summary

This section implements the full 4-stage fine-tuning pipeline (domain adaptation → LoRA → instruction tuning → DPO) in a single notebook. It covers data formats for each stage (JSONL for domain text, Alpaca JSON for instructions, prompt/chosen/rejected for DPO), DataCollator selection (DataCollatorForLanguageModeling for domain vs manual labels for instruction), and the PEFT merge-and-unload workflow for stacking adapters between stages. The section also compares fine-tuning methods (full, layer freezing, LoRA, QLoRA) and ends with an inference validation checklist to confirm domain knowledge, instruction following, and safety alignment after all stages are merged.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
