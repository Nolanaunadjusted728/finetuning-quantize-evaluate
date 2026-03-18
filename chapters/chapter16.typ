#import "../template.typ": *

#let Chapter16() = [
== 16 - Small Language Model (SLM) Fine-Tuning

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-22-Finetune-Any-SLM/finetune_any_SLM.ipynb")[`finetune_any_SLM.ipynb`]
]

=== 16.1 SLM vs LLM

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Aspect*], [*LLM (Large Language Model)*], [*SLM (Small Language Model)*]),
    [*Model Size*], [7B -\> 70B+ parameters], [~0.5B to 7B parameters],
    [*Primary Goal*], [General intelligence for many unknown tasks], [High accuracy for known, specific tasks],
    [*Training Data*], [Massive, multi-domain, internet-scale], [Small, curated, domain-specific],
    [*Training Cost*], [Extremely expensive; large GPU clusters, weeks], [Affordable; single GPU, hours or days],
    [*Inference Cost*], [High cost, higher latency], [Cheap, very fast inference],
    [*Architecture*], [Deep transformer stacks with many layers], [Fewer layers, optimized transformer blocks],
    [*Reasoning*], [Strong multi-step reasoning, CoT], [Basic to moderate reasoning],
    [*Context Handling*], [Long context, better memory], [Short-medium context],
    [*Deployment*], [Mostly cloud or API-based], [Local, on-prem, edge, single GPU],
    [*Best Use*], [Chatbots, reasoning, exploration], [Agents, automation, RAG, local tools],
    [*Fine-tuning Cost*], [Expensive (multi-GPU, days/weeks)], [Very cheap (single GPU, hours)],
    [*Examples*], [GPT-5, Claude Opus 4.6, Gemini 3 Pro, Llama 4 Behemoth], [Phi-4-mini, Gemma-3B, Mistral 3 (3B/8B), Qwen 3 (4B/8B), DeepSeek R1 Distill],
  ),
  caption: [SLM vs LLM Comparison],
  kind: table,
)

=== 16.2 Types of SLMs

1. *Distilled models* - Compressed from larger models (e.g., DeepSeek R1 Distill 1.5B)
2. *Task-specific models* - Trained for specific domains from the ground up
3. *Lightweight general-purpose models* - Open-source models like TinyLlama (1.1B), Phi (Microsoft), Gemma (Google)

=== 16.3 Key Research: #link("https://arxiv.org/abs/2409.00009")["SLMs Are the Future of Agentic AI"] (Nvidia, Sept 2025)

*Core finding:* In agentic AI systems, most work consists of small, repetitive, well-defined subtasks. For these tasks, SLMs are more than powerful enough, 10-30x cheaper, have much lower latency, and are easy to fine-tune.

*SLM definition (per this paper):* Below 10 billion parameters - can run on a consumer device or single GPU.

*Recommended SLMs:* Microsoft Phi-3/4, DeepSeek R1 Distill, Nvidia Nemotron, Gemma

*Resource:* LLM Explorer website - 6,000+ SLM models catalogued with maintainer, size, VRAM, license, context length, and download stats.

=== 16.4 SLM Fine-Tuning Practical

The fine-tuning process for SLMs is identical to LLM fine-tuning (using Hugging Face, Unsloth, or LLaMA Factory). The key difference is that SLMs require significantly less VRAM and train much faster, making them ideal for domain-specific deployment on resource-constrained environments.

#pagebreak(weak: true)

=== 16.5 Worked Example: IoT Sensor Anomaly Classifier

*Scenario:* A manufacturing plant has 500 IoT sensors monitoring temperature, vibration, and pressure on industrial equipment. They need a model that classifies sensor readings as normal, warning, or critical - deployed on edge hardware (Raspberry Pi 5 with 8GB RAM). An LLM is too large; a fine-tuned SLM is ideal.

*Model choice:* TinyLlama 1.1B (fits in ~2GB RAM with 4-bit quantization)

*Training data (Alpaca format):*

#figure(
```json
[
  {
    "instruction": "Classify the following sensor reading and explain your reasoning.",
    "input": "Sensor: VIB-042 (Bearing vibration, Pump #7)\nTimestamp: 2025-10-15T14:23:00Z\nReading: 7.2 mm/s RMS\nBaseline: 2.1 mm/s RMS\nTrend: Increasing 0.3 mm/s/day over past 5 days\nTemperature: 82°C (baseline: 65°C)",
    "output": "**Classification: CRITICAL**\n\nReasoning:\n1. Vibration at 7.2 mm/s RMS is 3.4x baseline (2.1 mm/s), exceeding the ISO 10816-3 alert threshold of 4.5 mm/s for Class II machinery\n2. Temperature 17°C above baseline indicates bearing degradation\n3. Consistent upward trend (0.3 mm/s/day) suggests progressive failure, not transient event\n\nRecommended action: Schedule immediate inspection of Pump #7 bearing assembly. Risk of catastrophic bearing failure within 48-72 hours if trend continues."
  },
  {
    "instruction": "Classify the following sensor reading and explain your reasoning.",
    "input": "Sensor: TEMP-118 (Coolant temperature, CNC Mill #3)\nTimestamp: 2025-10-15T09:45:00Z\nReading: 38°C\nBaseline: 35°C\nTrend: Stable (±1°C over past 7 days)\nCoolant flow rate: 12 L/min (baseline: 12 L/min)",
    "output": "**Classification: NORMAL**\n\nReasoning:\n1. Temperature 38°C is 3°C above baseline, within normal operating variance (±5°C)\n2. Trend is stable - no progressive drift\n3. Coolant flow rate matches baseline exactly, indicating no pump or line issues\n\nNo action required. Next scheduled maintenance: per standard interval."
  },
  {
    "instruction": "Classify the following sensor reading and explain your reasoning.",
    "input": "Sensor: PRES-007 (Hydraulic pressure, Press #1)\nTimestamp: 2025-10-15T16:10:00Z\nReading: 185 bar\nBaseline: 210 bar\nTrend: Dropped 15 bar in past 2 hours\nOil temperature: 71°C (baseline: 55°C)",
    "output": "**Classification: WARNING**\n\nReasoning:\n1. Pressure at 185 bar is 12% below baseline (210 bar), approaching minimum operating threshold of 180 bar\n2. Rapid decline (15 bar in 2 hours) indicates active issue - likely internal leak or pump cavitation\n3. Oil temperature 16°C above baseline is consistent with increased bypass flow from a leak\n\nRecommended action: Monitor closely. If pressure drops below 180 bar, shut down Press #1 and inspect hydraulic lines and seals. Check hydraulic fluid level."
  }
]
```
, caption: [SLM training data: IoT sensor anomaly classification in Alpaca format],
)

#pagebreak(weak: true)

*Why an SLM, not an LLM:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Requirement*], [*LLM (Llama 70B)*], [*SLM (TinyLlama 1.1B)*]),
    [Deployment target], [Cloud server, \$2-5K/month], [Raspberry Pi 5, \$80 one-time],
    [Inference latency], [2-5 seconds (API call)], [\<200ms (local)],
    [Offline capability], [No (requires internet)], [Yes (fully local)],
    [Cost per 1M classifications], [~\$50 (API tokens)], [~\$0 (runs locally)],
    [Accuracy on this specific task], [~95% (overkill)], [~92% (sufficient after fine-tuning)],
  ),
  caption: [Why SLM Over LLM for Edge Deployment],
  kind: table,
)

*Key insight from #link("https://arxiv.org/abs/2402.14905")[Nvidia's paper]:* This is exactly the kind of task SLMs excel at - well-defined, repetitive, domain-specific, with a fixed output format. The 3% accuracy gap vs. a 70B model is negligible; the 100x cost reduction is not.

*Inference after training — deploying on edge hardware:*

#figure(
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the fine-tuned SLM (quantized to 4-bit for Raspberry Pi deployment)
model = AutoModelForCausalLM.from_pretrained("./iot-anomaly-classifier-4bit")
tokenizer = AutoTokenizer.from_pretrained("./iot-anomaly-classifier-4bit")

# Real-time sensor reading classification
sensor_reading = """Sensor: VIB-042 (Bearing vibration, Pump #7)
Timestamp: 2025-10-15T14:23:00Z
Reading: 7.2 mm/s RMS
Baseline: 2.1 mm/s RMS
Trend: Increasing 0.3 mm/s/day over past 5 days
Temperature: 82°C (baseline: 65°C)"""

prompt = f"""Below is an instruction that describes a task, paired with an input.

### Instruction:
Classify the following sensor reading and explain your reasoning.

### Input:
{sensor_reading}

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.2)
classification = tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:")[-1].strip()
print(classification)
# Expected: **Classification: CRITICAL** with ISO 10816-3 references and maintenance recommendation
```
, caption: [Fine-tuned SLM inference on edge hardware for real-time sensor classification],
)

=== 16.6 Why SLMs Fit Agent-Based Systems

Agent tasks are usually: non-chatty, repetitive, format-restricted (JSON, tool calls, structured outputs). Agents already control the model tightly using prompts, tools, and logic - so most of an LLM's "general intelligence" is wasted. A task-specialized SLM is a better match.

*Economics:* 10-30x cheaper inference than 70B-175B LLMs, much lower latency and energy usage, easy fine-tuning with LoRA/QLoRA, can run locally or on-device for better privacy. Recommended strategy: SLM-first, call LLM only when really needed.

=== 16.7 Where SLMs Fall Short

Despite their advantages in cost and latency, SLMs have clear limitations that must be understood when choosing between SLM and LLM deployment:

- *Multi-step reasoning:* SLMs struggle significantly with problems requiring 5+ reasoning steps. Tasks like multi-hop question answering, complex math word problems, or chained logical deductions expose the capacity gap between small and large models. The reasoning capability scales roughly with parameter count, and sub-10B models hit a wall on deeply sequential reasoning.

- *Long-range dependencies:* Performance degrades significantly beyond the training context length for smaller models. Even when the context window technically supports longer inputs, SLMs lose track of information introduced early in the context more readily than LLMs. This makes them unreliable for tasks like summarizing long documents or maintaining coherence across extended conversations.

- *Instruction following on novel formats:* SLMs are less robust when prompted with unfamiliar instruction templates. They tend to overfit to the instruction formats seen during fine-tuning, meaning a slight rephrasing or a new output schema can cause failures. LLMs generalize across instruction styles much more reliably due to broader training data and greater capacity.

- *Structured output adherence:* SLMs are notoriously unreliable at generating valid JSON, XML, or other strict schemas without assistance. In production, *constrained decoding* is essential — libraries like #link("https://github.com/dottxt-ai/outlines")[Outlines], #link("https://github.com/jxnl/instructor")[Instructor], or vLLM's guided decoding constrain the model's token sampling to only produce outputs that conform to a given schema. Fine-tuning teaches the model _what_ to output; constrained decoding guarantees the output is _structurally valid_.

- *Knowledge breadth:* SLMs have smaller internal knowledge bases, leading to more hallucinations on niche or specialized topics. They are best suited for domains covered thoroughly in their fine-tuning data - outside that domain, factual accuracy drops steeply compared to larger models.

- *The takeaway:* SLMs are ideal for well-defined, repetitive tasks in agent pipelines - classification, extraction, formatting, routing - but should be paired with LLMs for complex reasoning, novel tasks, and quality-critical outputs. The recommended architecture is SLM-first with LLM fallback: route straightforward requests to the SLM and escalate to an LLM when the task requires broader knowledge, multi-step reasoning, or flexible instruction interpretation.

=== 16.8 Section Summary

Small Language Models (0.5B-10B parameters) offer a cost-effective, fast, deployable alternative to LLMs for specific tasks. Nvidia's research demonstrates they are the future of agentic AI. Fine-tuning follows the same pipeline as LLMs but with dramatically lower resource requirements.
]
