#import "../template.typ": *

#let Chapter1() = [

== 1 - The LLM Training Pipeline

=== 1.1 Three Stages of LLM Training

Every modern large language model goes through three sequential training stages:

#figure(
  image("../diagrams/12-three-stages-training.png", width: 65%),
  caption: [Three Stages of LLM Training],
)

==== Stage 1: Unsupervised Pre-training

*What it is:* Training a model on massive amounts of unlabeled text data from the internet - documentation, research papers, Common Crawl, Wikipedia, encyclopedias, books in multiple languages.

*Why it is called "self-supervised":* There are no human-provided labels. The label is the next token itself within the text - the model learns to predict the next word given all previous words.

*Objective:* Next-token prediction (language modeling). Because this is performed at enormous scale, it is called _large language modeling_ - this is where the term "LLM" originates.

*Result:* A *base model* (e.g., Llama base, Mistral base, GPT base, DeepSeek base, Gemini base). Base models understand language patterns and have general knowledge, but they cannot follow instructions, maintain conversational tone, or produce structured answers.

*Infrastructure requirements:* Massive Graphics Processing Unit (GPU) clusters, trillions of tokens of data, weeks to months of training. This stage is a bottleneck - only well-resourced companies (Meta, OpenAI, Google, DeepSeek) can afford it.

*Practical implication:* As practitioners, we do not perform pre-training from scratch. We take existing base models and fine-tune them.

#pagebreak()

==== Stage 2: Supervised Fine-Tuning (SFT)

SFT can be analyzed along two dimensions:

*A. Parameter Level - How many parameters we train:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Method*], [*Description*], [*Requirements*]),
    [*Full Fine-tuning*], [Train ALL parameters (weights and biases)], [Huge GPU memory, multi-GPU setup],
    [*Partial Fine-tuning (Old School)*], [Freeze all layers and train only the last output layer; OR freeze starting layers and retrain later layers], [Used in Convolutional Neural Network (CNN)-based architectures and early LLMs (BERT, T5, BART)],
    [*PEFT (Parameter-Efficient Fine-Tuning)*], [Train only a small subset of parameters using specialized techniques], [Can work on a single GPU with smaller Video Random Access Memory (VRAM)],
  ),
  caption: [Parameter-Level Fine-Tuning Methods],
  kind: table,
)

PEFT Techniques:

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Technique*], [*Full Name*], [*Description*]),
    [*#link("https://arxiv.org/abs/2106.09685")[LoRA]*], [Low-Rank Adaptation], [The foundational PEFT technique],
    [*#link("https://arxiv.org/abs/2305.14314")[QLoRA]*], [Quantized LoRA], [LoRA applied to quantized models for memory-efficient loading],
    [*#link("https://arxiv.org/abs/2402.09353")[DoRA]*], [Weight-Decomposed Low-Rank Adaptation], [An improvement on LoRA],
    [*#link("https://arxiv.org/abs/1902.00751")[Adapter Layers]*], [-], [Append additional layers within transformer blocks],
    [*#link("https://arxiv.org/abs/2106.10199")[BitFit]*], [Bias-term Fine-Tuning], [Fine-tune only bias terms],
    [*#link("https://arxiv.org/abs/2205.05638")[IA3]*], [Infused Adapter by Inhibiting and Amplifying Inner Activations], [Learns rescaling vectors for key, value, and Feed-Forward Network (FFN) activations (independent of LoRA)],
    [*#link("https://arxiv.org/abs/2101.00190")[Prefix Tuning]*], [-], [Prepend trainable vectors to hidden states at every transformer layer],
    [*#link("https://arxiv.org/abs/2104.08691")[Prompt Tuning]*], [-], [Learn soft prompt embeddings],
  ),
  caption: [PEFT Techniques Overview],
  kind: table,
)

*B. Data Level - How we prepare the training data:*

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Type*], [*Data Format*], [*Purpose*], [*Output Style*]),
    [*Non-instructional Fine-tuning*], [Plain text (PDFs, documents, txt files)], [Domain adaptation - teach the model domain-specific language, terminology, vocabulary], [Produces text continuously; not necessarily capable of following instructions],
    [*Instructional Fine-tuning*], [Input/output pairs (instruction + response)], [Teach the model to follow instructions and generate structured answers], [Direct, helpful, structured answers],
  ),
  caption: [Data-Level Fine-Tuning Types],
  kind: table,
)

*Why non-instructional fine-tuning matters:* Before teaching a model to answer questions about a domain, the model should first understand the domain's language. Most YouTube tutorials skip this step. The recommended workflow is:

#figure(
  image("../diagrams/13-finetuning-workflow.png", width: 65%),
  caption: [Recommended Fine-Tuning Workflow],
)

*How ChatGPT was built:* OpenAI took the GPT base model, performed instruction fine-tuning on demonstrations written by human labelers (contractors who wrote high-quality prompt-response pairs), then aligned with RLHF using human preference rankings.

==== Stage 3: Preference-Based Alignment

*Purpose:* Align model responses with human preferences - making the model polite, safe, helpful, and aligned with human values.

*Data format:* Pairs of responses ranked by humans (or AI) - chosen vs. rejected responses for a given prompt.

*Two main techniques:*

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Technique*], [*Full Name*], [*Algorithm*], [*Type*]),
    [*RLHF*], [Reinforcement Learning from Human Feedback], [PPO (Proximal Policy Optimization)], [Reinforcement learning],
    [*DPO*], [Direct Preference Optimization], [Contrastive loss over preference pairs], [Preference optimization],
    [*RLAIF*], [Reinforcement Learning from AI Feedback], [Same as RLHF but with AI-generated labels], [Reinforcement learning],
  ),
  caption: [Preference Alignment Techniques],
  kind: table,
)

DPO is the currently preferred technique - it is simpler to implement and does not require a separate reward model.

=== 1.2 Family-Wise Breakdown

#figure(
  table(
    columns: 5,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Model Family*], [*Pre-trained*], [*SFT / Instruct*], [*Preference Aligned*], [*Access*]),
    [*Llama* (Meta)], [Llama 4 Scout/Maverick (base)], [Llama 4 Instruct], [Chat variants with RLHF], [Open weights (Mixture of Experts (MoE), natively multimodal)],
    [*GPT* (OpenAI)], [GPT-3 (2020)], [InstructGPT (2022)], [GPT-5.x with RLHF + safety filters], [Closed source (Application Programming Interface (API) only)],
    [*Mistral*], [Mistral 3 (3B–14B dense), Mistral Large 3 (675B MoE)], [Instruct versions], [Models with RLHF/DPO], [Open weights (Apache 2.0)],
    [*DeepSeek*], [Base models in all variants], [DeepSeek Coder (code SFT)], [R1, V3.2 (aligned)], [Open weights],
    [*Gemini* (Google)], [Gemini 3 Pro/Flash (base)], [SFT variants], [Aligned variants], [API-based],
  ),
  caption: [LLM Family-Wise Training Breakdown],
  kind: table,
)

=== 1.3 Pros and Cons of Fine-Tuning

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Aspect*], [*Advantages*], [*Disadvantages*]),
    [Task Accuracy], [Significantly improves performance on domain-specific tasks], [Risk of overfitting on small datasets],
    [Domain Adaptation], [Teaches model specialized vocabulary and domain knowledge], [Requires high-quality, curated training data],
    [Brand Customization], [Controls tone, style, and response format to match brand voice], [Catastrophic forgetting - model may lose general capabilities],
    [Cost Efficiency], [Cheaper than pre-training from scratch; pay only for fine-tuning compute], [GPU requirements remain significant (even with PEFT)],
    [Response Control], [Produces predictable, structured outputs for specific use cases], [Hyperparameter tuning is difficult - learning rate, epochs, rank all interact],
  ),
  caption: [Pros and Cons of Fine-Tuning],
  kind: table,
)

#pagebreak(weak: true)

==== 1.3.1 Decision Framework: Fine-Tuning vs. Alternatives

Before investing in a fine-tuning pipeline, evaluate whether the problem actually requires it. Fine-tuning is a powerful tool, but it is not always the right one — and choosing the wrong approach wastes compute, data engineering effort, and calendar time.

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Approach*], [*When to Use*], [*When NOT to Use*], [*Typical Time to Production*]),
    [*Prompt Engineering*], [The base model already has the knowledge; you need better formatting, tone, or structure. Few-shot examples in the prompt solve the problem.], [The model lacks domain knowledge entirely (e.g., proprietary terminology, internal processes).], [Hours],
    [*RAG (Retrieval-Augmented Generation)*], [The model needs access to private/current information. Answers must be grounded in specific documents. Knowledge changes frequently.], [The problem is about _style_ or _behavior_ (e.g., tone, safety), not _knowledge_. Retrieval latency is unacceptable.], [Days to weeks],
    [*Fine-Tuning*], [You need to change the model's _behavior_ — output format, domain language, safety alignment, response style. The task is well-defined and you have 500+ quality examples. RAG + prompt engineering has been tried and is insufficient.], [You have fewer than ~200 high-quality examples. The knowledge changes weekly (fine-tuning is a snapshot). You need the model to cite sources (RAG is better).], [Weeks],
    [*Pre-Training*], [No existing model covers your language or domain at all (e.g., rare language, highly specialized scientific field with no public data).], [Almost always — use a pre-trained base model instead.], [Months],
  ),
  caption: [When to Use Each Approach],
  kind: table,
)

#pagebreak()

*The decision flowchart:*

#figure(
  image("../diagrams/01-decision-flowchart.png", width: 100%),
  caption: [When to Fine-Tune: Decision Flowchart],
)

*Common anti-patterns (when fine-tuning is the wrong choice):*

- *"The model doesn't know about our product"* → RAG is almost always better here. Product information changes; fine-tuned knowledge is frozen at training time.
- *"We want the model to always respond in JSON"* → Try constrained decoding (Outlines, Instructor, vLLM guided decoding) or structured output APIs first. This is cheaper and more reliable than fine-tuning for format compliance alone.
- *"We have 50 examples"* → This is too few for meaningful fine-tuning. Invest in prompt engineering with few-shot examples, or generate synthetic training data to reach 500+ examples before fine-tuning.
- *"We want better accuracy on a benchmark"* → If the benchmark is contaminated or the improvement is marginal, fine-tuning may be overfitting to the test set rather than genuinely improving capability.

#blockquote[
  *Rule of thumb:* Try prompt engineering first (hours), then RAG (days), then fine-tuning (weeks). Each subsequent approach requires more investment but solves problems the previous one cannot. Fine-tuning is the right choice when you've exhausted the cheaper alternatives and need to change the model's fundamental behavior, not just its knowledge.
]

#pagebreak(weak: true)

=== 1.4 Fine-Tuning Framework Comparison

#figure(
  table(
    columns: 5,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Framework*], [*Type*], [*Key Feature*], [*Best For*], [*Difficulty*]),
    [Hugging Face (transformers + peft + trl)], [Library], [Full control, maximum flexibility], [Custom training pipelines, research], [Medium-High],
    [LLaMA Factory], [UI + CLI], [Zero-code fine-tuning via Web UI], [Beginners, rapid prototyping], [Low],
    [Unsloth], [Optimized Engine], [2-3x faster, 50-80% less VRAM], [Resource-constrained training (Colab/Kaggle)], [Low-Medium],
    [Axolotl], [Config-driven CLI], [Single YAML controls entire pipeline], [Production/reproducible experiments, multi-GPU], [Medium],
    [OpenAI API], [Cloud API], [Managed fine-tuning, no GPU needed], [GPT model customization], [Low],
    [Google Vertex AI], [Cloud API], [Managed Gemini fine-tuning], [Gemini model customization], [Low],
    [FastChat], [Library], [Multi-model serving + training], [Model serving and benchmarking], [Medium],
    [DeepSpeed], [Distributed Framework], [ZeRO optimizer, pipeline parallelism], [Multi-GPU/multi-node training], [High],
    [Colossal-AI], [Distributed Framework], [Efficient parallelism strategies], [Large-scale distributed training], [High],
    [vLLM / LightLLM], [Inference Engine], [PagedAttention, high throughput], [Serving fine-tuned models], [Medium],
  ),
  caption: [Fine-Tuning Framework Comparison],
  kind: table,
)

#pagebreak(weak: true)

=== 1.5 Important Research Papers

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Paper*], [*Year*], [*Key Contribution*]),
    [#link("https://arxiv.org/abs/1801.06146")[_ULMFiT_]], [2018], [Introduced transfer learning for Natural Language Processing (NLP) (Howard & Ruder)],
    [#link("https://arxiv.org/abs/1810.04805")[_BERT_]], [2018], [Bidirectional pre-training with masked language modeling (Devlin et al.)],
    [#link("https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf")[_GPT_]], [2018], [Autoregressive pre-training for language generation (Radford et al.)],
    [#link("https://arxiv.org/abs/1910.10683")[_T5_]], [2019], [Text-to-text framework for all NLP tasks (Raffel et al.)],
    [#link("https://arxiv.org/abs/1902.00751")[_Adapters_]], [2019], [Adapter layers for parameter-efficient transfer (Houlsby et al.)],
    [#link("https://arxiv.org/abs/2005.14165")[_GPT-3_]], [2020], [Demonstrated few-shot learning at scale (Brown et al.)],
    [#link("https://arxiv.org/abs/2001.08361")[_Scaling Laws_]], [2020], [Quantified compute/data/model size scaling relationships (Kaplan et al.)],
    [#link("https://arxiv.org/abs/2106.09685")[_LoRA_]], [2021], [Low-rank adaptation for parameter-efficient fine-tuning (Hu et al.)],
    [#link("https://arxiv.org/abs/2101.00190")[_Prefix Tuning_]], [2021], [Prepend trainable vectors to hidden states at every layer (Li & Liang)],
    [#link("https://arxiv.org/abs/2203.02155")[_InstructGPT_]], [2022], [RLHF for alignment (Ouyang et al.)],
    [#link("https://arxiv.org/abs/2212.10560")[_Self-Instruct_]], [2022], [LLM-generated instruction data (Wang et al.)],
    [#link("https://arxiv.org/abs/2305.14314")[_QLoRA_]], [2023], [4-bit quantization + LoRA (Dettmers et al.)],
    [#link("https://arxiv.org/abs/2305.18290")[_DPO_]], [2023], [Direct preference optimization without reward model (Rafailov et al.)],
    [#link("https://arxiv.org/abs/2205.05638")[_IA3_]], [2022], [Few-shot PEFT via learned rescaling vectors on keys, values, and FFN activations (Liu et al.)],
    [#link("https://arxiv.org/abs/2402.09353")[_DoRA_]], [2024], [Weight-decomposed LoRA — separates magnitude and direction for closer full fine-tuning performance (Liu et al.)],
    [#link("https://arxiv.org/abs/2402.03300")[_GRPO_]], [2024], [Group Relative Policy Optimization — eliminates critic model using group-relative advantages (Shao et al., DeepSeek)],
    [#link("https://arxiv.org/abs/2402.01306")[_KTO_]], [2024], [Kahneman-Tversky Optimization — preference alignment from binary (good/bad) signals without paired data (Ethayarajh et al.)],
    [#link("https://arxiv.org/abs/2403.07691")[_ORPO_]], [2024], [Odds Ratio Preference Optimization — combines SFT and preference alignment in a single training phase (Hong et al.)],
    [#link("https://arxiv.org/abs/2205.13147")[_Matryoshka Representation Learning_]], [2022], [Truncatable embeddings with front-loaded information for flexible dimensionality (Kusupati et al.)],
    [#link("https://arxiv.org/abs/2501.12948")[_DeepSeek-R1_]], [2025], [Demonstrated GRPO at scale for reasoning; introduced distillation of reasoning into smaller models (DeepSeek-AI)],
  ),
  caption: [Important Research Papers],
  kind: table,
)

=== 1.6 Section Summary

The LLM training pipeline consists of three stages: (1) unsupervised pre-training for general intelligence, (2) supervised fine-tuning for instruction following and domain adaptation, and (3) preference alignment for human-value alignment. For enterprise use, we skip pre-training and instead take a base model, perform non-instructional fine-tuning for domain knowledge, instructional fine-tuning for structured output, and DPO for preference alignment.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
