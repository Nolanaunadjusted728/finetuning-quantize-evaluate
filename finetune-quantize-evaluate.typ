#import "template.typ": *

#import "chapters/chapter1.typ": Chapter1
#import "chapters/chapter2.typ": Chapter2
#import "chapters/chapter3.typ": Chapter3
#import "chapters/chapter4.typ": Chapter4
#import "chapters/chapter5.typ": Chapter5
#import "chapters/chapter6.typ": Chapter6
#import "chapters/chapter7.typ": Chapter7
#import "chapters/chapter8.typ": Chapter8
#import "chapters/chapter9.typ": Chapter9
#import "chapters/chapter10.typ": Chapter10
#import "chapters/chapter11.typ": Chapter11
#import "chapters/chapter12.typ": Chapter12
#import "chapters/chapter13.typ": Chapter13
#import "chapters/chapter14.typ": Chapter14
#import "chapters/chapter15.typ": Chapter15
#import "chapters/chapter16.typ": Chapter16
#import "chapters/chapter17.typ": Chapter17
#import "chapters/chapter18.typ": Chapter18
#import "chapters/chapter19.typ": Chapter19
#import "chapters/chapter20.typ": Chapter20

#show: project.with(
  title: "Fine-Tune, Quantize, Evaluate: The Complete Guide",
  subtitle: "Covering Large Language Models, Vision-Language Models, and Embedding Models — Theory to Practice",
  author: "Isham Rashik",
  date: "18/03/2026",
)

#blockquote[
  *A single, self-contained reference that covers five pillars of modern AI model development — from theory to practice:*

  + *Fine-Tuning* — LLMs (SFT, LoRA/QLoRA, DPO/GRPO/RLHF, domain adaptation, instruction tuning), Vision-Language Models (frozen encoder + projection layer + LoRA LLM), and Embedding Models (contrastive learning, triplet/InfoNCE loss, Matryoshka representations, Sentence Transformers)
  + *Quantization* — GPTQ Hessian-based quantization, AWQ activation-aware scaling, QAT straight-through estimator, GGUF/llama.cpp for CPU inference, symmetric vs asymmetric quantization math
  + *Evaluation* — Human evaluation with inter-rater agreement (Cohen's/Fleiss' Kappa), rule-based metrics (BLEU/ROUGE/METEOR/BERTScore), LLM-as-a-Judge (pointwise + pairwise with bias mitigation), factuality decomposition, and standard benchmarks via lm-evaluation-harness (MMLU, AIME, SWE-Bench, HarmBench)
  + *Embedding Benchmarking* — Private benchmark creation, multi-model embedding (Sentence Transformers, Qwen3, OpenAI, Gemini, llama.cpp), ranking metrics (MRR, Recall\@K, NDCG\@K) with statistical significance testing (Ranx + Fisher's randomization test), and multilingual evaluation with t-SNE visualization
  + *Knowledge Distillation* — Temperature scaling, dark knowledge transfer, logit-based and feature-based distillation from large teacher models to deployable students

  Spanning 20 deeply annotated sections, 6,300+ lines, 86 runnable Python code snippets, 23 diagrams, 78 tables, and 24 mathematical derivations. This is the resource that means you never need to re-watch a lecture or piece together scattered blog posts again.

  *Scope note:* These notes focus on *single-GPU training* workflows (including quantized approaches like QLoRA that make large models trainable on a single GPU). Multi-GPU distributed training (FSDP, DeepSpeed ZeRO, tensor/pipeline parallelism) is not covered here.
]

#v(1.5em)

*Based on the tutorials by #link("https://github.com/sunnysavita10")[Sunny Savita]* — His #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning")[Complete-LLM-Finetuning] repository, with 20+ notebooks and config files, is the backbone of these notes. Section 19 on embedding benchmarking is based on *#link("https://www.youtube.com/watch?v=7G9q_5q82hY")[Imad Saddik's]* course on benchmarking embedding models on private data, with source code on #link("https://github.com/ImadSaddik/Benchmark_Embedding_Models")[GitHub] and datasets on #link("https://huggingface.co")[Hugging Face]. Additional content draws from Stanford CME295 (evaluation methodology), HuggingFace cookbooks, and primary research papers.

*What makes these notes different:*

- *86 runnable code snippets* — not pseudocode. Every concept has working Python you can copy into a notebook: evaluation metrics, embedding generation, benchmarking pipelines, model training, and quantization workflows
- *Five model types in one document* — LLMs, BERT-family models, Small Language Models, Vision-Language Models, and Embedding Models each get dedicated sections with training and inference code
- *Quantization covered with actual math* — GPTQ Hessian, AWQ activation-aware scaling, symmetric vs asymmetric formulas, QAT straight-through estimator, and GGUF format internals that most resources skip entirely
- *Evaluation as a first-class topic* — not an afterthought. Full section with runnable code for human evaluation (Cohen's/Fleiss' Kappa), rule-based metrics (BLEU/ROUGE/METEOR/BERTScore), LLM-as-a-Judge with bias taxonomy and pairwise comparison, factuality decomposition pipeline, standard benchmarks (MMLU, AIME, SWE-Bench, HarmBench), and end-to-end evaluation functions per fine-tuning stage
- *23 diagrams* covering decision flowcharts, LoRA decomposition, knowledge distillation, VLM architecture, embedding pipelines, retrieval scoring, and multilingual evaluation
- *Complete embedding benchmarking pipeline* — text extraction, QA pair generation, multi-model embedding, Ranx ranking evaluation with statistical significance tests, and multilingual evaluation with t-SNE visualization
- *Six fine-tuning frameworks compared* — HuggingFace, LLaMA Factory, Unsloth, Axolotl, OpenAI API, and Google Vertex AI, each with dedicated sections
- Key equations with complete symbol definitions — MRR, Recall\@K, NDCG\@K, Cohen's Kappa, METEOR, LoRA decomposition, DPO contrastive loss, GRPO group-relative optimization, Factuality Score, and QAT straight-through estimator
- 45+ glossary terms, 50+ arXiv paper references, 78 tables, and a comprehensive framework comparison table

#Chapter1()
#Chapter2()
#Chapter3()
#Chapter4()
#Chapter5()
#Chapter6()
#Chapter7()
#Chapter8()
#Chapter9()
#Chapter10()
#Chapter11()
#Chapter12()
#Chapter13()
#Chapter14()
#Chapter15()
#Chapter16()
#Chapter17()
#Chapter18()
#Chapter19()
#Chapter20()

== Key Takeaways

*Fine-Tuning:*
- The LLM training pipeline has three stages: unsupervised pre-training → supervised fine-tuning → preference alignment. For practitioners, we skip pre-training and start from base models.
- Non-instructional fine-tuning (domain adaptation on plain text) is a critical step most tutorials skip. It teaches the model domain vocabulary before instruction fine-tuning.
- LoRA is the foundational PEFT technique that makes fine-tuning practical on consumer GPUs by training only a small subset of parameters.
- DPO replaces the complex RLHF pipeline with a single supervised loss function for preference alignment.
- *Never stack LoRA adapters* — always merge-and-unload before applying a new adapter.
- Pre-training and SFT both use the *next-token prediction* cross-entropy loss. DPO uses a different contrastive loss over preference pairs, though it still computes per-token log-probabilities internally.

*Frameworks:*
- Unsloth provides 2-3x speedup and 50-80% VRAM reduction with zero accuracy loss through custom CUDA/Triton kernels and optimized operations.
- LLaMA Factory wraps Hugging Face libraries with a UI/CLI interface, supporting SFT, DPO, RLHF with Alpaca, ShareGPT, and DPO data formats.
- Axolotl offers maximum configurability via YAML/Python configs, with first-class Docker support, FSDP/DeepSpeed for multi-GPU, and config-driven DPO.
- The HuggingFace ecosystem (`transformers`, `datasets`, `peft`, `trl`, `evaluate`) is the foundation that all major fine-tuning frameworks build upon.

*Specialized Domains:*
- Small Language Models (\<10B params) are 10-30x cheaper than LLMs and are the future of agentic AI for repetitive, well-defined subtasks.
- Multimodal fine-tuning treats image patches like text tokens — both are embedded, position-encoded, and fed through transformer attention.
- BERT-family models remain the practical choice for classification, NER, and extractive QA — smaller, faster, and sufficient for many production tasks.

*Embeddings & Benchmarking:*
- Embedding fine-tuning through contrastive learning can substantially improve RAG pipeline accuracy for domain-specific applications — the magnitude depends on domain, baseline model, and dataset quality.
- *Public benchmarks are starting points, not final answers* — always create private benchmarks on your own domain data with statistical significance testing (Fisher's randomization test) before selecting an embedding model.
- Multilingual embedding models that cluster by *topic* (not language) enable cross-lingual retrieval; models not trained on your target language fail catastrophically, not gracefully.

*Evaluation & Quantization:*
- LLM evaluation requires a multi-method approach: rule-based metrics (BLEU/ROUGE/BERTScore) for automated baselines, LLM-as-a-Judge with bias mitigation for scalable evaluation, factuality decomposition for verifiable claims, and human evaluation with inter-rater agreement as the gold standard.
- Knowledge distillation transfers "dark knowledge" (soft probability distributions) from large teacher models to small students, sometimes matching or exceeding teacher performance on specific tasks.
- Quantization (GPTQ, AWQ, GGUF) enables running 7B+ models on consumer hardware — `q4_K_M` is the sweet spot for quality vs. size tradeoff.

#line(length: 100%, stroke: 0.5pt + luma(200))

== Glossary

#two-col[
#figure(
  table(
    columns: 2,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Term*], [*Definition*]),
    [*Base Model*], [A model that has only undergone unsupervised pre-training; knows language patterns but cannot follow instructions],
    [*SFT*], [Supervised Fine-Tuning - training on labeled input/output pairs to teach instruction-following],
    [*PEFT*], [Parameter-Efficient Fine-Tuning - techniques that train only a subset of model parameters],
    [*LoRA*], [Low-Rank Adaptation - decomposes weight updates into low-rank matrices, training far fewer parameters],
    [*QLoRA*], [Quantized LoRA - applies LoRA to 4-bit quantized models for maximum memory efficiency],
    [*DoRA*], [Weight-Decomposed Low-Rank Adaptation - a variant of LoRA],
    [*DPO*], [Direct Preference Optimization - a supervised loss function for preference alignment without requiring a reward model],
    [*RLHF*], [Reinforcement Learning from Human Feedback - preference alignment using PPO and a separate reward model],
    [*PPO*], [Proximal Policy Optimization - the RL algorithm used in RLHF],
    [*RLAIF*], [Reinforcement Learning from AI Feedback - replaces human annotators with AI],
    [*Alpaca Format*], [Instruction fine-tuning data format with instruction, input, and output columns],
    [*ShareGPT Format*], [Conversational data format with from/value pairs in a conversations array],
    [*Tokenization*], [Converting text into numerical token IDs that the model can process],
    [*Padding Token*], [A special token used to equalize sequence lengths in a batch],
    [*merge_and_unload()*], [PEFT method that merges LoRA adapter weights into the base model and removes the adapter layer],
    [*Flash Attention*], [IO-aware exact attention mechanism that is faster and more memory efficient],
    [*Triton*], [OpenAI's open GPU kernel language for writing optimized GPU operations],
    [*CUDA*], [Nvidia's C++ framework for GPU computing],
    [*SLM*], [Small Language Model - typically \<10B parameters, deployable on consumer hardware],
    [*Knowledge Distillation*], [Technique to compress a large model into a smaller one that mimics its behavior],
    [*ViT*], [Vision Transformer - processes images by splitting them into patches and using transformer attention],
    [*CLIP*], [Contrastive Language-Image Pre-training - aligns text and image embeddings in a shared space],
    [*Projection Layer*], [A linear layer that maps image patch features into the same embedding space as text tokens],
    [*Contrastive Learning*], [Training paradigm that learns by comparing similar and dissimilar pairs],
    [*Triplet Loss*], [Loss function using anchor, positive, and negative examples],
    [*InfoNCE Loss*], [Contrastive loss that maximizes agreement between positive pairs relative to negative pairs],
    [*MTEB*], [Massive Text Embedding Benchmark - the authoritative leaderboard for embedding models],
    [*Dual Encoder*], [Architecture where query and document are encoded separately; fast and scalable],
    [*Cross Encoder*], [Architecture where query and document are encoded together; more accurate but slower],
    [*RAG*], [Retrieval-Augmented Generation - augments LLM queries with relevant context from a vector database],
    [*Sentence Transformers (SBERT)*], [Library for state-of-the-art contextual sentence/paragraph embeddings],
    [*TRL*], [Transformer Reinforcement Learning - Hugging Face library for SFT, DPO, and RLHF trainers],
    [*Temperature Scaling*], [Dividing logits by a temperature T before softmax to produce softer probability distributions; in knowledge distillation, higher T reveals inter-class similarity ("dark knowledge") that hard labels discard],
    [*GPTQ*], [Generative Pre-Trained Transformer Quantization - quantizes weights layer-by-layer using calibration data],
    [*AWQ*], [Activation-Aware Weight Quantization - preserves precision for weights that interact with large activations],
    [*QAT*], [Quantization-Aware Training - inserts fake quantization during training so the model learns to be robust to quantization noise],
    [*GGUF*], [File format for storing quantized models, used by llama.cpp for CPU inference],
    [*GGML*], [Georgi Gerganov Machine Learning - C-based tensor library and inference engine powering llama.cpp],
    [*llama.cpp*], [C/C++ inference engine for running quantized LLMs on CPU without Python dependencies],
    [*BIO Tagging*], [Named entity labeling scheme: B (beginning), I (inside), O (outside) an entity span],
    [*Axolotl*], [Config-driven fine-tuning framework with Docker support, FSDP, and YAML/Python configuration],
    [*BPE*], [Byte Pair Encoding - subword tokenization algorithm that iteratively merges the most frequent character pairs],
    [*BLEU*], [Bilingual Evaluation Understudy - precision-based metric for evaluating machine translation quality],
    [*ROUGE*], [Recall-Oriented Understudy for Gisting Evaluation - recall-based metric for summarization quality],
    [*Perplexity*], [Measures how "surprised" a language model is by text - lower is better; ranges are model- and dataset-dependent (well-trained LLMs typically achieve 5-15 on in-domain text)],
    [*DataCollatorForLanguageModeling*], [HuggingFace collator that handles padding and sets labels = input_ids for causal language modeling],
    [*NF4*], [NormalFloat 4-bit — quantization type optimized for normally distributed neural network weights, used in QLoRA],
    [*MRR*], [Mean Reciprocal Rank — measures how quickly the first relevant result appears; MRR = 1/rank of correct document],
    [*NDCG\@K*], [Normalized Discounted Cumulative Gain — position-aware relevance metric that weights higher-ranked results more],
    [*Recall\@K*], [Fraction of relevant documents appearing in the top-K results],
    [*Ranx*], [Python library for ranking evaluation and statistical significance testing in information retrieval],
    [*BERTScore*], [Evaluation metric using contextual embeddings (BERT/DeBERTa) to compute semantic similarity between texts],
    [*METEOR*], [Metric for Evaluation of Translation with Explicit ORdering — harmonic mean of precision and recall with fragmentation penalty],
    [*Cohen's Kappa*], [Inter-rater agreement statistic that adjusts for chance; ranges from -1 (systematic disagreement) to 1 (perfect agreement)],
    [*LLM-as-a-Judge*], [Using a strong LLM (GPT-5, Claude, Gemini) to evaluate model outputs — supports pointwise and pairwise modes],
    [*lm-evaluation-harness*], [EleutherAI's standard framework for running LLM benchmarks (MMLU, HellaSwag, ARC) programmatically],
    [*Qrels*], [Query relevance judgments — the ground truth mapping of questions to correct documents in information retrieval],
    [*Fisher's Randomization Test*], [Non-parametric statistical test for comparing ranking systems; determines if score differences are significant],
    [*t-SNE*], [t-distributed Stochastic Neighbor Embedding — dimensionality reduction technique for visualizing high-dimensional data in 2D],
  ),
  caption: [Glossary of Key Terms],
  kind: table,
)
]

#line(length: 100%, stroke: 0.5pt + luma(200))

== Open Questions / Areas for Further Study

- *Transformer Architecture Deep Dive:* The course assumes transformer knowledge. A dedicated study of multi-head self-attention, cross-attention, and the encoder-decoder architecture would provide deeper understanding.
- *GSPO:* Mentioned alongside GRPO as one of the latest policy optimization techniques in Unsloth - less documented than GRPO and warrants further investigation.
- *Diffusion Models for Image Generation:* The DALL-E architecture discussion introduces diffusion models but does not deep-dive into the denoising process.
- *RoPE Scaling:* Mentioned as a technique enabling Unsloth's long context training - warrants further investigation.
- *Cross-Encoder Re-ranking:* The embedding section briefly mentions cross-encoders for re-ranking in RAG pipelines — a full practical implementation with Ranx integration would be valuable.
- *Embedding Model Fine-Tuning with Benchmarking:* Section 19 covers evaluation of existing models — a natural extension is fine-tuning a domain-specific embedding model (using Sentence Transformers `SentenceTransformerTrainer`) and measuring improvement on the private benchmark.
- *Multi-Agent Systems with SLMs:* The Nvidia research paper on SLMs for agentic AI opens questions about optimal model routing, cost-aware agent architectures, and when to use SLM vs LLM in agent pipelines.
- *Advanced Evaluation Pipelines:* Combining LLM-as-a-Judge with human evaluation in a hybrid loop (machine labels → human calibration → active learning) for production evaluation at scale.
- *#link("https://arxiv.org/abs/2305.02301")["Distilling Step-by-Step"]* (Google, ACL 2023): A 770M T5 student outperformed 540B PaLM by distilling the reasoning process - how does this change the cost calculus for production deployments?
- *Quantization Method Selection:* When to use GPTQ vs AWQ vs GGUF in production. Performance benchmarks across different hardware (GPU vs CPU vs Apple Silicon) would clarify deployment decisions.
- *#link("https://arxiv.org/abs/2407.11062")[EfficientQAT]:* Full 2-bit quantization on a single GPU (LLaMA-2 up to 70B) - how does this compare to standard QLoRA in quality and speed?
- *vLLM and llm-compressor:* The successor to `autoawq` - how does it compare for production quantization workflows?

#pagebreak()

#flow-figure("assets/qr-code.png", img-width: 120pt, position: top + right)[
  #v(1em)
  #text(size: 20pt, weight: "bold", fill: rgb("#1e40af"))[Follow me for More AI Content]
  #v(0.6em)
  #text(size: 11pt, fill: luma(80))[If you found these notes useful, connect with me on LinkedIn for more deep dives into Machine Learning, Artificial Intelligence, and Computer Vision.]
  #v(1em)
  #text(size: 12pt, weight: "semibold", fill: rgb("#1e40af"))[#link("https://www.linkedin.com/in/isham-rashik-5a547711b/")[Isham Rashik on LinkedIn]]
  #v(0.4em)
  #text(size: 9pt, fill: luma(160))[Scan the QR code or click the link above]
]
