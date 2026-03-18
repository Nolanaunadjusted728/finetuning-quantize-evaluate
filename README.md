# Fine-Tune, Quantize, Evaluate: The Complete Guide

**Covering Large Language Models, Vision-Language Models, and Embedding Models — Theory to Practice**

[![Deploy Typst PDF to Pages](https://github.com/di37/finetuning-quantize-evaluate/actions/workflows/deploy.yml/badge.svg)](https://github.com/di37/finetuning-quantize-evaluate/actions/workflows/deploy.yml)

> **[Read the full PDF →](https://di37.github.io/finetuning-quantize-evaluate/)**

A single, self-contained reference that covers five pillars of modern AI model development — from theory to practice:

1. **Fine-Tuning** — LLMs (SFT, LoRA/QLoRA, DPO/GRPO/RLHF, domain adaptation, instruction tuning), Vision-Language Models (frozen encoder + projection layer + LoRA LLM), and Embedding Models (contrastive learning, triplet/InfoNCE loss, Matryoshka representations, Sentence Transformers)
2. **Quantization** — GPTQ Hessian-based quantization, AWQ activation-aware scaling, QAT straight-through estimator, GGUF/llama.cpp for CPU inference, symmetric vs asymmetric quantization math
3. **Evaluation** — Human evaluation with inter-rater agreement (Cohen's/Fleiss' Kappa), rule-based metrics (BLEU/ROUGE/METEOR/BERTScore), LLM-as-a-Judge (pointwise + pairwise with bias mitigation), factuality decomposition, and standard benchmarks via lm-evaluation-harness (MMLU, AIME, SWE-Bench, HarmBench)
4. **Embedding Benchmarking** — Private benchmark creation, multi-model embedding (Sentence Transformers, Qwen3, OpenAI, Gemini, llama.cpp), ranking metrics (MRR, Recall@K, NDCG@K) with statistical significance testing (Ranx + Fisher's randomization test), and multilingual evaluation with t-SNE visualization
5. **Knowledge Distillation** — Temperature scaling, dark knowledge transfer, logit-based and feature-based distillation from large teacher models to deployable students

## By the Numbers

| Metric | Count |
|--------|-------|
| Sections | 20 |
| Lines of Typst source | 6,300+ |
| Runnable Python code snippets | 86 |
| Diagrams | 23 |
| Tables | 78 |
| Mathematical derivations | 24 |
| Glossary terms | 45+ |
| arXiv paper references | 50+ |

## Table of Contents

| # | Section | Topics |
|---|---------|--------|
| 1 | The LLM Training Pipeline | Three stages (pre-training → SFT → RLHF), full vs. partial vs. PEFT fine-tuning, LoRA/QLoRA/DoRA, data formats |
| 2 | Why Fine-Tuning Was Hard Before Transformers | LSTM limitations, vanishing gradients, attention mechanism breakthrough |
| 3 | HuggingFace Ecosystem | Transformers, Datasets, Tokenizers, PEFT, TRL, Accelerate, BitsAndBytes |
| 4 | LLM Evaluation | Human eval (Cohen's/Fleiss' Kappa), BLEU/ROUGE/METEOR/BERTScore, LLM-as-a-Judge, factuality decomposition, lm-evaluation-harness benchmarks |
| 5 | BERT Fine-Tuning | Question answering, extractive QA, span prediction, tokenizer handling |
| 6 | Non-Instructional Fine-Tuning | Domain adaptation, plain-text training, continued pre-training |
| 7 | Instruction Fine-Tuning | Chat templates, SFTTrainer, dataset preparation, Alpaca/ShareGPT formats |
| 8 | Preference Alignment with DPO | DPO vs. RLHF, contrastive loss, GRPO group-relative optimization, chosen/rejected pair construction |
| 9 | LLM Quantization | GPTQ, AWQ, QAT, GGUF/llama.cpp, symmetric vs asymmetric math, BitsAndBytes 4-bit/8-bit |
| 10 | Knowledge Distillation | Temperature scaling, soft targets, logit-based and feature-based distillation |
| 11 | LLaMA Factory Framework | Web UI, YAML configs, dataset registration, multi-stage training |
| 12 | Unsloth Framework | 2× faster fine-tuning, memory optimization, Unsloth Zoo, 4-bit quantized training |
| 13 | Axolotl Framework | YAML-driven training, multi-dataset mixing, DeepSpeed integration |
| 14 | OpenAI API Fine-Tuning | API-based fine-tuning, JSONL format, hyperparameter tuning, model deployment |
| 15 | Google Vertex AI / Gemini Fine-Tuning | Vertex AI pipelines, Gemini model tuning, GCP integration |
| 16 | Small Language Model (SLM) Fine-Tuning | Phi, Qwen, Gemma, efficient deployment, edge-device considerations |
| 17 | Multimodal Fine-Tuning | Vision-Language Models, frozen encoder + projection layer, image-text training |
| 18 | Embedding Fine-Tuning | Sentence Transformers, contrastive learning, triplet loss, InfoNCE, Matryoshka representations |
| 19 | Embedding Evaluation & Benchmarking | Private benchmarks, MRR/Recall@K/NDCG@K, Ranx, statistical significance, multilingual eval, t-SNE |
| 20 | All-in-One Fine-Tuning Pipeline | End-to-end crash course tying all sections together |

## What Makes This Different

- **86 runnable code snippets** — not pseudocode. Every concept has working Python you can copy into a notebook.
- **Five model types in one document** — LLMs, BERT-family models, Small Language Models, Vision-Language Models, and Embedding Models each get dedicated sections with training and inference code.
- **Quantization covered with actual math** — GPTQ Hessian, AWQ activation-aware scaling, symmetric vs asymmetric formulas, QAT straight-through estimator, and GGUF format internals.
- **Evaluation as a first-class topic** — full section with runnable code for human evaluation, rule-based metrics, LLM-as-a-Judge with bias taxonomy, factuality decomposition, and standard benchmarks.
- **Six fine-tuning frameworks compared** — HuggingFace, LLaMA Factory, Unsloth, Axolotl, OpenAI API, and Google Vertex AI.
- **Complete embedding benchmarking pipeline** — text extraction, QA pair generation, multi-model embedding, Ranx ranking evaluation with statistical significance tests, and multilingual evaluation with t-SNE visualization.
- **Key equations with complete symbol definitions** — MRR, Recall@K, NDCG@K, Cohen's Kappa, METEOR, LoRA decomposition, DPO contrastive loss, GRPO group-relative optimization, Factuality Score, and QAT straight-through estimator.

## Diagrams

The `diagrams/` folder contains 23 diagrams (PNG + SVG source) covering:

| Diagram | Description |
|---------|-------------|
| Decision Flowchart | When to fine-tune vs. prompt-engineer vs. RAG |
| LSTM vs Transformer | Architectural comparison and attention mechanism |
| BERT QA | Extractive question answering pipeline |
| Perplexity Monitoring | Training loss and perplexity tracking |
| LoRA Decomposition | Low-rank weight matrix factorization |
| Transformer LoRA Targets | Which attention matrices to target with LoRA |
| LoRA Stacking | Multiple LoRA adapters composition |
| Knowledge Distillation | Teacher-student architecture |
| VLM Architecture | Vision-Language Model (frozen encoder + projector + LLM) |
| Embedding Retrieval | Vector similarity search pipeline |
| Matryoshka Embedding | Nested dimensionality representation |
| Three Stages Training | Pre-training → SFT → RLHF pipeline |
| Fine-tuning Workflow | End-to-end training workflow |
| Data Preparation Pipeline | Dataset formatting and preprocessing |
| Unsloth Tech Stack | Unsloth framework internals |
| Embedding Training Pipeline | Sentence Transformers training flow |
| Dual vs Cross Encoder | Bi-encoder vs cross-encoder trade-offs |
| RAG Pipeline | Retrieval-augmented generation flow |
| Four-Stage Pipeline | Complete model development lifecycle |
| Embedding Benchmark Pipeline | End-to-end benchmarking workflow |
| Retrieval Scoring Metrics | MRR, Recall@K, NDCG@K visual explanation |
| Multilingual Embedding Eval | Cross-lingual evaluation with t-SNE |
| RLHF Three-Stage Pipeline | Reward model → PPO → alignment |

## Scope

These notes focus on **single-GPU training** workflows (including quantized approaches like QLoRA that make large models trainable on a single GPU). Multi-GPU distributed training (FSDP, DeepSpeed ZeRO, tensor/pipeline parallelism) is not covered.

## Credits

- **[Sunny Savita](https://github.com/sunnysavita10)** — His [Complete-LLM-Finetuning](https://github.com/sunnysavita10/Complete-LLM-Finetuning) repository (20+ notebooks and config files) is the backbone of these notes.
- **[Imad Saddik](https://www.youtube.com/watch?v=7G9q_5q82hY)** — Section 19 on embedding benchmarking is based on his course, with source code on [GitHub](https://github.com/ImadSaddik/Benchmark_Embedding_Models).
- Additional content draws from Stanford CME295 (evaluation methodology), HuggingFace cookbooks, and primary research papers.

## Building from Source

The document is written in [Typst](https://typst.app/). To compile locally:

```bash
typst compile finetune-quantize-evaluate.typ finetune-quantize-evaluate.pdf
```

The PDF is also automatically compiled and deployed to GitHub Pages on every push to `main`.

## License

This project is licensed under the [MIT License](LICENSE).
