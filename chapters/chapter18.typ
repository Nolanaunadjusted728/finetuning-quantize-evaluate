#import "../template.typ": *

#let Chapter18() = [
== 18 - Embedding Fine-Tuning

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-24-Embedding-and-Embedding-Finetuning/Embedding_FT.ipynb")[`Embedding_FT.ipynb`]
]

=== 18.1 Embeddings: Static vs. Contextual

*Static embeddings* (#link("https://arxiv.org/abs/1301.3781")[Word2Vec], #link("https://arxiv.org/abs/1607.04606")[FastText]): Each word always gets the same vector, regardless of context.

*Problem:* "I sat on a river *bank*" and "I deposited money in the *bank*" - "bank" gets the same embedding despite having completely different meanings.

*Contextual embeddings* (#link("https://arxiv.org/abs/1908.10084")[SBERT/Sentence Transformers]): The embedding of each word depends on its surrounding context. "Bank" gets different vectors in different sentences. These are transformer-based models.

=== 18.2 Embedding Model Training Pipeline

Embedding models follow a three-stage training process similar to LLMs. A pre-trained transformer is first fine-tuned with contrastive learning on large-scale sentence pairs, then further fine-tuned on domain-specific data for your use case.

#figure(
  image("../diagrams/17-embedding-training-pipeline.png", width: 65%),
  caption: [Embedding model three-stage training pipeline: pre-train, contrastive, custom],
)

*Example:* Microsoft's MPNet (base model) → all-mpnet-base-v2 (fine-tuned on 1B sentence pairs using contrastive learning) → Your custom domain model

#pagebreak(weak: true)

=== 18.3 Contrastive Learning

Contrastive learning teaches the model to produce similar vectors for semantically similar text and distant vectors for dissimilar text. The training data formats include:

*Format A - Sentence Pair with Label:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*sentence_1*], [*sentence_2*], [*label*]),
    ["Metformin treats diabetes"], ["Metformin is a diabetes drug"], [1.0 (similar)],
    ["Metformin treats diabetes"], ["Python is a programming language"], [0.0 (dissimilar)],
  ),
  caption: [Sentence Pair with Label Format],
  kind: table,
)

*Format B - Triplet Format:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*query (anchor)*], [*positive*], [*negative*]),
    ["Sunny is an AI master"], ["Sunny teaches AI"], ["Sunny teaches Java"],
  ),
  caption: [Triplet Training Format],
  kind: table,
)

*Format C - N-way Format:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*query*], [*positive_1, positive_2, ...*], [*negative_1, negative_2, ...*]),
  ),
  caption: [N-way Training Format],
  kind: table,
)

=== 18.4 Loss Functions for Embedding Training

*Cosine Similarity:*

#figure($ "sim"(A, B) = frac(A dot.c B, ||A|| dot.c ||B||) $, caption: [Cosine similarity between embedding vectors], kind: math.equation)

Used to compute similarity between embeddings. Values range from -1 (opposite) to 1 (identical).

*Triplet Loss:*

#figure($ cal(L) = max(0, "sim"(q, n) - "sim"(q, p) + m) $, caption: [Triplet loss for embedding training], kind: math.equation)

Where q = query, p = positive document, n = negative document, m = margin hyperparameter. Pushes positive pairs closer and negative pairs farther apart.

*InfoNCE / Contrastive Loss:*

#figure($ cal(L) = -log frac(exp("sim"(q, p) / tau), sum_(i) exp("sim"(q, d_i) / tau)) $, caption: [InfoNCE contrastive loss for embedding training], kind: math.equation)

Where τ is a temperature parameter. This is the most commonly used loss for modern embedding training.

#pagebreak(weak: true)

=== 18.5 Dual Encoder vs. Cross Encoder

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Architecture*], [*How It Works*], [*Speed*], [*Use Case*]),
    [*Dual Encoder*], [Query and document encoded separately; similarity computed between embeddings], [Fast, scalable to millions of documents], [Retrieval, semantic search],
    [*Cross Encoder*], [Query and document concatenated and passed together through encoder], [Slower but more accurate], [Re-ranking],
  ),
  caption: [Dual Encoder vs Cross Encoder],
  kind: table,
)

#figure(
  image("../diagrams/18-dual-vs-cross-encoder.png", width: 90%),
  caption: [Dual encoder vs cross encoder architecture comparison],
)

=== 18.6 Why Fine-Tune Embeddings?

General-purpose embedding models often fail on domain-specific vocabulary because they were not trained on specialized terminology. In a RAG pipeline, poor embeddings cause the entire retrieval chain to break down.

In a RAG pipeline:

#figure(
  image("../diagrams/19-rag-pipeline.png", width: 90%),
  caption: [RAG pipeline showing where embedding quality impacts retrieval],
)

If the embedding model doesn't understand domain-specific terminology (e.g., pharmaceutical terms), semantic search fails → irrelevant context is retrieved → LLM produces poor answers. Fine-tuning the embedding model on domain data can yield *substantial retrieval improvements* — the #link("https://arxiv.org/abs/2309.07597")[BGE] and #link("https://arxiv.org/abs/2401.00368")[E5-Mistral] papers report large gains on domain-specific retrieval benchmarks, though the exact improvement depends heavily on the domain, baseline model quality, dataset size, and evaluation metric.

#figure(
  image("../diagrams/10-embedding-retrieval.png", width: 85%),
  caption: [Retrieval: Generic vs Fine-Tuned Embeddings],
)

#pagebreak(weak: true)

=== 18.7 Practical Implementation

The Sentence Transformers library provides a high-level training API for embedding fine-tuning. The workflow loads a pre-trained embedding model, pairs it with a contrastive loss function (such as `MultipleNegativesRankingLoss`), and trains on anchor-positive text pairs from your domain.

#figure(
```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from datasets import load_from_disk

# Load pre-trained embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Load custom training data (anchor + positive pairs)
train_dataset = load_from_disk("./train_pairs")

# Initialize loss function
loss = MultipleNegativesRankingLoss(model)

# Training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="pharma-embedding-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

# Train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()
model.save("pharma-embedding-finetuned")
```
, caption: [Sentence Transformers embedding fine-tuning with MultipleNegativesRankingLoss],
)

=== 18.8 Embedding Model Leaderboard

*MTEB (Massive Text Embedding Benchmark)* - The authoritative leaderboard for embedding models. Available at Hugging Face. Allows filtering by:

- Performance metrics
- Model size
- Task type (retrieval, classification, clustering, etc.)
- Language
- Modality (text, image, etc.)

*Top providers:* Google (KALM), Meta, Alibaba (BGE), Microsoft (MiniLM), OpenAI, Cohere, AWS (Titan)

=== 18.9 Worked Example: Legal Document Retrieval for RAG

*Scenario:* A law firm's RAG system retrieves relevant case law and statutes when lawyers ask questions. The generic embedding model (`all-mpnet-base-v2`) fails on legal queries because it doesn't understand legal terminology - "consideration" means something entirely different in contract law vs. everyday English.

*The retrieval failure (before fine-tuning):*

#figure(
```
Query: "What constitutes adequate consideration in a unilateral contract?"

Top 3 retrieved chunks (cosine similarity):
1. "When considering the terms of any agreement, both parties should..."  (sim: 0.72)
   ← Wrong! This is about "considering" (thinking about), not legal "consideration"
2. "Adequate preparation involves reviewing all contract documents..."   (sim: 0.68)
   ← Wrong! Matched on "adequate" + "contract" but irrelevant
3. "The court held that mutual consideration requires..."               (sim: 0.65)
   ← Correct, but ranked too low

LLM answer based on these chunks: Poor - the most relevant chunk was ranked 3rd
```
, caption: [Generic embedding retrieval failure: legal queries mismatched to wrong documents],
)

#pagebreak(weak: true)

*Training data (triplet format for contrastive learning):*

#figure(
```json
[
  {
    "anchor": "What constitutes adequate consideration in a unilateral contract?",
    "positive": "In Carlill v Carbolic Smoke Ball Co [1893], the court established that performance of the requested act constitutes valid consideration in a unilateral contract. Consideration need not flow directly to the promisor - a detriment to the promisee is sufficient. The traditional rule from Chappell & Co v Nestle [1960] confirms that consideration must be sufficient but need not be adequate.",
    "negative": "When considering whether to enter a unilateral agreement, parties should weigh the costs and benefits carefully. It is advisable to consider all terms before signing any contract."
  },
  {
    "anchor": "Does the parol evidence rule apply to partially integrated agreements?",
    "positive": "Under the parol evidence rule, extrinsic evidence is inadmissible to contradict the terms of a fully integrated written agreement. However, for partially integrated agreements, parol evidence may supplement (but not contradict) the written terms. The UCC §2-202 codifies this distinction, permitting consistent additional terms unless the court finds the writing was intended as a complete and exclusive statement.",
    "negative": "The company's employee handbook integrates all workplace policies into a single document. Verbal agreements made during hiring are not part of the official policy."
  },
  {
    "anchor": "What remedies are available for anticipatory breach?",
    "positive": "When one party repudiates the contract before performance is due (anticipatory breach), the non-breaching party may: (1) treat the contract as breached and sue immediately for damages under Hochster v De La Tour [1853]; (2) wait until the performance date and then sue; or (3) treat the repudiation as an offer to rescind and accept it. UCC §2-610 provides that the aggrieved party may await performance for a commercially reasonable time.",
    "negative": "The company anticipates breaching the 100-unit milestone by Q3. The team is working on remediation plans to address the production shortfall."
  }
]
```
, caption: [Embedding fine-tuning triplet data: legal anchor, positive case law, negative distractor],
)

#pagebreak(weak: true)

*After embedding fine-tuning - same query:*

#figure(
```
Query: "What constitutes adequate consideration in a unilateral contract?"

Top 3 retrieved chunks (cosine similarity):
1. "In Carlill v Carbolic Smoke Ball Co [1893], the court established..."  (sim: 0.91)
   ← Correct! Directly relevant case law on consideration
2. "The court held that mutual consideration requires..."                 (sim: 0.87)
   ← Correct! Related legal precedent
3. "Consideration must be sufficient but need not be adequate..."          (sim: 0.84)
   ← Correct! The key legal principle

LLM answer based on these chunks: Excellent - all three chunks are directly relevant
```
, caption: [Fine-tuned embedding retrieval: legal queries correctly matched to relevant case law],
)

*Similarity score improvement:*

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Query-Document Pair*], [*Before*], [*After*], [*Change*]),
    [Legal query → relevant case law], [0.65], [0.91], [+40%],
    [Legal query → similar-words-but-irrelevant], [0.72], [0.31], [-57%],
    ["Consideration" (legal) → "consideration" (everyday)], [0.78], [0.22], [-72%],
  ),
  caption: [Similarity Score Improvement After Fine-Tuning],
  kind: table,
)

*What changed:* The embedding model learned that "consideration" in a legal context is a specific contractual concept, not the everyday word meaning "thinking about." It learned that "anticipatory breach" is a legal doctrine, not someone anticipating a breach. These domain-specific semantic distinctions are what generic embedding models miss — and why embedding fine-tuning can dramatically improve domain-specific RAG accuracy.

#pagebreak(weak: true)

=== 18.10 Matryoshka Representation Learning (MRL)

#blockquote[
  *Resources:*
  - #link("https://huggingface.co/blog/matryoshka")[HuggingFace Blog: Matryoshka Embeddings]
  - #link("https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/training/matryoshka/README.md")[Sentence Transformers Matryoshka Training Examples]
  - #link("https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-embedding-model-for-rag.ipynb")[`fine-tune-embedding-model-for-rag.ipynb`] — End-to-end Matryoshka embedding fine-tuning for RAG (Phil Schmid / HuggingFace)
]

*#link("https://arxiv.org/abs/2205.13147")[Matryoshka Representation Learning]* (Kusupati et al., 2022) is a training technique that produces embeddings which can be *truncated to smaller dimensions* without significant accuracy loss — like Russian nesting dolls, where each smaller doll is self-contained.

*Why it matters for production:* Standard embedding models produce fixed-dimension vectors (e.g., 1024-dim). Storing millions of these in a vector database is expensive. With MRL, you can truncate to 256 or even 128 dimensions at query time, reducing storage by 4-8× while retaining 90-95% of retrieval accuracy.

*How it works:* During training, the loss function is computed at multiple dimensionality checkpoints (e.g., 64, 128, 256, 512, 1024). The model learns to front-load the most important information into the first dimensions. Sentence Transformers provides a built-in `MatryoshkaLoss` wrapper that handles this automatically:

#figure(
  image("../diagrams/11-matryoshka-embedding.png", width: 100%),
  caption: [Matryoshka Embedding: Dimension Truncation],
)

#figure(
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

# Load base embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Wrap the base loss with MatryoshkaLoss
base_loss = MultipleNegativesRankingLoss(model)
matryoshka_loss = MatryoshkaLoss(
    model,
    loss=base_loss,
    matryoshka_dims=[768, 512, 256, 128, 64],  # Truncation checkpoints
)

# Train as usual — MatryoshkaLoss computes the base loss at each dimension
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=matryoshka_loss,
)
trainer.train()
```
, caption: [Matryoshka embedding training with MatryoshkaLoss at multiple dimension checkpoints],
)

#pagebreak(weak: true)

*Testing the trained model — retrieval at multiple dimensions:*

#figure(
```python
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load trained Matryoshka model
model = SentenceTransformer("path/to/trained-matryoshka-model")

# Define query and corpus
query = "What constitutes adequate consideration in a unilateral contract?"
corpus = [
    "In Carlill v Carbolic Smoke Ball Co [1893], the court established that performance "
    "of the requested act constitutes valid consideration in a unilateral contract.",
    "When considering whether to enter a unilateral agreement, parties should weigh "
    "the costs and benefits carefully.",
    "The doctrine of promissory estoppel may substitute for consideration in certain cases.",
]

# Encode at full dimensions
query_embedding = model.encode(query)
corpus_embeddings = model.encode(corpus)

# Compare retrieval quality at different truncation levels
for dim in [768, 256, 128, 64]:
    # Truncate and normalize
    q = query_embedding[:dim]
    c = corpus_embeddings[:, :dim]
    similarities = cos_sim([q], c)[0]
    top_idx = similarities.argmax().item()
    print(f"dim={dim:>4d} | top match: doc {top_idx} (sim={similarities[top_idx]:.4f})")

# Expected output (MRL-trained model):
# dim= 768 | top match: doc 0 (sim=0.9142)
# dim= 256 | top match: doc 0 (sim=0.8987)  ← only ~1.7% drop
# dim= 128 | top match: doc 0 (sim=0.8831)  ← still correct ranking
# dim=  64 | top match: doc 0 (sim=0.8544)  ← 4x smaller, still works
```
, caption: [Matryoshka model evaluation: comparing retrieval quality at truncated dimensions],
)

*Key takeaway:* With MRL training, truncating from 768 to 256 dimensions (4× storage reduction) typically loses only 1-3% retrieval accuracy. Without MRL training, the same truncation can drop accuracy by 10-20%.

*Practical usage:* Models like `nomic-embed-text-v1.5`, `mxbai-embed-large-v1`, and `snowflake-arctic-embed` support MRL natively. When using these models, you can specify the desired dimensionality at query time — smaller dimensions for fast approximate search, full dimensions for precision-critical retrieval.

*2D Matryoshka (advanced):* Sentence Transformers also supports #link("https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/training/matryoshka/2d_matryoshka_nli.py")[`Matryoshka2dLoss`], which reduces both embedding dimensions _and_ the number of transformer layers used. This enables even faster inference by skipping later layers entirely for simple queries.

=== 18.11 Section Summary

Embedding fine-tuning uses contrastive learning to adapt a pre-trained embedding model for domain-specific semantic search. The model learns from anchor-positive(-negative) pairs to produce semantically meaningful vector representations, substantially improving domain-specific RAG retrieval quality. For faster embedding fine-tuning with LoRA/QLoRA support, see Unsloth's embedding integration in Section 12.11.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
