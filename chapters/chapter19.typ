#import "../template.typ": *

#let Chapter19() = [
== 19 - Embedding Evaluation & Benchmarking

#blockquote[
  *Based on:* #link("https://www.youtube.com/watch?v=7G9q_5q82hY")[Imad Saddik's course] on benchmarking embedding models on private data. Source code on #link("https://github.com/ImadSaddik/Benchmark_Embedding_Models")[GitHub]; datasets hosted on #link("https://huggingface.co")[Hugging Face]. Uses the #link("https://amenra.github.io/ranx/")[Ranx] library for ranking evaluation and statistical testing.
]

This section covers how to *scientifically evaluate and compare embedding models* on your own domain-specific data — moving beyond public leaderboards like MTEB to create private benchmarks with statistical rigor.

=== 19.1 Why Private Benchmarks Matter

Public benchmarks (e.g., the #link("https://huggingface.co/spaces/mteb/leaderboard")[MTEB Leaderboard]) rank models across hundreds of tasks and languages. They provide a useful starting point, but have limitations:

- They may not include your specific domain (e.g., pharmaceutical terminology, legal contracts, or astronomy).
- Models may overfit to public benchmark distributions.
- Your retrieval pipeline has specific characteristics (chunk sizes, query patterns, languages) that generic benchmarks don't capture.

*The solution:* Create a *golden dataset* from your private documents, generate question-answer pairs, embed everything with multiple models, then compare performance using proper metrics and statistical tests.

=== 19.2 The Evaluation Pipeline

The full benchmarking pipeline consists of five stages:

+ *Extract text from documents* — Use VL models (e.g., Gemini 3 Pro/Flash) for complex/scanned PDFs, or Python libraries (PyMuPDF, pdfplumber) for simple documents.

+ *Divide text into chunks* — Each chunk should focus on a specific idea. Use LLMs to produce semantically meaningful chunks (with manual supervision), rather than naive programmatic splitting.

+ *Generate question-answer pairs* — Use an LLM to generate questions answerable by each chunk. This mapping forms the ground truth for evaluation.

+ *Generate embeddings* — Pass both chunks and questions through each candidate model to produce dense vectors.

+ *Benchmark and compare* — Compute retrieval metrics, run statistical significance tests, and generate comparison tables.

#figure(
  image("../diagrams/21-embedding-benchmark-pipeline.png", width: 95%),
  caption: [Embedding benchmark pipeline: five stages from raw documents to statistically validated model rankings],
)

*Step 1 — Text extraction with PyMuPDF vs VL models:*

#figure(
```python
import fitz  # PyMuPDF

doc = fitz.open("report.pdf")
print(f"Pages: {len(doc)}")
text = ""
for page in doc:
    text += page.get_text()
print(f"Extracted {len(text)} characters")
doc.close()
```
, caption: [Text extraction with PyMuPDF — fast but loses structure and cannot handle scanned PDFs],
)

#figure(
```python
from google import genai

client = genai.Client()
sample_file = client.files.upload(file="report.pdf")

system_prompt = """You will receive a PDF document. Extract the entire text page by page.
- Extract selectable text as-is, preserving paragraph structure
- For images: provide a detailed description
- For tables: output in Markdown format
- For scanned pages: use OCR to extract all visible text"""

response = client.models.generate_content(
    model="gemini-3-flash",
    contents=[system_prompt, sample_file],
    config={"max_output_tokens": 32000},
)
extracted_text = response.text
```
, caption: [Text extraction with Gemini — preserves structure and handles scanned/complex documents],
)

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Criterion*], [*Python Libraries*], [*VL Models*]),
    [Cost], [Free], [Free / Paid],
    [Scanned input], [No], [Yes],
    [Resources], [Low], [High (local) / Low (API)],
    [Preserve structure], [No], [Yes],
    [Handle complex layouts], [No], [Yes],
    [Understand content], [No], [Yes],
    [Speed], [Fast], [Slower],
  ),
  caption: [Text extraction: Python libraries vs vision language models],
  kind: table,
)

*Step 3 — Generating question-answer pairs with an LLM:*

#figure(
```python
import json
from google import genai

client = genai.Client()

system_prompt = """Given the following text chunk, generate questions that can be
answered ONLY by this chunk. Output valid JSON:
{"questions": ["question1", "question2", ...]}
Rules:
- Questions must be specific and answerable from the chunk alone
- Generate 3-8 questions per chunk depending on content density
- Avoid yes/no questions — prefer 'what', 'how', 'why' questions"""

def generate_questions(chunk_text: str) -> list[str]:
    response = client.models.generate_content(
        model="gemini-3-flash",
        contents=[system_prompt, f"Text chunk:\n{chunk_text}"],
        config={"max_output_tokens": 2000},
    )
    return json.loads(response.text)["questions"]

# Example: generate from first chunk
questions = generate_questions(chunks[0]["text"])
print(f"Generated {len(questions)} questions from chunk 0")
for q in questions:
    print(f"  - {q}")
```
, caption: [Generating question-answer pairs from text chunks using an LLM],
)

#pagebreak(weak: true)

*Step 4 — Generating embeddings with multiple models:*

The goal is to embed all chunks and questions with every candidate model, storing results in a unified structure for comparison.

*Open-source models (Sentence Transformers):*

#figure(
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

chunk_texts = [c["text"] for c in chunks]
question_texts = [q["question"] for q in questions]

chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True)
question_embeddings = model.encode(question_texts, show_progress_bar=True)

print(f"Chunk embeddings shape: {chunk_embeddings.shape}")    # (43, 384)
print(f"Question embeddings shape: {question_embeddings.shape}")  # (377, 384)
```
, caption: [Embedding text with Sentence Transformers — all-MiniLM-L6-v2 embeds 400+ texts in under 1 second],
)

*Prompted embedding models (Qwen3-Embedding):*

#figure(
```python
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")

# Qwen models support instruction prompts for better retrieval
query_prompt = "Given a web search query, retrieve relevant passages that answer the query"

chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True)
question_embeddings = model.encode(
    question_texts,
    prompt=query_prompt,
    show_progress_bar=True,
)
print(f"Qwen embedding dim: {chunk_embeddings.shape[1]}")  # 1024
```
, caption: [Prompted embeddings with Qwen3 — instruction prefix improves retrieval quality],
)

#pagebreak(weak: true)

*Proprietary models (Google Gemini):*

#figure(
```python
from google import genai
import time

client = genai.Client()

def embed_text(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return result.embeddings[0].values

# Embed chunks with rate limiting for free tier
chunk_embeddings = []
for chunk in chunks:
    embedding = embed_text(chunk["text"])
    chunk["embeddings"]["gemini-embedding-001"] = embedding
    chunk_embeddings.append(embedding)
    time.sleep(1)  # respect free-tier rate limits (remove for paid tier)

print(f"Gemini embedding dim: {len(chunk_embeddings[0])}")  # 3072
```
, caption: [Embedding with Google Gemini API — 3072-dim vectors with free-tier access],
)

*Proprietary models (OpenAI):*

#figure(
```python
from openai import OpenAI

client = OpenAI()

def embed_text_openai(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

embedding = embed_text_openai("What is quantization in LLMs?")
print(f"OpenAI small dim: {len(embedding)}")  # 1536

# Cost: ~$0.02 per 1M tokens — embedding 23K tokens costs ~$0.0005
```
, caption: [Embedding with OpenAI API — text-embedding-3-small at \$0.02/1M tokens],
)

*Running large models locally via llama.cpp:*

#figure(
```bash
# Download GGUF from huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF
# Start embedding server with quantized model
llama-server -m Qwen3-Embedding-4B-Q4_K_M.gguf \
  --embedding --pooling cls \
  -ngl 18 -c 6144 --port 8080
```
, caption: [Starting a local embedding server with llama.cpp — use `-ngl` to control GPU layer offloading],
)

#figure(
```python
import requests

def embed_local(text: str) -> list[float]:
    response = requests.post(
        "http://localhost:8080/v1/embeddings",
        json={"input": text},
    )
    return response.json()["data"][0]["embedding"]

embedding = embed_local("What is LoRA fine-tuning?")
print(f"Qwen3-4B dim: {len(embedding)}")  # 2560
```
, caption: [Calling the local llama.cpp embedding server — OpenAI-compatible API],
)

*Unified storage format — single JSON with all models:*

#figure(
```python
import json

data = {
    "chunks": [
        {
            "id": 0,
            "text": "bitsandbytes enables accessible large language models...",
            "embeddings": {
                "all-MiniLM-L6-v2": [0.023, -0.041, ...],      # 384-dim
                "Qwen3-Embedding-0.6B": [0.112, 0.058, ...],    # 1024-dim
                "gemini-embedding-001": [-0.007, 0.031, ...],    # 3072-dim
            }
        },
        # ... more chunks
    ],
    "questions": [
        {
            "chunk_id": 0,
            "question": "What is the primary purpose of bitsandbytes?",
            "embeddings": {
                "all-MiniLM-L6-v2": [0.067, -0.012, ...],
                "Qwen3-Embedding-0.6B": [0.089, 0.044, ...],
                "gemini-embedding-001": [-0.015, 0.022, ...],
            }
        },
        # ... more questions
    ]
}

with open("data/embeddings/embeddings.json", "w") as f:
    json.dump(data, f)
```
, caption: [Unified embedding storage — all models in one JSON file keyed by model name],
)

#pagebreak(weak: true)

=== 19.3 Embedding Model Selection

#figure(
  table(
    columns: 4,
    align: (left, right, right, left),
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Model*], [*Output Dim*], [*10K Vectors (MB)*], [*Type*]),
    [all-MiniLM-L6-v2], [384], [14], [Open source],
    [Qwen3-Embedding-0.6B], [1024], [39], [Open source],
    [Qwen3-Embedding-4B], [2560], [98], [Open source],
    [Qwen3-Embedding-8B], [4096], [156], [Open source],
    [text-embedding-3-small], [1536], [59], [Proprietary (OpenAI)],
    [text-embedding-3-large], [3072], [117], [Proprietary (OpenAI)],
    [Gemini Embedding 001], [3072], [117], [Proprietary (Google)],
  ),
  caption: [Embedding models compared: dimensionality and storage requirements],
  kind: table,
)

*Dimension trade-offs:*
- *Large dimensions* (2048+): Capture more semantic nuance, distinguish finer details, but require more storage, compute, and increase vector database latency.
- *Small dimensions* (\<1000): Faster processing, lower storage, reduced latency, but may lose subtle semantic distinctions.

=== 19.4 Retrieval Metrics

Three core metrics are used to evaluate embedding model quality for retrieval tasks:

#figure(
  image("../diagrams/22-retrieval-scoring-metrics.png", width: 75%),
  caption: [Retrieval scoring flow: query embedding → cosine similarity → ranked results → evaluation metrics],
)

*Mean Reciprocal Rank (MRR):*

Measures how quickly the first relevant result appears in the ranked list:

#figure($ "MRR" = frac(1, "rank") $, caption: [Mean Reciprocal Rank — position of first relevant document], kind: math.equation)

Where $"rank"$ is the position of the correct document. If the correct document appears first, $"MRR" = 1$; at position 5, $"MRR" = 0.2$.

*Recall\@K:*

Indicates whether the relevant document appears within the top-K results. For datasets with one relevant document per question:

#figure($ "Recall@K" = frac(r, R) $, caption: [Recall\@K — presence of relevant document in top-K], kind: math.equation)

Where $r = 1$ if the relevant document is in the top-K, otherwise $r = 0$, and $R = 1$ (one relevant document). Recall\@K does not consider the _position_ within top-K — only whether the document appears at all.

*NDCG\@K (Normalized Discounted Cumulative Gain):*

Combines aspects of both MRR and Recall by considering the position of the relevant document:

#figure($ "DCG@K" = cases(frac(1, log_2("rank" + 1)) &"if rank" <= K, 0 &"otherwise") $, caption: [DCG\@K — position-aware relevance scoring], kind: math.equation)

#figure($ "NDCG@K" = frac("DCG@K", "IDCG@K") = "DCG@K" $, caption: [NDCG\@K — normalized discounted cumulative gain (IDCG\@K = 1 for single-document relevance)], kind: math.equation)

*Implementing metrics in Python:*

#figure(
```python
import math

def mrr(rank: int | None) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    return 1.0 / rank if rank else 0.0

def recall_at_k(rank: int | None, k: int) -> float:
    """Recall@K: 1 if relevant doc is in top-K, else 0."""
    return 1.0 if rank and rank <= k else 0.0

def ndcg_at_k(rank: int | None, k: int) -> float:
    """NDCG@K: position-aware relevance (simplified for single relevant doc)."""
    if rank and rank <= k:
        return 1.0 / math.log2(rank + 1)
    return 0.0

# Example: correct document at position 3 out of 3 results
rank = 3
print(f"MRR     = {mrr(rank):.4f}")          # 0.3333 — penalizes heavily
print(f"Recall@3 = {recall_at_k(rank, 3)}")   # 1.0    — only cares about presence
print(f"NDCG@3   = {ndcg_at_k(rank, 3):.4f}") # 0.5000 — moderate penalty
```
, caption: [Metric implementations — MRR penalizes most\, Recall\@K is binary\, NDCG\@K falls between],
)

#pagebreak(weak: true)

*Manual benchmark loop — computing metrics across all models:*

#figure(
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_rank(ranked_chunk_ids: list, correct_chunk_id: int) -> int | None:
    try:
        return ranked_chunk_ids.index(correct_chunk_id) + 1  # 1-indexed
    except ValueError:
        return None

# ground_truth: {question_idx: correct_chunk_id}
ground_truth = {i: q["chunk_id"] for i, q in enumerate(questions)}
chunk_ids = [c["id"] for c in chunks]
results = {}

for model_name in embedding_models:
    q_embs = np.array([q["embeddings"][model_name] for q in questions])
    c_embs = np.array([c["embeddings"][model_name] for c in chunks])

    sim_matrix = cosine_similarity(q_embs, c_embs)  # (377, 43)

    scores = {"mrr": [], "recall@1": [], "recall@5": [], "ndcg@5": []}
    for i in range(len(questions)):
        sorted_indices = np.argsort(-sim_matrix[i])  # descending similarity
        ranked_ids = [chunk_ids[idx] for idx in sorted_indices]
        rank = get_rank(ranked_ids, ground_truth[i])

        scores["mrr"].append(mrr(rank))
        scores["recall@1"].append(recall_at_k(rank, 1))
        scores["recall@5"].append(recall_at_k(rank, 5))
        scores["ndcg@5"].append(ndcg_at_k(rank, 5))

    results[model_name] = {k: np.mean(v) for k, v in scores.items()}
    print(f"{model_name}: MRR={results[model_name]['mrr']:.4f}  "
          f"R@5={results[model_name]['recall@5']:.4f}  "
          f"NDCG@5={results[model_name]['ndcg@5']:.4f}")
```
, caption: [Manual benchmark: cosine similarity \→ ranking \→ per-query metrics \→ mean scores per model],
)

#pagebreak(weak: true)

=== 19.5 Statistical Significance Testing

When benchmarking models, a higher mean score on your test set might be due to random noise rather than genuine superiority. Statistical tests determine if differences are *real* (statistically significant) or *random flukes*.

*The null hypothesis ($H_0$):* "The two models are identical, and any difference in their mean scores is just due to random chance."

*Process:*
+ Compute the metric difference between models across all queries.
+ Run a statistical test to obtain a *p-value* — the probability of observing your results if $H_0$ were true.
+ If $p < 0.05$ (threshold): reject $H_0$ — the improvement is statistically significant.
+ If $p >= 0.05$: cannot reject $H_0$ — insufficient evidence that one model is better.

*Paired t-test:*
Examines the per-query differences between two models. Checks whether the mean difference is significantly different from zero.

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Query*], [*Model A*], [*Model B*], [*Diff (B−A)*]),
    [Q1], [0.5], [0.6], [+0.1],
    [Q2], [0.4], [0.3], [−0.1],
    [Q3], [0.7], [0.8], [+0.1],
    [Q4], [0.5], [0.7], [+0.2],
    [Q5], [0.6], [0.6], [0.0],
    [*Mean*], [*0.54*], [*0.60*], [*+0.06*],
  ),
  caption: [Paired t-test: per-query score comparison between two models],
  kind: table,
)

*Fisher's randomization test:*
Simulates the null hypothesis by randomly shuffling scores between models thousands of times. Counts how many shuffled configurations produce a difference as large as or larger than the observed difference (e.g., 850/10,000 shuffles $arrow.r$ $p = 0.085 > 0.05$ $arrow.r$ cannot conclude Model B is significantly better).

#pagebreak(weak: true)

*Automated benchmarking with Ranx:*

#figure(
```python
from ranx import Qrels, Run, compare
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Build ground truth (Qrels) — maps questions to correct chunks
qrels_dict = {}
for i, q in enumerate(questions):
    qrels_dict[f"q_{i}"] = {f"chunk_{q['chunk_id']}": 1}
qrels = Qrels(qrels_dict)

# Step 2: Build runs — one per model with similarity scores
runs = {}
for model_name in embedding_models:
    q_embs = np.array([q["embeddings"][model_name] for q in questions])
    c_embs = np.array([c["embeddings"][model_name] for c in chunks])
    sim_matrix = cosine_similarity(q_embs, c_embs)

    run_dict = {}
    for i in range(len(questions)):
        chunk_scores = {f"chunk_{chunks[j]['id']}": float(sim_matrix[i][j])
                        for j in range(len(chunks))}
        run_dict[f"q_{i}"] = chunk_scores

    runs[model_name] = Run(run_dict, name=model_name)

# Step 3: Compare all models with statistical testing
report = compare(
    qrels=qrels,
    runs=list(runs.values()),
    metrics=["mrr", "recall@1", "recall@5", "ndcg@1", "ndcg@5"],
    max_p=0.05,          # significance threshold
    stat_test="fisher",  # Fisher's randomization test
)
print(report)
```
, caption: [Full Ranx benchmark pipeline: Qrels \→ Runs \→ compare with statistical tests],
)

The output table annotates each score with letters indicating which other models it *statistically significantly outperforms*. For instance, `0.8075 acfg` means the model is significantly better than models a, c, f, and g — validated by the chosen statistical test, not just raw score comparison.

#pagebreak(weak: true)

*Extracting win/tie/loss comparisons:*

#figure(
```python
# Access pairwise win/tie/loss data from the report
wtl = report.win_tie_loss

for pair_key, metrics_data in wtl.items():
    print(f"\n{pair_key}:")
    for metric, values in metrics_data.items():
        wins, ties, losses = values["W"], values["T"], values["L"]
        p_val = values["p_value"]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
              "*" if p_val < 0.05 else "ns"
        print(f"  {metric:>10s}: W={wins:3d}  T={ties:3d}  L={losses:3d}  "
              f"p={p_val:.4f} {sig}")
```
, caption: [Extracting pairwise win/tie/loss statistics from the Ranx report],
)

=== 19.6 Multilingual Evaluation

Multilingual embedding models learn *abstract meaning* rather than language-specific surface forms. The same concept in English and Arabic maps to the same region in vector space — enabling cross-lingual retrieval where users can search in their native language and find documents written in a foreign language.

*Testing protocol — four language combinations:*

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header(repeat: true, [*Sr \#*], [*Benchmark*], [*Questions Language*], [*Chunks Language*]),
    [1], [Monolingual (EN)], [English], [English],
    [2], [Cross-lingual (AR→EN)], [Arabic], [English],
    [3], [Cross-lingual (EN→AR)], [English], [Arabic],
    [4], [Monolingual (AR)], [Arabic], [Arabic],
  ),
  caption: [Four-benchmark multilingual evaluation protocol],
  kind: table,
)

#pagebreak(weak: true)

*Translating datasets for cross-lingual testing:*

#figure(
```python
from google import genai

client = genai.Client()

system_prompt = """You are a world-class translator. Translate the following text
from English to Arabic.
Rules:
- Do NOT add, remove, or modify any information
- Preserve formatting (bullet points, numbers, structure)
- Maintain technical terminology accurately
- Output ONLY the translation, no explanations"""

def translate_text(text: str) -> str:
    response = client.models.generate_content(
        model="gemini-3-flash",
        contents=[system_prompt, text],
    )
    return response.text

# Translate all chunks and questions
for chunk in chunks:
    chunk["text_ar"] = translate_text(chunk["text"])
for q in questions:
    q["question_ar"] = translate_text(q["question"])
```
, caption: [Translating benchmark data for cross-lingual evaluation using Gemini],
)

#pagebreak(weak: true)

*Visualizing multilingual embedding clusters with t-SNE:*

#figure(
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

model_name = "Qwen3-Embedding-4B"
en_embs = np.array([q["embeddings"][model_name] for q in questions])
ar_embs = np.array([q["embeddings_ar"][model_name] for q in questions])

all_embs = np.vstack([en_embs, ar_embs])
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(all_embs)

n = len(questions)
plt.figure(figsize=(10, 7))
plt.scatter(coords[:n, 0], coords[:n, 1], c="steelblue", alpha=0.6, label="English", s=20)
plt.scatter(coords[n:, 0], coords[n:, 1], c="crimson", alpha=0.6, label="Arabic", s=20)
plt.legend(fontsize=12)
plt.title(f"Multilingual Embedding Space — {model_name}", fontsize=14)
plt.xlabel("t-SNE component 1")
plt.ylabel("t-SNE component 2")
plt.tight_layout()
plt.savefig("multilingual_clusters.png", dpi=150)
```
, caption: [t-SNE visualization — multilingual models cluster by topic\, not by language],
)

With a well-trained multilingual model, English and Arabic versions of the same question appear at nearly identical positions in the embedding space. Models not trained on Arabic (e.g., all-MiniLM-L6-v2) show two separate clusters with no overlap — the Arabic queries land far from their English counterparts.

#figure(
  image("../diagrams/23-multilingual-embedding-eval.png", width: 90%),
  caption: [Multilingual vs monolingual embedding spaces — good multilingual models cluster by topic (left)\, while monolingual models separate by language (right)],
)

*Key findings:*
- Models trained on multiple languages (Gemini, Qwen3-Embedding-4B/8B) maintain >70% average scores across all four benchmarks.
- Models trained primarily on English (all-MiniLM-L6-v2) collapse to near-zero performance ($"MRR" approx 0.07$) when queries or documents are in Arabic.
- OpenAI models show moderate degradation (~20% drop) on cross-lingual tasks.
- Merging languages into a single vector index introduces minor *interference* (typically 2-3% accuracy drop) as the vector space becomes more crowded.

*Practical recommendation:* When selecting an embedding model for multilingual use, always verify language support — check the model card on Hugging Face (Languages tab) for open-source models, or the official documentation for proprietary models. A model not trained on your target language will fail catastrophically, not just degrade gracefully.

=== 19.7 Benchmarking Key Takeaways

- *Public benchmarks are useful starting points*, but your private data is the ultimate test for model selection.
- *Use p-values* to confirm if a model is genuinely better or just lucky on your test set.
- *Cross-lingual retrieval is powerful* with properly trained multilingual models, but beware of interference in mixed-language indexes.
- The *quality of your evaluation depends on the quality of your question-answer pairs* — invest time in supervision and validation.
- *Smaller models can surprise* — Qwen3-Embedding-4B often matches the 8B variant, making it a cost-effective choice when statistical tests confirm comparable performance.
- *Create reusable benchmarks* — when a new model is released, run it through your existing pipeline and see where it ranks. This is far more informative than relying solely on public leaderboard positions.

=== 19.8 Section Summary

Embedding evaluation requires building private benchmarks with your own domain data rather than relying solely on public leaderboards. The complete pipeline covers text extraction, semantic chunking, QA pair generation, multi-model embedding (Sentence Transformers, Qwen3, OpenAI, Gemini, llama.cpp), and ranking evaluation with Ranx (MRR, Recall\@K, NDCG\@K) using Fisher's randomization test for statistical significance. Multilingual evaluation with t-SNE visualization reveals cross-lingual clustering quality. For production deployment, create reusable benchmarks that any new model can be scored against.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
