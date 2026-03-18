#import "../template.typ": *

#let Chapter4() = [
== 4 - LLM Evaluation

=== 4.1 Evaluation Metrics

HuggingFace's `evaluate` library provides standardized implementations of common NLP metrics. These metrics quantify how well a fine-tuned model performs on tasks like translation, summarization, and classification.

#figure(
```python
import evaluate

accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
perplexity = evaluate.load("perplexity")
```
, caption: [Loading evaluation metrics with HuggingFace evaluate library],
)

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Metric*], [*Measures*], [*Good For*], [*Interpretation*]),
    [*BLEU*], [Precision (n-gram overlap with reference)], [Translation], [0-1, higher = better],
    [*ROUGE*], [Recall (overlap between generated and reference)], [Summarization], [ROUGE-1 (unigram), ROUGE-2 (bigram), ROUGE-L (longest common subsequence)],
    [*Perplexity*], [How surprised the model is by the text], [Language modeling], [Lower is better; ranges are model- and dataset-dependent (e.g., 5-15 typical for well-trained LLMs on in-domain text)],
    [*Accuracy*], [Correct predictions / total predictions], [Classification], [0-1, higher = better],
  ),
  caption: [Evaluation Metrics for Fine-Tuned Models],
  kind: table,
)

=== 4.2 Evaluating Fine-Tuned LLMs in Practice

The metrics above (BLEU, ROUGE, Perplexity) are useful for automated benchmarks, but insufficient for evaluating fine-tuned LLMs in production. Modern evaluation forms a taxonomy of three approaches — *human evaluation*, *rule-based metrics*, and *model-based evaluation* — each with distinct trade-offs.

==== Human Evaluation

Human evaluation remains the gold standard. For production deployment, sample 100–200 outputs and have domain experts rate them. This is especially critical for safety-critical applications (medical, legal, financial).

*The inter-rater agreement problem:* When multiple humans judge the same output, they often disagree. Raw agreement percentage is misleading because evaluators can agree by chance. *Cohen's Kappa* adjusts for this:

#figure($ kappa = frac(p_o - p_e, 1 - p_e) $, caption: [Cohen's Kappa: inter-rater agreement adjusted for chance], kind: math.equation)

Where $p_o$ is the observed agreement (fraction of cases where raters agree) and $p_e$ is the expected agreement by chance. A Kappa of 1.0 means perfect agreement, 0.0 means agreement no better than chance, and negative values mean systematic disagreement.

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Metric*], [*Raters*], [*Scale*], [*Use Case*]),
    [*Cohen's Kappa*], [2 raters], [Categorical], [Pairwise human evaluation],
    [*Fleiss' Kappa*], [3+ raters], [Categorical], [Panel-based evaluation],
    [*Krippendorff's Alpha*], [Any number], [Any scale (nominal, ordinal, interval)], [Most flexible; preferred for complex annotation tasks],
  ),
  caption: [Inter-Rater Agreement Metrics],
  kind: table,
)

#blockquote[
  *Practical tip:* If your inter-rater agreement (Kappa) is below 0.6, your evaluation guidelines are too vague. Refine the rubric before collecting more ratings — disagreement in the labels means your evaluation signal is noisy.
]

#pagebreak(weak: true)

*Computing inter-rater agreement with scikit-learn:*

#figure(
```python
from sklearn.metrics import cohen_kappa_score
import numpy as np

# Two human raters evaluated 20 model outputs as "good" (1) or "bad" (0)
rater_1 = [1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]
rater_2 = [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]

# Cohen's Kappa — adjusts raw agreement for chance
kappa = cohen_kappa_score(rater_1, rater_2)
raw_agreement = np.mean(np.array(rater_1) == np.array(rater_2))

print(f"Raw agreement:  {raw_agreement:.2%}")   # 85.00%
print(f"Cohen's Kappa:  {kappa:.4f}")            # 0.6842
print(f"Interpretation: {'Substantial' if kappa > 0.6 else 'Moderate'} agreement")

# For 3+ raters, use Fleiss' Kappa (statsmodels)
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

# Each row = one sample, each column = one rater's label
ratings = np.array([
    [1, 1, 1],  # all raters agree: good
    [0, 0, 1],  # two say bad, one says good
    [1, 0, 0],  # one says good, two say bad
    [1, 1, 0],  # two say good, one says bad
])
table, _ = aggregate_raters(ratings)
fleiss_k = fleiss_kappa(table)
print(f"Fleiss' Kappa (3 raters): {fleiss_k:.4f}")
```
, caption: [Computing Cohen's Kappa (2 raters) and Fleiss' Kappa (3+ raters) for inter-rater agreement],
)

==== Rule-Based Metrics: Strengths and Limitations

*METEOR* (Metric for Evaluation of Translation with Explicit ORdering) computes a harmonic mean of unigram precision and recall, then applies a fragmentation penalty based on the number of contiguous matching chunks:

#figure($ "METEOR" = F_("score") times (1 - "Penalty") $, caption: [METEOR score: harmonic mean with fragmentation penalty], kind: math.equation)

The penalty increases when matching words are scattered across the output rather than appearing in contiguous chunks, rewarding fluency alongside correctness.

*BLEU* (Bilingual Evaluation Understudy) is precision-focused: it measures what fraction of n-grams in the generated text appear in the reference, with a brevity penalty to discourage overly short outputs. Commonly used for translation.

*ROUGE* (Recall-Oriented Understudy for Gisting Evaluation) is recall-focused: it measures what fraction of n-grams in the reference appear in the generated text. Commonly used for summarization:
- ROUGE-1: unigram overlap
- ROUGE-2: bigram overlap
- ROUGE-L: longest common subsequence

*BERTScore* uses contextual embeddings (from models like BERT) to compute token-level cosine similarity between generated and reference text. Unlike n-gram metrics, BERTScore captures semantic similarity — "couch" and "sofa" score highly even though they share no n-grams.

#blockquote[
  *Key limitation of all rule-based metrics:* They require a reference text, do not allow stylistic variation (a correct paraphrase scores poorly), and correlate only moderately with human judgments. For example, "A plush teddy bear can comfort a child during bedtime" and "Many youngsters rest more easily at night when they cuddle a gentle toy companion" convey the same meaning but would score poorly against each other on BLEU/ROUGE.
]

*Computing BLEU, ROUGE, and METEOR with HuggingFace evaluate:*

#figure(
```python
import evaluate

# Model predictions and ground-truth references
predictions = [
    "LoRA reduces memory by training low-rank matrices instead of full weights.",
    "Fine-tuning adapts a pre-trained model to a specific downstream task.",
]
references = [
    "LoRA achieves memory efficiency by decomposing weight updates into low-rank matrices.",
    "Fine-tuning is the process of adapting a pre-trained model to a target task.",
]

# BLEU — precision-focused, penalizes short outputs
bleu = evaluate.load("bleu")
bleu_result = bleu.compute(
    predictions=predictions,
    references=[[r] for r in references],  # BLEU expects list-of-lists for references
)
print(f"BLEU: {bleu_result['bleu']:.4f}")

# ROUGE — recall-focused, standard for summarization
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(predictions=predictions, references=references)
print(f"ROUGE-1: {rouge_result['rouge1']:.4f}")
print(f"ROUGE-2: {rouge_result['rouge2']:.4f}")
print(f"ROUGE-L: {rouge_result['rougeL']:.4f}")

# METEOR — harmonic mean of precision/recall with fragmentation penalty
meteor = evaluate.load("meteor")
meteor_result = meteor.compute(predictions=predictions, references=references)
print(f"METEOR: {meteor_result['meteor']:.4f}")
```
, caption: [Computing BLEU\, ROUGE\, and METEOR using the HuggingFace `evaluate` library],
)

#pagebreak(weak: true)

*Computing BERTScore — semantic similarity via contextual embeddings:*

#figure(
```python
from bert_score import score as bert_score

predictions = [
    "LoRA reduces memory by training low-rank matrices instead of full weights.",
    "A plush teddy bear can comfort a child during bedtime.",
]
references = [
    "LoRA achieves memory efficiency by decomposing weight updates into low-rank matrices.",
    "Many youngsters rest more easily at night when they cuddle a gentle toy companion.",
]

# BERTScore captures semantic similarity — "couch" and "sofa" score highly
P, R, F1 = bert_score(
    predictions, references,
    lang="en",
    model_type="microsoft/deberta-xlarge-mnli",  # best correlation with human judgments
    verbose=True,
)
for i, (p, r, f) in enumerate(zip(P, R, F1)):
    print(f"Pair {i+1}: P={p:.4f}  R={r:.4f}  F1={f:.4f}")

# Pair 1 (paraphrase):    F1 ≈ 0.92 — high despite different wording
# Pair 2 (semantic match): F1 ≈ 0.85 — captures meaning that BLEU/ROUGE miss
```
, caption: [BERTScore captures semantic similarity that n-gram metrics miss — paraphrases score highly],
)

==== LLM-as-a-Judge

The core idea: use a strong LLM (GPT-5, Claude Opus 4.6, Gemini 3 Pro) to rate outputs on specific criteria. This approach does *not* require reference texts or human ratings to get started, and critically, it provides a *rationale* alongside the score — explaining _why_ a response was rated a certain way.

*Two evaluation modes:*

1. *Pointwise:* Evaluate a single response — "Is this response good or bad?"
2. *Pairwise:* Compare two responses — "Is Response A or Response B better?" (useful for generating synthetic preference data for DPO/RLHF)

#figure(
```python
# LLM-as-a-Judge with structured output (recommended)
from pydantic import BaseModel

class EvalResult(BaseModel):
    rationale: str   # Output rationale FIRST (improves score quality — same principle as chain-of-thought)
    score: int       # Binary (0/1 pass/fail) preferred over granular scales

eval_prompt = f"""Evaluate this response for {criteria}.

Question: {question}
Response: {model_output}

First explain your reasoning, then provide a pass (1) or fail (0) score."""

# Use structured output / constrained decoding to guarantee parseable results
# OpenAI: response_format=EvalResult
# Anthropic: tool_use with schema
# Open-source: use Outlines or vLLM guided decoding
```
, caption: [LLM-as-a-Judge evaluation with structured output and binary scoring],
)

#blockquote[
  *Why rationale before score?* Empirically, asking the judge to output reasoning before the score improves evaluation quality — the same principle behind chain-of-thought and reasoning models. The model "thinks through" its assessment before committing to a number.
]

*Known biases in LLM-as-a-Judge:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Bias*], [*Description*], [*Mitigation*]),
    [*Position bias*], [Prefers whichever response appears first (A vs B)], [Swap order and take majority vote; if results flip, flag as uncertain],
    [*Verbosity bias*], [Prefers longer responses regardless of quality], [Explicitly instruct judge to ignore length; add length penalty; provide few-shot examples],
    [*Self-enhancement bias*], [Prefers responses generated by itself], [Use a different (ideally larger) model as judge than the one that generated responses],
  ),
  caption: [LLM-as-a-Judge Biases and Mitigations],
  kind: table,
)

*Best practices for LLM-as-a-Judge:*

1. *Use binary scales* (pass/fail) — granular 1–5 scales add noise without meaningful signal, and humans also find binary judgments easier to calibrate
2. *Write crisp evaluation criteria* — vague guidelines produce inconsistent scores
3. *Output rationale before score* — chain-of-thought for the judge
4. *Use low temperature* (0.0–0.2) for reproducible evaluations across runs
5. *Calibrate against human ratings* — periodically collect human judgments and run correlation analysis to ensure the LLM judge hasn't drifted from human preferences
6. *Use structured output* — constrained/guided decoding guarantees a parseable response format

*Pairwise LLM-as-a-Judge — generating preference data for DPO:*

#block(breakable: true)[
#figure(
```python
from openai import OpenAI
from pydantic import BaseModel

class PairwiseResult(BaseModel):
    rationale: str
    winner: str       # "A" or "B"
    confidence: str   # "high" or "low"

client = OpenAI()

def pairwise_judge(question: str, response_a: str, response_b: str) -> dict:
    """Compare two responses with position-bias mitigation (swap and vote)."""
    prompt_template = """Compare these two responses to the question.
Question: {q}
Response A: {a}
Response B: {b}
Which response is better? Consider: accuracy, completeness, clarity, relevance.
First explain your reasoning, then declare the winner (A or B)."""

    # Round 1: A first, B second
    r1 = client.beta.chat.completions.parse(
        model="gpt-4.1", temperature=0.0,
        messages=[{"role": "user", "content": prompt_template.format(
            q=question, a=response_a, b=response_b)}],
        response_format=PairwiseResult,
    )

    # Round 2: swap order to mitigate position bias
    r2 = client.beta.chat.completions.parse(
        model="gpt-4.1", temperature=0.0,
        messages=[{"role": "user", "content": prompt_template.format(
            q=question, a=response_b, b=response_a)}],  # swapped
        response_format=PairwiseResult,
    )

    vote_1 = r1.choices[0].message.parsed.winner
    vote_2 = "B" if r2.choices[0].message.parsed.winner == "A" else "A"

    if vote_1 == vote_2:
        return {"winner": vote_1, "consistent": True}
    return {"winner": "tie", "consistent": False}  # flag for human review
```
, caption: [Pairwise LLM-as-a-Judge with position-bias mitigation — swap order and vote for DPO preference data],
)
]

#blockquote[
  *Warning: Proxy over-optimization.* The LLM-as-a-Judge score is an _approximation_ of human preference. If you optimize your model solely to maximize the judge's score, you risk Goodhart's Law — "when a measure becomes a target, it ceases to be a good measure." Always validate against real human ratings periodically.
]

==== Factuality Evaluation

For evaluating whether generated text is factually correct, a multi-step decomposition approach is standard:

1. *Decompose* the text into atomic facts using an LLM (e.g., "Teddy bears were first created in the 1900s" → individual claims)
2. *Verify* each fact independently using RAG, web search, or knowledge base lookup (binary: correct or incorrect)
3. *Aggregate* with optional importance weighting:

#figure($ "Factuality Score" = sum_(i=1)^(N) alpha_i dot.c "correct"_i $, caption: [Weighted factuality score across extracted claims], kind: math.equation)

Where $alpha_i$ weights each fact by importance (set all equal for simplicity). This captures nuance: a text with one minor error in 10 facts scores differently than one with 5 major errors.

#pagebreak(weak: true)

*Factuality evaluation pipeline — decompose, verify, aggregate:*

#figure(
```python
from openai import OpenAI
import json

client = OpenAI()

def decompose_claims(text: str) -> list[str]:
    """Step 1: Break model output into atomic, verifiable claims."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        messages=[{"role": "user", "content": f"""Extract all atomic factual claims from this text.
Each claim should be a single, independently verifiable statement.
Return as a JSON list of strings.

Text: {text}"""}],
    )
    return json.loads(response.choices[0].message.content)

def verify_claim(claim: str, context: str) -> dict:
    """Step 2: Verify each claim against source context."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        messages=[{"role": "user", "content": f"""Is this claim supported by the context?

Claim: {claim}
Context: {context}

Respond with JSON: {{"supported": true/false, "evidence": "quote or explanation"}}"""}],
    )
    return json.loads(response.choices[0].message.content)

def factuality_score(text: str, context: str) -> float:
    """Step 3: Aggregate — fraction of supported claims."""
    claims = decompose_claims(text)
    results = [verify_claim(c, context) for c in claims]
    supported = sum(1 for r in results if r["supported"])
    score = supported / len(claims) if claims else 0.0

    print(f"Claims: {len(claims)}, Supported: {supported}, Score: {score:.2%}")
    for c, r in zip(claims, results):
        status = "✓" if r["supported"] else "✗"
        print(f"  {status} {c}")
    return score
```
, caption: [Three-step factuality evaluation: decompose text into atomic claims\, verify each against source\, and aggregate],
)

#pagebreak(weak: true)

==== Evaluation Dimensions

Two broad axes for evaluating LLM outputs:

- *Task performance:* Was the response useful, factual, relevant, and complete?
- *Alignment:* Does the response match the desired tone, style, format, and safety requirements?

==== Standard Benchmarks

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Category*], [*Benchmark*], [*What It Tests*], [*Format*]),
    [*Knowledge*], [MMLU (Massive Multitask Language Understanding)], [Factual knowledge across ~60 domains (law, medicine, STEM, humanities)], [4-way multiple choice],
    [*Math Reasoning*], [AIME (American Invitational Mathematics Examination)], [Multi-step mathematical reasoning (Olympiad-level)], [3-digit numerical answer],
    [*Common Sense*], [PIQA (Physical Interaction QA)], [Everyday physical reasoning (20K examples)], [2-way multiple choice],
    [*Coding*], [SWE-Bench], [Real GitHub issues from popular Python repos; assessed by whether introduced tests pass], [Code patch (test-driven)],
    [*Safety*], [HarmBench], [Harmful behavior detection across 4 categories (standard, copyright, contextual, multimodal)], [Classifier-assessed],
  ),
  caption: [Major LLM Benchmarks],
  kind: table,
)

#blockquote[
  *Benchmark contamination:* Benchmarks are only meaningful if the model hasn't seen the test data during training. Techniques to prevent contamination include hash-based deduplication, blocklisting benchmark websites during web crawling, and using fresh test sets (e.g., new AIME exams each year). *Goodhart's Law* applies: "When a measure becomes a target, it ceases to be a good measure."
]

#pagebreak(weak: true)

*Running standard benchmarks with lm-evaluation-harness:*

#figure(
```python
# EleutherAI's lm-evaluation-harness — the standard tool for benchmark evaluation
# Install: pip install lm-eval

# CLI usage — evaluate your fine-tuned model on MMLU (5-shot)
# lm_eval --model hf \
#   --model_args pretrained=your-model-path,dtype=bfloat16 \
#   --tasks mmlu \
#   --num_fewshot 5 \
#   --batch_size 8 \
#   --output_path results/

# Python API for programmatic evaluation
from lm_eval import evaluator, tasks

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=your-model-path,dtype=bfloat16",
    tasks=["mmlu", "hellaswag", "arc_challenge"],
    num_fewshot=5,
    batch_size=8,
)

# Compare base vs fine-tuned model
for task_name, task_results in results["results"].items():
    acc = task_results.get("acc,none", task_results.get("acc_norm,none", 0))
    print(f"{task_name:>20s}: {acc:.4f}")

# Key check: fine-tuned model should not *regress* on general benchmarks
# If MMLU drops >2% after domain fine-tuning, you have catastrophic forgetting
```
, caption: [Running standard benchmarks with EleutherAI's lm-evaluation-harness to check for catastrophic forgetting],
)

==== Evaluation Strategy by Fine-Tuning Stage

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Stage*], [*What to Evaluate*], [*Recommended Method*]),
    [Non-instructional FT], [Domain perplexity (should decrease) + general perplexity (should not spike)], [Automated (perplexity on held-out sets)],
    [Instruction FT], [Instruction-following quality, response format compliance], [LLM-as-a-Judge + MT-Bench],
    [DPO/RLHF], [Safety, helpfulness, preference alignment], [Human evaluation + reward model scores],
  ),
  caption: [Evaluation Strategy by Fine-Tuning Stage],
  kind: table,
)

#pagebreak(weak: true)

*Putting it all together — automated evaluation after each fine-tuning stage:*

#figure(
```python
import evaluate
from bert_score import score as bert_score
from openai import OpenAI
import json

client = OpenAI()

def evaluate_stage(model_outputs: list[str], references: list[str],
                   questions: list[str], stage: str) -> dict:
    """Run the full evaluation suite appropriate for a given fine-tuning stage."""
    results = {}

    # 1. Rule-based metrics (always run — cheap baseline)
    rouge = evaluate.load("rouge")
    results["rouge"] = rouge.compute(predictions=model_outputs, references=references)

    _, _, f1 = bert_score(model_outputs, references, lang="en",
                           model_type="microsoft/deberta-xlarge-mnli")
    results["bertscore_f1"] = f1.mean().item()

    # 2. LLM-as-a-Judge (pointwise, binary)
    criteria = {
        "non_instructional": "domain accuracy and terminology usage",
        "instruction_ft": "instruction-following quality and format compliance",
        "dpo_rlhf": "helpfulness, safety, and preference alignment",
    }
    judge_scores = []
    for q, out in zip(questions, model_outputs):
        resp = client.chat.completions.create(
            model="gpt-4.1-mini", temperature=0.0,
            messages=[{"role": "user", "content":
                f"Evaluate for {criteria[stage]}.\n"
                f"Question: {q}\nResponse: {out}\n"
                "Rationale first, then score (1=pass, 0=fail). "
                "Return JSON: {{\"rationale\": \"...\", \"score\": 0 or 1}}"}],
        )
        judge_scores.append(json.loads(resp.choices[0].message.content)["score"])

    results["judge_pass_rate"] = sum(judge_scores) / len(judge_scores)

    # 3. Summary
    print(f"\n{'='*50}")
    print(f"Stage: {stage}")
    print(f"  ROUGE-L:        {results['rouge']['rougeL']:.4f}")
    print(f"  BERTScore F1:   {results['bertscore_f1']:.4f}")
    print(f"  Judge pass rate: {results['judge_pass_rate']:.2%}")
    print(f"{'='*50}")
    return results
```
, caption: [End-to-end evaluation function combining rule-based metrics\, BERTScore\, and LLM-as-a-Judge per stage],
)

=== 4.3 Debugging Fine-Tuning Runs

*What healthy loss curves look like:*

- *Training loss* should decrease steadily and plateau — not oscillate wildly (lr too high) or barely move (lr too low)
- *Validation loss* should track training loss. If validation loss rises while training loss decreases → *overfitting* (reduce epochs, add regularization, increase data)
- *Sudden loss spikes* indicate exploding gradients (lower lr, increase gradient clipping `max_grad_norm`)

*Common failure modes and fixes:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Symptom*], [*Likely Cause*], [*Fix*]),
    [Loss doesn't decrease], [Learning rate too low, or data issue], [Increase lr (try 2e-4 → 5e-4), check data format],
    [Loss oscillates wildly], [Learning rate too high], [Reduce lr by 2-5x, increase warmup steps],
    [Val loss rises after epoch 1], [Overfitting (too many epochs or too little data)], [Reduce epochs, use early stopping, add more training data],
    [Output is gibberish], [Wrong chat template, broken tokenizer, or catastrophic lr], [Verify template matches model, check tokenizer `pad_token`, reduce lr drastically],
    [Model repeats itself], [Repetition penalty too low, or overfit on repetitive data], [Add `repetition_penalty=1.15`, diversify training data],
    [OOM during training], [Batch size too large, sequence too long], [Reduce batch size, enable gradient checkpointing, reduce `max_seq_length`],
  ),
  caption: [Common failure modes and fixes],
  kind: table,
)

*Key debugging commands:*

#figure(
```python
# Monitor training with TensorBoard or Weights & Biases
from transformers import TrainingArguments
args = TrainingArguments(
    logging_steps=10,          # Log every 10 steps
    eval_strategy="steps",     # Evaluate during training
    eval_steps=50,             # Evaluate every 50 steps
    save_strategy="steps",
    load_best_model_at_end=True,  # Restore best checkpoint at end
    report_to="tensorboard",   # or "wandb"
)
```
, caption: [TrainingArguments setup for monitoring with TensorBoard or W&B],
)

#pagebreak(weak: true)

=== 4.4 Reproducibility and Version Pinning

The fine-tuning ecosystem evolves rapidly — `trl`, `peft`, `unsloth`, and `transformers` push breaking changes frequently. A pipeline that works today may fail next month without version pinning.

*Critical version dependencies to pin:*

#figure(
```
# requirements-finetune.txt — pin these explicitly
transformers==4.47.0        # SFTTrainer API changed significantly between 4.40→4.45
peft==0.14.0                # LoraConfig parameter names evolve
trl==0.13.0                 # DPOTrainer/SFTTrainer interface changes often
datasets==3.2.0
bitsandbytes==0.45.0        # Quantization kernel compatibility
accelerate==1.2.0
torch==2.5.1                # CUDA kernel compatibility
unsloth==2025.3.19          # Date-versioned; pin to specific release
```
, caption: [requirements-finetune.txt with pinned library versions for reproducibility],
)

*Reproducibility checklist:*

- Pin all library versions in `requirements.txt`
- Set random seeds: `torch.manual_seed(42)`, `random.seed(42)`, `np.random.seed(42)`
- Log the exact model revision: `model = AutoModel.from_pretrained("model-name", revision="main")` #sym.dash.em better yet, log the commit hash
- Save the full training config (hyperparameters, data preprocessing steps, LoRA config) as JSON alongside model checkpoints
- Record GPU type and CUDA version: `torch.cuda.get_device_name()`, `torch.version.cuda`
- Use `deterministic=True` where supported (note: this may slow training)

#blockquote[
  *Why this matters:* "My fine-tuned model worked great last week but now it doesn't" is almost always a library version mismatch, not a model issue. Five minutes of version pinning saves hours of debugging.
]

*Experiment tracking tools:* While a deep dive into experiment tracking is beyond the scope of these notes, any serious fine-tuning workflow should integrate one of the following to log hyperparameters, loss curves, evaluation metrics, and model artifacts across runs:

- *Weights & Biases (W&B)* — the most widely used tracker in the LLM fine-tuning community. HuggingFace `Trainer` has native W&B integration via `report_to="wandb"`. Tracks GPU utilization, gradient norms, and learning rate schedules automatically. Free tier is generous for individual researchers.
- *MLflow* — open-source and self-hostable, making it popular in enterprise environments where data must stay on-premises. Supports model registry, experiment comparison, and deployment packaging. Integrates with `Trainer` via `report_to="mlflow"`.
- *Trackio* — a lightweight, minimal-config alternative focused on simplicity. Good for quick experiments where full W&B/MLflow infrastructure is overkill. Provides real-time loss visualization with minimal setup overhead.

All three tools help answer the critical question: "Which combination of hyperparameters, data mix, and LoRA config produced my best checkpoint?" — something that becomes essential once you are running more than a handful of fine-tuning experiments.

=== 4.5 Section Summary

LLM evaluation spans three tiers: rule-based metrics (BLEU, ROUGE, METEOR, BERTScore) for automated baselines, LLM-as-a-Judge with pointwise and pairwise modes for nuanced quality assessment, and human evaluation with inter-rater agreement (Cohen's/Fleiss' Kappa) as the gold standard. Factuality evaluation uses a decompose-verify-aggregate pipeline, and standard benchmarks (MMLU, AIME, SWE-Bench, HarmBench) are run via EleutherAI's lm-evaluation-harness to detect catastrophic forgetting. Debugging fine-tuning runs relies on loss curve interpretation and perplexity monitoring, while reproducibility requires version pinning and experiment tracking (W&B, MLflow, Trackio).
]
