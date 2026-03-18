#import "../template.typ": *

#let Chapter10() = [
== 10 - Knowledge Distillation

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-10-11-knowledge-distillation/Knowledge_DIstillation_in_Deep_Learning.ipynb")[`Knowledge_DIstillation_in_Deep_Learning.ipynb`]
]

=== 10.1 What Is Knowledge Distillation

Knowledge distillation, introduced by #link("https://arxiv.org/abs/1503.02531")[Hinton et al. (2015)], compresses a large *teacher* model into a smaller *student* model that mimics the teacher's behavior. The student learns from the teacher's _soft probability distributions_ (which contain more information than hard labels) rather than from ground-truth labels alone.

#figure(
```
Teacher (large, slow, accurate)
    ↓ soft predictions (probability distributions)
Student (small, fast, nearly as accurate)
```
, caption: [Knowledge distillation flow: teacher soft predictions to student model],
)

=== 10.2 Temperature Scaling - The Core Mechanism

Standard softmax produces _sharp_ distributions (one class gets ~99% probability). Temperature scaling _softens_ the distribution, revealing the teacher's learned inter-class relationships:

#figure($ "softmax"(z_i, T) = frac(exp(z_i / T), sum_j exp(z_j / T)) $, caption: [Temperature-scaled softmax for knowledge distillation], kind: math.equation)

#figure(
  table(
    columns: 2,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Symbol*], [*Meaning*]),
    [$z_i$], [Logit (raw output) for class $i$],
    [$T$], [Temperature (typically 2.0–5.0)],
  ),
  caption: [Temperature Scaling Formula Symbols],
  kind: table,
)

*Effect of temperature:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Temperature*], [*Distribution Shape*], [*Information Content*]),
    [T = 1 (standard)], [Sharp: \[0.97, 0.02, 0.01\]], [Only tells you "class 0"],
    [T = 2], [Softer: \[0.72, 0.18, 0.10\]], [Reveals "class 1 is more similar to class 0 than class 2 is"],
    [T = 5], [Very soft: \[0.48, 0.31, 0.21\]], [Maximum inter-class relationship information],
  ),
  caption: [Effect of Temperature on Soft Labels],
  kind: table,
)

This "dark knowledge" - the relative similarities between classes - is what makes distillation work. The student doesn't just learn _what_ the right answer is, but _how wrong_ the other answers are.

#pagebreak(weak: true)

=== 10.3 The Distillation Loss Function

#figure($ cal(L) = alpha dot.c T^2 dot.c "KL"("softmax"(frac(z_t, T)) || "softmax"(frac(z_s, T))) + (1 - alpha) dot.c "CE"(y, z_s) $, caption: [Knowledge distillation loss: KL divergence + cross-entropy], kind: math.equation)

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Symbol*], [*Meaning*], [*Typical Value*]),
    [$z_t$], [Teacher logits], [-],
    [$z_s$], [Student logits], [-],
    [$T$], [Temperature], [2.0],
    [$alpha$], [Weight for soft vs hard targets], [0.7 (favor soft targets)],
    [$"KL"$], [KL divergence (soft target loss)], [-],
    [$"CE"$], [Cross-entropy (hard label loss)], [-],
    [$T^2$], [Gradient magnitude compensation], [-],
  ),
  caption: [Knowledge Distillation Loss Symbols],
  kind: table,
)

*Why $T^2$:* Temperature scaling reduces gradient magnitudes by a factor of $1/T^2$. Multiplying by $T^2$ compensates, keeping the soft-target gradients comparable to the hard-label gradients.

#figure(
  image("../diagrams/08-knowledge-distillation.png", width: 85%),
  caption: [Knowledge Distillation — Training Data Flow],
)

#blockquote[
  *Implementation note:* The formula above uses $"softmax"$ for both teacher and student distributions (mathematically correct — KL divergence is defined over probability distributions). However, PyTorch's `F.kl_div()` expects *log-probabilities* for the input (student) to avoid numerical underflow. That is why the code below uses `F.log_softmax` for the student and `F.softmax` for the teacher — this is numerically stable and mathematically equivalent.
]

*Implementation:*

#figure(
```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.7):
    # Soft target loss (KL divergence between teacher and student distributions)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')

    # Hard target loss (standard cross-entropy with ground truth)
    ce_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    return alpha * (temperature ** 2) * kl_loss + (1 - alpha) * ce_loss
```
, caption: [Distillation loss combining KL divergence on soft targets with cross-entropy on hard labels],
)

=== 10.4 Three Levels of Distillation

The notebook demonstrates distillation at three scales:

*Level 1 - MLP (MNIST):*

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Model*], [*Architecture*], [*Parameters*], [*Accuracy*]),
    [Teacher], [2 hidden layers (512 + 256)], [~530K], [97.8%],
    [Student (no distillation)], [1 hidden layer (128)], [~100K], [96.1%],
    [Student (with distillation)], [1 hidden layer (128)], [~100K], [98.0%],
  ),
  caption: [Distillation Level 1: MLP on MNIST],
  kind: table,
)

The distilled student _exceeded_ the teacher - the soft targets acted as regularization.

*Level 2 - BERT (Tweet Sentiment):*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Model*], [*Size*], [*Accuracy*]),
    [Teacher: `bert-large-uncased`], [340M params], [22% (unfine-tuned!)],
    [Student: `bert-base-uncased`], [110M params], [60.8% (distilled)],
  ),
  caption: [Distillation Level 2: BERT on Tweet Sentiment],
  kind: table,
)

#blockquote[
  *Pedagogical caveat:* This example is atypical — the teacher was _not fine-tuned_ for sentiment classification, so the student's improvement comes primarily from the hard labels (ground truth), not the soft targets. In a proper distillation setup, the teacher should be a strong model fine-tuned for the target task. The student would then benefit from the teacher's soft probability distributions ("dark knowledge") revealing inter-class similarities that hard labels cannot express. This example demonstrates the mechanics of the distillation pipeline, but not the typical benefit of soft targets.
]

*Level 3 - LLM (Causal Language Model):*

#figure(
  table(
    columns: 2,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Model*], [*Size*]),
    [Teacher: `microsoft/phi-2`], [2.7B params],
    [Student: `microsoft/phi-1.5`], [1.3B params],
  ),
  caption: [Distillation Level 3: LLM Causal Language Model],
  kind: table,
)

Both loaded in 8-bit quantization. Per-prompt distillation loop with `ignore_index=tokenizer.pad_token_id` in cross-entropy loss.

*Real-world production pattern - DeepSeek R1 distillation:* DeepSeek R1 (671B parameters, Mixture-of-Experts architecture) was distilled into a family of smaller variants: 1.5B, 7B, 8B, 14B, 32B, and 70B parameter models. The distilled models retained much of the original's reasoning capability because the distillation process preserved the chain-of-thought reasoning patterns, not just the final answers. This is known as *"reasoning distillation"* - the student learns the teacher's _reasoning process_, including intermediate steps and self-correction patterns. This approach is closely related to Google's "Distilling Step-by-Step" method, where the key insight is that distilling _how_ a model thinks (rationales, step-by-step breakdowns) transfers far more capability than distilling _what_ it outputs (final answers alone). The DeepSeek R1 distilled variants demonstrated that even a 7B model can exhibit strong multi-step reasoning when trained on the full reasoning traces of a much larger model, making reasoning distillation one of the most effective techniques for compressing large reasoning models into deployable sizes.

=== 10.5 Key Research References

- *#link("https://arxiv.org/abs/2305.02301")["Distilling Step-by-Step"]* (Google, ACL 2023): A 770M T5 student outperformed 540B PaLM by distilling the _reasoning process_, not just the final answers
- *#link("https://arxiv.org/abs/1909.10351")[TinyBERT]* (Huawei, 2020): 4-layer BERT achieving 96.8% of BERT-base performance at 7.5× smaller size
- *#link("https://arxiv.org/abs/2501.12948")[DeepSeek R1 Distill]*: Production example of distilling a large reasoning model into 1.5B-7B variants

#pagebreak(weak: true)

=== 10.6 Worked Example: Distilling a Fraud Detection Model

*Scenario:* A bank has a large BERT-based fraud detection model (340M params, 200ms latency) deployed on cloud servers. They need a smaller model for real-time transaction screening at the payment terminal (\< 20ms latency).

*Teacher model:* Fine-tuned `bert-large` on 5M labeled transactions (fraud/legitimate)

*Student model:* `distilbert-base` (66M params, ~6x smaller)

*Training data with soft labels from teacher:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Transaction Features*], [*Hard Label*], [*Teacher Soft Output*]),
    ["\$4,999 wire transfer, new payee, 3am"], [Fraud (1)], [\[0.03, *0.97*\] - very confident fraud],
    ["\$50 grocery store, usual location"], [Legit (0)], [\[*0.99*, 0.01\] - very confident legit],
    ["\$800 online purchase, known merchant, new device"], [Legit (0)], [\[*0.71*, 0.29\] - uncertain, 29% fraud signal],
  ),
  caption: [Training Data with Soft Labels from Teacher],
  kind: table,
)

*The value of soft targets:* The third example is labeled "legitimate" but the teacher assigns 29% fraud probability - this uncertainty teaches the student that "new device + large amount" is a risk signal worth learning, even when the final label is legitimate. Hard labels alone would lose this nuance.

*Inference after distillation — testing the student model:*

#figure(
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load the distilled student model
student_model = AutoModelForSequenceClassification.from_pretrained("./distilled-fraud-detector")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
student_model.eval()

# Test transaction
transaction = "$4,999 wire transfer, new payee, 3am, first international transfer"
inputs = tokenizer(transaction, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = student_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    prediction = "FRAUD" if probs[0][1] > 0.5 else "LEGITIMATE"
    confidence = probs[0][1].item() if prediction == "FRAUD" else probs[0][0].item()

print(f"Prediction: {prediction} (confidence: {confidence:.2%})")
# Expected: Prediction: FRAUD (confidence: 96.42%)
# Student achieves ~95% of teacher accuracy at 6x smaller size and 10x faster inference
```
, caption: [Distilled fraud detection student model inference with confidence scoring],
)

#pagebreak(weak: true)

=== 10.7 Section Summary

Knowledge distillation compresses a large teacher model into a smaller student model by training on the teacher's soft probability distributions rather than hard labels alone, using temperature scaling (T = 2–5) to reveal inter-class relationships ("dark knowledge"). The distillation loss combines KL divergence on temperature-softened outputs with standard cross-entropy on ground-truth labels, weighted by α and compensated by T². The section demonstrates distillation at three scales — MLP on MNIST, BERT on tweet sentiment, and causal LLM (Phi-2 to Phi-1.5) — and discusses production applications such as DeepSeek R1's reasoning distillation into 1.5B–70B variants and Google's "Distilling Step-by-Step" method.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
