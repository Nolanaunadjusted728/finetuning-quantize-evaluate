#import "../template.typ": *

#let Chapter5() = [
== 5 - BERT Fine-Tuning

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-09-Bert-finetuning/BERT_Finetuning.ipynb")[`BERT_Finetuning.ipynb`]
]

=== 5.1 Why BERT Still Matters

While generative LLMs dominate headlines, BERT-family models remain the workhorse for:

- *Classification* - sentiment, spam detection, content moderation
- *Named Entity Recognition (NER)* - extracting people, organizations, locations
- *Question Answering* - extractive QA from documents
- *RAG retrieval* - encoding queries and documents for semantic search
- *On-device inference* - small enough for mobile/edge deployment

=== 5.2 Task 1: Text Classification (Sentiment Analysis)

*Using HF Trainer (high-level API):*

#figure(
```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./bert-imdb",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
)

trainer = Trainer(model=model, args=training_args,
                  train_dataset=tokenized_train, eval_dataset=tokenized_test,
                  compute_metrics=compute_metrics)
trainer.train()

# Push to Hub and use via pipeline
trainer.push_to_hub()
classifier = pipeline("text-classification", model="your-username/bert-imdb")
```
, caption: [BERT sentiment classifier fine-tuning with HuggingFace Trainer],
)

#pagebreak()

*Using manual PyTorch loop (low-level control):*

#figure(
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                            num_training_steps=total_steps)

for epoch in range(epochs):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```
, caption: [Manual PyTorch training loop for BERT with AdamW and linear warmup scheduler],
)

*When to use which:* The `Trainer` API handles logging, checkpointing, distributed training, and mixed precision automatically. The manual loop gives control over gradient accumulation, custom loss functions, and multi-task training. Use `Trainer` by default; switch to manual when you need custom training logic.

=== 5.3 Task 2: Named Entity Recognition (NER)

NER extracts entities from text using the *BIO tagging scheme:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Tag*], [*Meaning*], [*Example*]),
    [`B-PER`], [Beginning of a person name], [*B-PER*: "Barack"],
    [`I-PER`], [Inside (continuation) of a person name], [*I-PER*: "Obama"],
    [`B-ORG`], [Beginning of an organization], [*B-ORG*: "Google"],
    [`I-ORG`], [Inside an organization name], [*I-ORG*: "DeepMind" (in "Google DeepMind")],
    [`B-LOC`], [Beginning of a location], [*B-LOC*: "Paris"],
    [`O`], [Outside any entity], [*O*: "visited", "the", "in"],
  ),
  caption: [BIO Tagging Scheme for NER],
  kind: table,
)

*The subword alignment problem:* BERT's tokenizer splits words into subword tokens (e.g., "Obama" → "Obama" but "Schwarzenegger" → "Schwarz", "\#\#enegger"). NER labels are assigned per _word_, not per subword. Solution:

#figure(
```python
def align_labels_with_tokens(labels, word_ids):
    aligned = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)  # Special tokens (CLS, SEP, PAD) - ignored by loss
        elif word_id != previous_word_id:
            aligned.append(labels[word_id])  # First subword gets the label
        else:
            aligned.append(-100)  # Subsequent subwords - ignored by loss
        previous_word_id = word_id
    return aligned
```
, caption: [NER subword alignment: mapping word-level labels to BERT token positions],
)

*Key insight:* `-100` is PyTorch's `CrossEntropyLoss` ignore index - any token with label `-100` is excluded from loss computation. This is the same technique used for response masking in instruction fine-tuning (Section 7.5).

=== 5.4 Task 3: Extractive Question Answering (SQuAD)

BERT predicts answer spans by outputting *start and end positions* within the context:

#figure(
  image("../diagrams/03-bert-qa.png", width: 85%),
  caption: [BERT Extractive Question Answering],
)

#figure(
```python
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Input: [CLS] question [SEP] context [SEP]
inputs = tokenizer(question, context, return_tensors="pt", return_offsets_mapping=True)

# Output: start_logits and end_logits over all tokens
outputs = model(**inputs)
start_idx = torch.argmax(outputs.start_logits)
end_idx = torch.argmax(outputs.end_logits)

# Map token positions back to character positions using offset_mapping
answer = context[offsets[start_idx][0]:offsets[end_idx][1]]
```
, caption: [BERT extractive QA: predicting answer start/end positions with offset mapping],
)

*Offset mapping* (`return_offsets_mapping=True`) provides character-level start/end positions for each token, enabling precise answer extraction from the original text.

=== 5.5 Worked Example: Customer Review Entity Extraction

*Scenario:* An e-commerce platform wants to extract product mentions, brand names, and attributes from customer reviews.

*Training data (NER with BIO tags):*

#figure(
```
Token:  "The"  "Samsung"  "Galaxy"  "S24"  "has"  "amazing"  "battery"  "life"
Label:   O      B-BRAND    B-PROD    I-PROD  O      O          B-ATTR     I-ATTR
```
, caption: [NER BIO-tagged training data for e-commerce product entity extraction],
)

#figure(
```
Token:  "Returned"  "my"  "Nike"   "running"  "shoes"  "-"  "too"  "narrow"
Label:   O           O     B-BRAND  B-PROD     I-PROD   O     O      B-ATTR
```
, caption: [NER BIO-tagged training data: brand and product attribute annotations],
)

*Before fine-tuning (base BERT NER):*

#figure(
```
Input: "The Samsung Galaxy S24 has amazing battery life"
Output: Samsung → B-ORG, Galaxy → O, S24 → O, battery → O, life → O
        ← Recognizes Samsung as an org, but misses product and attributes
```
, caption: [Base BERT NER output before fine-tuning: misclassifies product entities],
)

*After fine-tuning on e-commerce review data:*

#figure(
```
Input: "The Samsung Galaxy S24 has amazing battery life"
Output: Samsung → B-BRAND, Galaxy → B-PROD, S24 → I-PROD, battery → B-ATTR, life → I-ATTR
        ← Correctly identifies brand, product name, and product attributes
```
, caption: [Fine-tuned BERT NER output: correctly tags brand, product, and attributes],
)

#pagebreak()

*Inference after BERT fine-tuning — testing all three tasks:*

#figure(
```python
from transformers import pipeline

# Task 1: Classification inference
classifier = pipeline("text-classification", model="./bert-ecommerce-classifier")
result = classifier("The Samsung Galaxy S24 has amazing battery life but the camera is disappointing")
print(result)  # [{'label': 'MIXED', 'score': 0.89}]

# Task 2: NER inference
ner = pipeline("token-classification", model="./bert-ecommerce-ner", aggregation_strategy="simple")
entities = ner("The Samsung Galaxy S24 has amazing battery life")
print(entities)
# [{'entity_group': 'BRAND', 'word': 'Samsung', 'score': 0.98},
#  {'entity_group': 'PROD', 'word': 'Galaxy S24', 'score': 0.96},
#  {'entity_group': 'ATTR', 'word': 'battery life', 'score': 0.94}]

# Task 3: Extractive QA inference
qa = pipeline("question-answering", model="./bert-squad-finetuned")
answer = qa(question="What is the bioavailability of atorvastatin?",
            context="Atorvastatin has an absolute bioavailability of approximately 14%.")
print(answer)  # {'answer': 'approximately 14%', 'score': 0.95, 'start': 52, 'end': 70}
```
, caption: [BERT inference for classification, NER, and extractive QA using pipeline API],
)

=== 5.6 Connecting BERT to Decoder-Only LLMs

BERT is an *encoder-only* transformer optimized for _understanding_ tasks: classification, NER, extractive QA, and semantic similarity. The decoder-only models covered in Sections 6-8 (GPT-2, LLaMA, Mistral) are optimized for _generation_ tasks: text completion, instruction following, summarization, and open-ended QA. This architectural distinction has direct practical implications for choosing which model family to fine-tune.

*When to use BERT vs decoder models for classification:* BERT (and its variants) is faster to train, cheaper to serve, and often more accurate for pure classification and extraction tasks. A fine-tuned `bert-base` (110M parameters) can match or outperform a fine-tuned 7B decoder model on sentiment analysis while being ~60x smaller. Use decoder models when you need generation capability alongside understanding — for example, classifying a customer complaint AND generating a suggested response, or extracting entities AND producing a natural-language summary of findings.

*Modern alternatives bridging both paradigms:*

- *For understanding tasks:* ModernBERT and DeBERTa-v3 offer improved encoder architectures with better performance on Natural Language Understanding (NLU) benchmarks than the original BERT while remaining lightweight and efficient to fine-tune.
- *For generation + understanding:* Small decoder models like Phi-3-mini (3.8B) and TinyLlama (1.1B) can handle both classification and generation, making them suitable when a single model must serve dual purposes. These are covered in later sections with LoRA/QLoRA fine-tuning techniques that make training them practical on consumer hardware.
- *Practical guidance:* If your task is purely extractive or classificatory (no generation needed), start with an encoder model. You will get faster inference, lower cost, and simpler deployment. Reserve decoder models for tasks where generation is a core requirement.

=== 5.7 Section Summary

This section covers fine-tuning BERT for three core NLU tasks — text classification, named entity recognition (NER), and extractive question answering — using both the HuggingFace Trainer API and manual PyTorch training loops. Key implementation details include BIO tagging with subword alignment for NER (using the -100 ignore index trick), and offset mapping for extracting precise answer spans in SQuAD-style QA. A worked example demonstrates entity extraction on e-commerce review data, and the section concludes by contrasting encoder-only models (BERT, ModernBERT, DeBERTa-v3) against decoder-only models, recommending encoder models for pure classification/extraction tasks due to their significantly smaller size and faster inference.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
