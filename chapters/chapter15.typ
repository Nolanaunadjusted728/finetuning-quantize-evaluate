#import "../template.typ": *

#let Chapter15() = [
== 15 - Google Vertex AI / Gemini Fine-Tuning

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-21-GEMINI-Finetuning/gemini_finetuning_clean.ipynb")[`gemini_finetuning_clean.ipynb`]
]

=== 15.1 Platform Overview

Google's fine-tuning is available through *Vertex AI* on Google Cloud Platform (GCP), accessed via the *Google GenAI SDK*. Two fine-tuning approaches are supported:

1. *Supervised Fine-tuning (SFT)*
2. *Preference Tuning (DPO/RLHF)*

Both *PEFT (LoRA)* and *Full Fine-tuning* are supported.

*Supported modalities:* Text, documents (PDF/docx), images, audio, video.

=== 15.2 Available Models and Pricing

#figure(
  table(
    columns: 2,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Model*], [*Pricing (per 1M tokens)*]),
    [Gemini 3 Pro], [~\$25],
    [Gemini 3 Flash], [~\$5],
    [Gemini 3 Light], [~\$1.50],
    [Open-source models (Gemma, Llama)], [Varies],
  ),
  caption: [Google Vertex AI Available Models and Pricing],
  kind: table,
)

=== 15.3 Setup Requirements

1. Create a GCP project and copy the *Project ID*
2. Set up billing on GCP console (required for Vertex AI model access)
3. Select a *location* (e.g., `us-central1`)
4. Authenticate from Colab: `google.colab.auth.authenticate_user()`
5. Use the same Google account for both GCP and Colab

#figure(
```python
from google import genai
from google.genai.types import HttpOptions, CreateTuningJobConfig, TuningDataset

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1")
)

# Verify connectivity
response = client.models.generate_content(model=MODEL_NAME, contents="Hello")
```
, caption: [Google GenAI SDK client setup with Vertex AI project and location],
)

#pagebreak(weak: true)

=== 15.4 Fine-Tuning Process

The tuning data format follows instruction/response pairs in JSON, with support for text, documents, images, audio, and video inputs. The fine-tuning job is created via the GenAI SDK's `CreateTuningJobConfig`.

*Data format:* Training data is stored as JSONL files in Google Cloud Storage. Each line contains an instruction/response pair, similar to OpenAI's format but with Google-specific fields for multimodal inputs (`document`, `image`, `audio`, `video` fields alongside `text`). The `input`/`output` structure maps to the model's instruction-following behavior.

*Full fine-tuning job creation:*

#figure(
```python
from google.genai.types import CreateTuningJobConfig, TuningDataset

tuning_job = client.tuning_jobs.create(
    config=CreateTuningJobConfig(
        source_model="gemini-3-flash",
        training_dataset=TuningDataset(
            gcs_uri="gs://your-bucket/train.jsonl",
        ),
        tuned_model_display_name="my-fine-tuned-gemini",
        epoch_count=3,
        learning_rate_multiplier=1.0,
    )
)

# Monitor training progress
while not tuning_job.has_ended:
    tuning_job = client.tuning_jobs.get(name=tuning_job.name)
    print(f"State: {tuning_job.state}, Step: {tuning_job.experiment}")

# Use the fine-tuned model
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents="Your prompt here"
)
```
, caption: [Vertex AI fine-tuning job: create, monitor, and use tuned Gemini model],
)

*Key configuration parameters:*

- `source_model`: The base Gemini model to fine-tune (e.g., `gemini-3-flash`)
- `training_dataset`: Google Cloud Storage (GCS) URI pointing to your JSONL training file
- `tuned_model_display_name`: Human-readable name for the resulting model
- `epoch_count`: Number of training passes over the dataset (typically 2-5)
- `learning_rate_multiplier`: Scaling factor for the learning rate (1.0 = default, lower for more conservative updates)

#pagebreak(weak: true)

=== 15.5 Worked Example: Invoice Data Extraction with Multimodal Gemini

*Scenario:* An accounting firm receives invoices in varied formats (PDFs, photos of paper invoices, scanned documents). They want to fine-tune Gemini 3 Flash to extract structured data from any invoice format.

*Training data (multimodal - image + text pairs):*

#figure(
```json
[
  {
    "input": {
      "text": "Extract all line items, totals, and payment terms from this invoice.",
      "document": "gs://my-bucket/invoices/invoice_001.pdf"
    },
    "output": "{\n  \"vendor\": \"Acme Industrial Supplies\",\n  \"invoice_number\": \"INV-2025-0847\",\n  \"date\": \"2025-11-15\",\n  \"due_date\": \"2025-12-15\",\n  \"line_items\": [\n    {\"description\": \"Steel bolts M10x40 (box of 500)\", \"qty\": 3, \"unit_price\": 42.50, \"total\": 127.50},\n    {\"description\": \"Safety goggles EN166\", \"qty\": 25, \"unit_price\": 8.99, \"total\": 224.75},\n    {\"description\": \"Welding rods 2.5mm (5kg)\", \"qty\": 2, \"unit_price\": 35.00, \"total\": 70.00}\n  ],\n  \"subtotal\": 422.25,\n  \"tax_rate\": \"20%\",\n  \"tax_amount\": 84.45,\n  \"total\": 506.70,\n  \"payment_terms\": \"Net 30\",\n  \"bank_details\": \"Sort: 12-34-56, Account: 12345678\"\n}"
  },
  {
    "input": {
      "text": "Extract all line items, totals, and payment terms from this invoice.",
      "image": "gs://my-bucket/invoices/handwritten_invoice_photo.jpg"
    },
    "output": "{\n  \"vendor\": \"Bob's Plumbing Services\",\n  \"invoice_number\": \"BP-392\",\n  \"date\": \"2025-10-28\",\n  \"line_items\": [\n    {\"description\": \"Emergency callout fee\", \"qty\": 1, \"unit_price\": 75.00, \"total\": 75.00},\n    {\"description\": \"Labor (2.5 hours @ $65/hr)\", \"qty\": 2.5, \"unit_price\": 65.00, \"total\": 162.50},\n    {\"description\": \"Replacement valve assembly\", \"qty\": 1, \"unit_price\": 48.00, \"total\": 48.00}\n  ],\n  \"subtotal\": 285.50,\n  \"tax_rate\": \"0%\",\n  \"tax_amount\": 0.00,\n  \"total\": 285.50,\n  \"payment_terms\": \"Due on receipt\"\n}"
  }
]
```
, caption: [Multimodal Gemini training data: invoice extraction with document and image inputs],
)

*Why Vertex AI / Gemini for this task:*

- *Native multimodal support:* Gemini processes PDFs, images, and scanned documents natively - no separate OCR pipeline needed
- *Document understanding:* The model sees the visual layout (tables, columns, headers) not just extracted text
- *Varied formats:* Training on professional PDFs, handwritten invoices, and photographed receipts teaches the model to handle real-world document variety

#pagebreak(weak: true)

*Before fine-tuning (base Gemini 3 Flash):*

#figure(
```
Prompt: [Photo of handwritten invoice] "Extract all line items and totals."
Output: "This appears to be a handwritten invoice from a plumbing company. I can see
         some items listed including a callout fee and labor charges. The total appears
         to be around $285."  ← vague, unstructured, missing details
```
, caption: [Base Gemini 3 Flash: vague unstructured invoice extraction],
)

*After fine-tuning:*

#figure(
```
Prompt: [Same photo]
Output: {"vendor": "Bob's Plumbing Services", "invoice_number": "BP-392", ...}
         ← precise JSON extraction, correct values, consistent schema
```
, caption: [Fine-tuned Gemini: precise JSON invoice extraction with correct schema],
)

*Inference after fine-tuning — testing the fine-tuned Gemini model:*

#figure(
```python
from google import genai

client = genai.Client()

# Use the fine-tuned model endpoint from the completed tuning job
response = client.models.generate_content(
    model=tuning_job.tuned_model.endpoint,
    contents={
        "text": "Extract all line items, totals, and payment terms from this invoice.",
        "document": "gs://my-bucket/invoices/new_invoice_test.pdf",
    },
)
print(response.text)
# Expected: structured JSON with vendor, invoice_number, line_items, totals, payment_terms

# Batch inference for processing multiple invoices
import json

test_invoices = ["gs://my-bucket/invoices/test_001.pdf", "gs://my-bucket/invoices/test_002.jpg"]
for invoice_path in test_invoices:
    field = "document" if invoice_path.endswith(".pdf") else "image"
    response = client.models.generate_content(
        model=tuning_job.tuned_model.endpoint,
        contents={
            "text": "Extract all line items, totals, and payment terms from this invoice.",
            field: invoice_path,
        },
    )
    extracted = json.loads(response.text)
    print(f"{invoice_path}: {extracted['vendor']} — ${extracted['total']}")
```
, caption: [Fine-tuned Gemini inference: single and batch invoice extraction],
)

=== 15.6 Section Summary

Google Vertex AI supports fine-tuning Gemini models via the Google GenAI SDK. It requires GCP billing setup and project authentication. Both SFT and preference tuning are supported with text, document, and multimodal data.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
