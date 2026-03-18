#import "../template.typ": *

#let Chapter14() = [
== 14 - OpenAI API Fine-Tuning

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-20-GPT-Finetuning/openai_api_and_finetuning_of_gpt_model.ipynb")[`openai_api_and_finetuning_of_gpt_model.ipynb`]
]

=== 14.1 Supported Methods

#figure(
  table(
    columns: 2,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Method*], [*Supported Models*]),
    [Supervised Fine-tuning], [GPT-5, GPT-4.1, GPT-4.1 mini, GPT-4.1 nano],
    [Vision Fine-tuning], [GPT-5, GPT-4.1],
    [DPO], [GPT-5, GPT-4.1],
    [Reinforcement Fine-tuning], [o3, o4-mini],
  ),
  caption: [OpenAI Fine-Tuning Supported Methods],
  kind: table,
)

=== 14.2 Data Format

Data must be in *JSONL* (JSON Lines) format:

#figure(
```json
{"messages": [
  {"role": "system", "content": "You are a helpful pharma assistant."},
  {"role": "user", "content": "What is metformin used for?"},
  {"role": "assistant", "content": "Metformin is a medication used to treat type 2 diabetes..."}
]}
```
, caption: [OpenAI fine-tuning JSONL format with system, user, and assistant messages],
)

*Requirements:*

- Minimum: 10 examples
- Recommended: 50–100 examples for meaningful improvement
- Maximum: depends on model

=== 14.3 Token Counting and Cost Estimation

Before submitting a fine-tuning job, it is important to estimate the token count of your training data to predict costs. OpenAI uses the `tiktoken` library — the encoding depends on the model: `cl100k_base` for GPT-3.5/GPT-4, and `o200k_base` for GPT-4o and newer models.

#figure(
```python
import tiktoken

# Use "cl100k_base" for GPT-3.5/GPT-4, "o200k_base" for GPT-4o and newer
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(messages):
    total = 0
    for msg in messages:
        total += len(encoding.encode(msg["content"]))
    return total
```
, caption: [Token counting with tiktoken to estimate OpenAI fine-tuning costs],
)

*Pricing structure (per 1M tokens):*

- Training compute: ~\$1.50/hour
- Input tokens: varies by model ($0.20–$4.00)
- Output tokens: varies by model (higher than input)
- Cached input: discounted rate

=== 14.4 Fine-Tuning via Python API

The OpenAI Python SDK provides a three-step workflow for fine-tuning: upload your JSONL training file, create a fine-tuning job specifying the base model and hyperparameters, and then use the resulting fine-tuned model for inference.

#figure(
```python
from openai import OpenAI
client = OpenAI()

# Step 1: Upload training file
file_obj = client.files.create(file=open("data.jsonl", "rb"), purpose="fine-tune")

# Step 2: Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file_obj.id,
    model="gpt-4.1-nano-2025-04-14",
    suffix="custom-model-name",
    method={"type": "supervised"},
    hyperparameters={
        "batch_size": 16,
        "learning_rate_multiplier": 1.0,
        "n_epochs": 3
    }
)

# Step 3: Monitor job
jobs_list = client.fine_tuning.jobs.list()

# Step 4: Use fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-4.1-nano:org:custom-model-name:id",
    messages=[{"role": "user", "content": "Your question"}]
)
```
, caption: [OpenAI fine-tuning API: upload file, create job, monitor, and use fine-tuned model],
)

#pagebreak(weak: true)

=== 14.5 Worked Example: Product Description Generator for E-Commerce

*Scenario:* An online furniture store wants GPT-4.1-nano to generate SEO-optimized product descriptions from raw product specs, matching the brand's warm, lifestyle-oriented voice.

*Training data (`furniture_descriptions.jsonl` - one JSON object per line):*

#figure(
```json
// Example 1 — Dining table (one JSON object per line in actual file)
{
  "messages": [
    {
      "role": "system",
      "content": "You are a copywriter for CozyHome Furniture. Write warm, lifestyle-oriented product descriptions that are SEO-optimized. Include dimensions, materials, and care instructions naturally in the prose."
    },
    {
      "role": "user",
      "content": "Product: Solid oak dining table\nDimensions: 72\"L x 36\"W x 30\"H\nSeats: 6-8\nFinish: Natural oil\nPrice: $1,299"
    },
    {
      "role": "assistant",
      "content": "Gather everyone around the Harvest Oak Dining Table — where Sunday brunches turn into afternoon conversations. Crafted from solid white oak with a natural oil finish that deepens over time, this table seats 6 comfortably. At 72\" long and36\" wide, it anchors your dining room without overwhelming it. Care: wipe with a damp cloth, re-oil once a year."
    }
  ]
}
```
, caption: "OpenAI JSONL training data — Example 1: dining table (formatted for readability; actual file has one JSON object per line)",
)

#figure(
```json
// Example 2 — Accent chair
{
  "messages": [
    {
      "role": "system",
      "content": "You are a copywriter for CozyHome Furniture. Write warm,
                  lifestyle-oriented product descriptions that are
                  SEO-optimized."
    },
    {
      "role": "user",
      "content": "Product: Velvet accent chair\nDimensions: 28\"W x 31\"D
                  x 33\"H\nSeat height: 18\"\nUpholstery: Performance velvet
                  (stain-resistant)\nFrame: Kiln-dried hardwood\nColor
                  options: Emerald, Navy, Blush\nPrice: $549"
    },
    {
      "role": "assistant",
      "content": "Meet your new favorite reading spot. The Luna Accent Chair
                  wraps you in sumptuous performance velvet that laughs off
                  coffee spills. At 28\" wide with an 18\" seat height, it's
                  sized for real comfort. Available in Emerald, Navy, and
                  Blush to complement your existing palette."
    }
  ]
}
```
, caption: [OpenAI JSONL training data — Example 2: accent chair],
)

*Token count estimate:* ~250 tokens per example × 75 examples = ~18,750 training tokens

*Estimated cost:* At GPT-4.1-nano pricing (~$0.10/1 M t r a i n i n g t o k e n s) × 3 e p o c h s = ~$0.006 total. Extremely affordable.

*Before fine-tuning (base GPT-4.1-nano):*

#figure(
```
Prompt: [Same product specs for a walnut bookshelf]
Output: "This walnut bookshelf measures 36x12x72 inches and features 5 shelves. It is
         made of solid walnut and weighs 65 lbs. It can hold books and decorative items.
         Assembly required."  ← generic, specification-list style, no brand voice
```
, caption: [Base GPT-4.1-nano product description: generic specification-list style],
)

*After fine-tuning:*

#figure(
```
Prompt: [Same product specs]
Output: "The Ridgeline Walnut Bookshelf turns your book collection into a design statement.
         Five generously spaced shelves in rich, hand-finished walnut give you room for
         hardcovers, ceramics, and those photo frames you keep meaning to display. At 72\"
         tall and just 12\" deep, it stands tall without eating floor space - perfect for
         hallways, home offices, or that bare wall in the living room..."
         ← brand voice, lifestyle framing, specs woven naturally into prose
```
, caption: [Fine-tuned GPT-4.1-nano product description: brand voice with lifestyle framing],
)

=== 14.6 Section Summary

OpenAI provides fine-tuning through its Python API for GPT models. Data must be in JSONL format with message role/content pairs. The process involves uploading data, creating a fine-tuning job, and monitoring via API or dashboard.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
