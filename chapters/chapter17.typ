#import "../template.typ": *

#let Chapter17() = [
== 17 - Multimodal Fine-Tuning

#blockquote[
  *Notebooks:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-23-Multimodal-LLM-Finetuning/Vision_Model_Finetuning.ipynb")[`Vision_Model_Finetuning.ipynb`] \
  *Data Builder:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-23-Multimodal-LLM-Finetuning/image_data_builder.ipynb")[`image_data_builder.ipynb`]
]

=== 17.1 Vision Language Model Architecture

Multimodal fine-tuning (in the context of images) adapts a model that processes both visual and textual inputs. The architecture has three main components:

#figure(
  image("../diagrams/09-vlm-architecture.png", width: 85%),
  caption: [Vision Language Model Architecture],
)

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Component*], [*Role*], [*Example*]),
    [*Vision Encoder*], [Extracts features from images using patch-based self-attention (not convolutions)], [ViT (Vision Transformer), CLIP],
    [*Projection Layer*], [Converts image features (patches) into the same embedding space as text tokens], [Linear projection of flattened patches],
    [*LLM*], [Processes combined image + text embeddings through attention and MLP layers], [Any decoder-based LLM],
  ),
  caption: [Multimodal Architecture Components],
  kind: table,
)

=== 17.2 Vision Transformer (ViT) Process

1. *Input image* → Split into fixed-size patches (typically 16×16 pixels)
2. *Flatten* each patch into a 1D vector
3. *Linear projection* → Convert to embeddings (analogous to word embeddings for text)
4. *Add positional embeddings* → So the model knows patch positions
5. *Feed to transformer encoder* → Standard attention + MLP processing
6. *Output* → Classification or feature embeddings

*Key insight:* The process mirrors text processing:

- Text: words → tokens → embeddings → positional encoding → attention
- Images: patches → flatten → embedding → positional encoding → attention

#blockquote[
  *Resolution limitation:* Basic ViTs resize all images to a fixed square (e.g., 224×224 or 336×336 pixels), which destroys fine detail — text in documents becomes unreadable, small objects disappear. Modern VLMs (Qwen-VL, LLaMA 3.2 Vision, InternVL) solve this with *dynamic resolution / AnyRes*: the image is sliced into a grid of tiles at its native aspect ratio, each tile is processed independently by the vision encoder, and the resulting patch embeddings are concatenated. This preserves high resolution without retraining the vision encoder on larger inputs.
]

=== 17.3 CLIP Model (Contrastive Language-Image Pre-training)

CLIP is trained on image-caption pairs. It produces aligned text and image embeddings in the same vector space. It is the backbone for both image-to-text (understanding) and text-to-image (generation like DALL-E) tasks.

=== 17.4 Multimodal Transformation Types

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Transformation*], [*Input*], [*Output*], [*Example Models / Tasks*]),
    [Text-to-Text], [Text], [Text], [GPT, LLaMA, T5 (standard LLMs)],
    [Image-to-Text], [Image], [Text], [CLIP, LLaVA, Qwen-VL (image captioning, VQA)],
    [Text-to-Image], [Text], [Image], [DALL-E, Stable Diffusion, Midjourney],
    [Text-to-Speech], [Text], [Audio], [Bark, Orpheus, XTTS],
    [Speech-to-Text], [Audio], [Text], [Whisper, Wav2Vec2],
    [Image+Text-to-Text], [Image + Text], [Text], [GPT-5, Gemini 3, Claude Opus 4.6, LLaVA, Llama 4 (multimodal chat)],
    [Text-to-Video], [Text], [Video], [Sora, RunwayML],
    [Audio+Text-to-Text], [Audio + Text], [Text], [Gemini, Qwen-Audio],
  ),
  caption: [Multimodal Transformation Types],
  kind: table,
)

=== 17.5 Fine-Tuning Options

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Option*], [*What Gets Trained*], [*Trade-offs*]),
    [*Train everything*], [Vision encoder + projection layer + LLM], [Most expensive; best results],
    [*Freeze vision encoder*], [Projection layer + LLM only], [Moderate cost; good results],
    [*Freeze vision encoder + LLM*], [Projection layer only], [Cheapest; limited adaptation],
  ),
  caption: [Multimodal Fine-Tuning Options],
  kind: table,
)

#pagebreak(weak: true)

==== 17.5.1 The 4 Fine-Tuning Strategies for Multimodal Models

#figure(
  table(
    columns: 5,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Strategy*], [*What Is Trained*], [*What Is Frozen*], [*Cost*], [*Best For*]),
    [*Strategy 1: Full Fine-Tuning*], [Vision encoder + projection + LLM (all weights)], [Nothing], [Very high (multi-GPU)], [Maximum accuracy, large labeled datasets],
    [*Strategy 2: LLM + Projection*], [Projection layer + LLM (with LoRA)], [Vision encoder], [Moderate], [Most practical default - vision features transfer well],
    [*Strategy 3: Projection Only*], [Projection layer only], [Vision encoder + LLM], [Low], [Quick domain adaptation, very limited data],
    [*Strategy 4: LoRA on All Components*], [LoRA adapters on vision encoder + projection + LLM], [Base weights of all components], [Low-Moderate], [Best balance of cost and quality for most use cases],
  ),
  caption: [Four Fine-Tuning Strategies for Multimodal Models],
  kind: table,
)

=== 17.6 Data Formats for Vision Fine-Tuning

Data pairs typically consist of:

- An image
- A text instruction/question about the image
- The expected text response

Formats include VQA (Visual Question Answering) style and LLaVA-style conversational data.

#pagebreak(weak: true)

=== 17.7 Worked Example: Chest X-Ray Report Generation

*Scenario:* A radiology department wants to fine-tune a vision-language model to generate preliminary radiology reports from chest X-ray images, flagging abnormalities for radiologist review.

*Training data (VQA-style image-text pairs):*

#figure(
```json
[
  {
    "image": "xrays/patient_0042_PA.jpg",
    "conversations": [
      {"from": "user", "value": "<image>\nGenerate a structured radiology report for this PA chest X-ray."},
      {"from": "assistant", "value": "**FINDINGS:**\n\n**Heart:** Mildly enlarged cardiac silhouette (cardiothoracic ratio ~0.55). No pericardial effusion.\n\n**Lungs:** Bilateral lower lobe airspace opacities, more prominent on the right. Air bronchograms visible in the right lower lobe, suggesting consolidation. No pneumothorax. No pleural effusion.\n\n**Mediastinum:** Normal mediastinal contour. No widening. Trachea midline.\n\n**Bones/Soft tissue:** No acute osseous abnormality. Degenerative changes in the thoracic spine.\n\n**IMPRESSION:**\n1. Right lower lobe consolidation - correlate with clinical history for pneumonia vs. aspiration\n2. Mild cardiomegaly - recommend comparison with prior studies\n\n**RECOMMENDATION:** Clinical correlation recommended. Consider CT chest if no improvement after antibiotic therapy."}
    ]
  },
  {
    "image": "xrays/patient_0089_PA.jpg",
    "conversations": [
      {"from": "user", "value": "<image>\nGenerate a structured radiology report for this PA chest X-ray."},
      {"from": "assistant", "value": "**FINDINGS:**\n\n**Heart:** Normal cardiac silhouette size (cardiothoracic ratio ~0.45).\n\n**Lungs:** Clear bilateral lung fields. No focal consolidation, mass, or nodule. No pleural effusion. No pneumothorax. Costophrenic angles are sharp bilaterally.\n\n**Mediastinum:** Normal mediastinal width. Trachea midline. Aortic knob is unremarkable.\n\n**Bones/Soft tissue:** No acute fracture or dislocation. No soft tissue abnormality.\n\n**IMPRESSION:**\nNo acute cardiopulmonary abnormality. Normal chest radiograph.\n\n**RECOMMENDATION:** No follow-up imaging required based on this study."}
    ]
  }
]
```
, caption: [VQA-style training data for chest X-ray radiology report generation],
)

#pagebreak(weak: true)

*Architecture during fine-tuning:*

#figure(
  image("../diagrams/16-vlm-architecture.png", width: 75%),
  caption: [VLM fine-tuning architecture: frozen encoder, trained projection and LoRA LLM],
)

*Fine-tuning strategy:* Freeze vision encoder (CLIP already understands visual features), train the projection layer (to map X-ray features to text-relevant embeddings), and apply LoRA to the LLM (to learn radiology report language and structure).

*Before fine-tuning (base vision-language model):*

#figure(
```
Input: [Chest X-ray with right lower lobe consolidation]
Output: "This is a chest X-ray image. It shows the ribcage and lungs. There appears
         to be some cloudy area in the lower right portion of the image."
         ← generic image description, no medical terminology, no structure
```
, caption: [Base VLM chest X-ray output: generic image description without medical structure],
)

*After fine-tuning:*

#figure(
```
Input: [Same chest X-ray]
Output: "FINDINGS: ... Right lower lobe consolidation with air bronchograms ...
         IMPRESSION: 1. Right lower lobe consolidation - correlate for pneumonia ..."
         ← structured medical report, correct terminology, actionable impression
```
, caption: [Fine-tuned VLM chest X-ray output: structured radiology report with correct terminology],
)

*What changed:* The projection layer learned to map X-ray visual features (opacity patterns, cardiac silhouette shape, lung field clarity) into the embedding space where the LLM can reason about them medically. The LoRA-adapted LLM learned to output structured radiology report format.

#pagebreak(weak: true)

*Inference after fine-tuning — generating reports from new X-rays:*

#figure(
```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# Load the fine-tuned multimodal model
model = LlavaForConditionalGeneration.from_pretrained("./xray-report-generator")
processor = AutoProcessor.from_pretrained("./xray-report-generator")

# Generate report from a new chest X-ray
image = Image.open("xrays/new_patient_PA.jpg")
prompt = "<image>\nGenerate a structured radiology report for this PA chest X-ray."

inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.3)
report = processor.decode(outputs[0], skip_special_tokens=True)
print(report)
# Expected: structured FINDINGS → IMPRESSION → RECOMMENDATION format
# with correct medical terminology and actionable clinical guidance
```
, caption: [Fine-tuned LLaVA inference: generating structured radiology reports from X-rays],
)

#blockquote[
  *Clinical deployment note:* AI-generated radiology reports should always be reviewed and approved by a qualified radiologist before becoming part of the patient record. The model generates _preliminary_ reports to accelerate workflow, not replace clinical judgment.
]

=== 17.8 Section Summary

Multimodal fine-tuning combines a vision encoder (ViT/CLIP) with a projection layer and LLM. Image patches are converted to embeddings just as text tokens are, enabling unified attention processing. Fine-tuning can target the projection layer alone, the LLM, or the entire pipeline.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
