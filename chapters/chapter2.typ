#import "../template.typ": *

#let Chapter2() = [
== 2 - Why Fine-Tuning Was Hard Before Transformers (LSTM Era)

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-05-Why-Finetuning-Hard-in-LSTM/Why_finetuning_was_challanging_in_LSTM%20(1).ipynb")[`Why_finetuning_was_challenging_in_LSTM.ipynb`]
]

=== 2.1 The Core Problem

Before transformers, fine-tuning a pre-trained model for a different task required *manually re-wiring the architecture*. Unlike modern LLMs where the same model can handle classification, generation, translation, and summarization through prompt engineering, LSTM-based models were structurally locked to their original task.

=== 2.2 Practical Demonstration

The notebook builds an LSTM sentiment classifier on IMDB, then attempts to repurpose it for summarization (sequence-to-sequence):

#figure(
```python
# Original: LSTM sentiment classifier
model = Sequential([
    Embedding(vocab_size=10000, output_dim=128),
    LSTM(128),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Attempted reuse: Extract encoder states for a seq2seq task
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
# Feed these states into a NEW decoder LSTM for text generation
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
```
, caption: [LSTM sentiment classifier repurposed for seq2seq, demonstrating transfer failure],
)

*Result:* ~0.01% accuracy on the seq2seq task. The model's learned representations were too task-specific to transfer.

=== 2.3 Why Transformers Solved This

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Problem*], [*LSTM Era*], [*Transformer Era*]),
    [Task transfer], [Manual architecture surgery], [Same architecture, different prompts],
    [Vocabulary mismatch], [Hard crash (dimension errors)], [Shared tokenizer across tasks],
    [Knowledge reuse], [Only lower layers transferable], [Attention patterns transfer across tasks],
    [Fine-tuning scope], [Freeze/unfreeze entire layers], [LoRA adapts within any layer],
  ),
  caption: [Transformer vs LSTM Comparison],
  kind: table,
)

*Key insight:* Transformers' self-attention mechanism learns _general_ language representations that transfer across tasks. LSTMs learn _sequential patterns_ that are task-specific. This is why the shift from LSTMs to transformers unlocked the entire modern fine-tuning paradigm.

#figure(
  image("../diagrams/02-lstm-vs-transformer.png", width: 85%),
  caption: [Transfer Learning: LSTM Era vs Transformer Era],
)

=== 2.4 Section Summary

Before transformers, fine-tuning required manual architecture surgery — an LSTM trained for classification could not be repurposed for generation without rebuilding the decoder, and learned representations were too task-specific to transfer (demonstrated by ~0.01% accuracy when repurposing an IMDB sentiment classifier for seq2seq summarization). Transformers solved this because self-attention learns general language representations that transfer across tasks, replacing per-task architectural rewiring with a single flexible architecture amenable to prompt engineering and parameter-efficient methods like LoRA.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
