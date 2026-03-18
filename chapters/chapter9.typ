#import "../template.typ": *

#let Chapter9() = [
== 9 - LLM Quantization

#blockquote[
  *Notebooks:*
  - #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-12-13-LLM-Quantization/LLM_Quantization/Model_Quantization_Final.ipynb")[`Model_Quantization_Final.ipynb`] - Foundational concepts
  - #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-12-13-LLM-Quantization/LLM_Quantization/LLM_Quantization_GPTQ.ipynb")[`LLM_Quantization_GPTQ.ipynb`] - GPTQ
  - #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-12-13-LLM-Quantization/LLM_Quantization/LLM_Quantization_AWQ.ipynb")[`LLM_Quantization_AWQ.ipynb`] - AWQ
  - #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-12-13-LLM-Quantization/LLM-Quantization-Part-2/QAT_in_LLM.ipynb")[`QAT_in_LLM.ipynb`] - QAT
  - #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-12-13-LLM-Quantization/LLM-Quantization-Part-2/gguf_ggml_practical.ipynb")[`gguf_ggml_practical.ipynb`] - GGUF/llama.cpp
]

=== 9.1 What Is Quantization

Quantization reduces model precision from high-precision floating point (FP32/FP16) to lower-precision integers (INT8/INT4), shrinking model size and accelerating inference.

#figure(
```
FP32 (32 bits per weight) → INT8 (8 bits) = 4× smaller
FP32 (32 bits per weight) → INT4 (4 bits) = 8× smaller
```
, caption: [Quantization size reduction: FP32 to INT8 (4x) and INT4 (8x)],
)

==== 9.1.1 Data Types and Memory Consumption

#figure(
  table(
    columns: 6,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Data Type*], [*Bits*], [*Memory per Param*], [*100M Params*], [*1B Params*], [*7B Params*]),
    [FP64 (double)], [64], [8 bytes], [800 MB], [8 GB], [56 GB],
    [FP32 (float)], [32], [4 bytes], [400 MB], [4 GB], [28 GB],
    [FP16 (half)], [16], [2 bytes], [200 MB], [2 GB], [14 GB],
    [BF16 (bfloat16)], [16], [2 bytes], [200 MB], [2 GB], [14 GB],
    [INT8], [8], [1 byte], [100 MB], [1 GB], [7 GB],
    [INT4], [4], [0.5 bytes], [50 MB], [500 MB], [3.5 GB],
  ),
  caption: [Data Types and Memory Consumption],
  kind: table,
)

==== 9.1.2 Quantization Analogies from the Course

- *Post-Training Quantization (PTQ):* "Train with a perfect cricket bat, play with a cheap bat" - the model is trained at full precision but compressed for deployment
- *QAT:* "Train and play with the same cheap bat" - the model learns to compensate for quantization noise during training
- *GPTQ:* "Check body shape, try fitting each shirt" - examines weight sensitivity via Hessian before quantizing
- *AWQ:* "Measures how the final dish tastes" - focuses on post-activation error to preserve output quality

#pagebreak(weak: true)

=== 9.2 Quantization Methods Comparison

#figure(
  table(
    columns: 6,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Method*], [*Type*], [*When Applied*], [*Calibration Data*], [*Quality*], [*Speed*]),
    [*PTQ Dynamic*], [Post-training], [Inference time], [None], [Good for linear layers], [Fast to apply],
    [*PTQ Static*], [Post-training], [Before deployment], [Yes (representative samples)], [Better than dynamic], [Fast to apply],
    [*QAT* (Quantization-Aware Training)], [During training], [Training time], [Full training data], [Best quality], [Slow (requires retraining)],
    [*GPTQ* (Generative Pre-trained Transformer Quantization)], [Post-training], [Before deployment], [Small calibration set], [Very good for LLMs], [~minutes],
    [*AWQ* (Activation-aware Weight Quantization)], [Post-training], [Before deployment], [Small calibration set], [Excellent (activation-aware)], [~minutes],
  ),
  caption: [Quantization Methods Comparison],
  kind: table,
)

=== 9.3 Scale and Zero-Point - The Core Math

Quantization maps floating-point values to integers using two parameters. There are two variants:

- *Symmetric quantization:* The zero-point is fixed at 0, and the range is centered around zero: $[-alpha, +alpha]$. Faster on hardware (simpler arithmetic) but wastes representational capacity on skewed weight distributions.
- *Asymmetric quantization:* The zero-point is a free parameter, allowing the quantized range to shift to match the actual weight distribution. More accurate for non-symmetric distributions (common in LLM weights) but slightly slower due to the extra subtraction.

AWQ uses asymmetric quantization by default (`zero_point: True`), which is why it achieves better quality for LLMs whose weight distributions are often skewed.

*The formulas:*

#figure($ "scale" = frac(max - min, q_(max) - q_(min)) $, caption: [Quantization scale factor], kind: math.equation)

#figure($ "zero_point" = "round"(frac(-min, "scale")) $, caption: [Quantization zero point offset], kind: math.equation)

#figure($ q = "clamp"("round"(frac(x, "scale") + "zero_point"), q_(min), q_(max)) $, caption: [Asymmetric quantization: float to integer mapping], kind: math.equation)

For INT8: $q_(min) = -128$, $q_(max) = 127$. For INT4: $q_(min) = -8$, $q_(max) = 7$.

*Dequantization* (recovering approximate original values):

#figure($ hat(x) = (q - "zero_point") times "scale" $, caption: [Dequantization: integer back to approximate float], kind: math.equation)

The difference between $x$ and $hat(x)$ is the *quantization error* - what you lose in precision.

=== 9.4 GPTQ (Generative Pre-Trained Transformer Quantization)

#link("https://arxiv.org/abs/2210.17323")[GPTQ] (Frantar et al., 2022) quantizes weights layer-by-layer using a calibration dataset. It minimizes the output error of each layer independently using approximate second-order information (Hessian).

#figure(
```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Configure quantization
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,     # Quantize in groups of 128 weights
    desc_act=False      # No activation reordering
)

# Quantize with calibration data
model = AutoGPTQForCausalLM.from_pretrained("tiiuae/falcon-rw-1b", quantize_config)
model.quantize(calibration_data)  # 5-10 example texts
model.save_quantized("falcon-1b-gptq-4bit")

# Load and use pre-quantized models (most common)
model = AutoGPTQForCausalLM.from_quantized("TheBloke/Llama-2-7B-Chat-GPTQ")
```
, caption: [GPTQ 4-bit quantization with calibration data and loading pre-quantized models],
)

=== 9.5 AWQ (Activation-Aware Weight Quantization)

#link("https://arxiv.org/abs/2306.00978")[AWQ] (Lin et al., 2023) identifies which weights are most important by examining _activation patterns_ - weights that frequently interact with large activations are kept at higher precision.

#figure(
```python
from awq import AutoAWQForCausalLM

quant_config = {
    "w_bit": 4,
    "q_group_size": 128,
    "zero_point": True,     # Asymmetric quantization
    "version": "GEMM"       # Matrix multiplication kernel variant
}

model = AutoAWQForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model.quantize(tokenizer, quant_config=quant_config)
```
, caption: [AWQ activation-aware 4-bit quantization with asymmetric zero-point],
)

*Note:* The `autoawq` library has reduced maintenance activity. For new projects, consider `llm-compressor` from vLLM as an alternative, or load pre-quantized models (e.g., from TheBloke on HuggingFace).

=== 9.6 QAT (Quantization-Aware Training)

QAT inserts "fake quantization" nodes during training so the model learns to be robust to quantization noise. The notebooks demonstrate three quantization approaches, though only the third is true QAT:

*Approach 1 - BitsAndBytes (PTQ — post-training quantization, most practical for LLMs):*

#figure(
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True       # Quantize the quantization constants too
)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
```
, caption: [BitsAndBytes 4-bit NF4 quantization config for post-training quantization],
)

*Approach 2 - LoRA + Quantization (QLoRA — also PTQ, with LoRA adapters trained on top):*

This is the standard QLoRA approach covered in Sections 2-6. Load in 4-bit, apply LoRA adapters, train only the adapters.

*Approach 3 - Custom learnable quantization (true QAT — scale and zero-point are learned during training):*

#figure(
```python
class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(1))       # Learnable!
        self.zero_point = nn.Parameter(torch.zeros(1))  # Learnable!

    def fake_quantize(self, x):
        x_q = torch.clamp(torch.round(x / self.scale + self.zero_point), -128, 127)
        x_dq = (x_q - self.zero_point) * self.scale  # Dequantize
        # Straight-Through Estimator (STE): bypass rounding in backward pass
        return (x_dq - x).detach() + x
```
, caption: [Custom QAT layer with learnable scale/zero-point and Straight-Through Estimator],
)

#blockquote[
  *Critical implementation detail:* `torch.round()` has zero gradient everywhere, so naive fake quantization blocks gradient flow entirely. The *Straight-Through Estimator (STE)* solves this by using `(rounded - x).detach() + x` — during the forward pass, the output equals the rounded value, but during the backward pass, gradients flow through as if rounding never happened. STE is the foundational trick that makes all QAT work.
]

=== 9.7 GGUF / GGML and llama.cpp

*GGML* (Georgi Gerganov Machine Learning) is a C-based tensor library and inference engine - no Python dependency. *GPT-Generated Unified Format (GGUF)* is the file format for storing quantized models.

*Conversion and quantization workflow:*

#figure(
```bash
# 1. Convert HuggingFace model to GGUF format
python3 convert_hf_to_gguf.py ./tinyllama-hf --outfile ./tinyllama.gguf

# 2. Quantize to 4-bit
./bin/quantize ./tinyllama.gguf ./tinyllama-q4_0.gguf q4_0

# 3. Run inference
./bin/llama-cli -m ./tinyllama-q4_0.gguf -p "What is machine learning?" -n 100
```
, caption: [GGUF conversion, 4-bit quantization, and inference with llama.cpp],
)

*Available quantization levels:*

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Format*], [*Bits*], [*Size (7B model)*], [*Quality*]),
    [`q2_K`], [2-bit], [~2.7 GB], [Lowest - significant degradation],
    [`q4_0`], [4-bit], [~3.8 GB], [Good balance of size and quality],
    [`q4_K_M`], [4-bit (mixed)], [~4.1 GB], [Better than q4_0, uses more bits for important layers],
    [`q5_K_M`], [5-bit (mixed)], [~4.8 GB], [Near-original quality],
    [`q8_0`], [8-bit], [~7.2 GB], [Minimal quality loss],
    [`f16`], [16-bit], [~13.5 GB], [Full precision (no quantization)],
  ),
  caption: [Available Quantization Levels],
  kind: table,
)

*Python bindings for GGUF models:*

#figure(
```python
from llama_cpp import Llama

llm = Llama(model_path="./tinyllama-q4_0.gguf", n_ctx=2048)
output = llm("What is LoRA?", max_tokens=100)
print(output["choices"][0]["text"])
```
, caption: [Python inference with GGUF quantized model using llama-cpp-python],
)

=== 9.8 Worked Example: Deploying a 7B Model on Consumer Hardware

*Scenario:* You want to run Llama 2 7B on a laptop with 8GB RAM (no GPU).

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Format*], [*RAM Required*], [*Speed (tokens/sec on M2 Mac)*], [*Quality (perplexity)*]),
    [FP16 (original)], [~14 GB], [Cannot load], [5.79 (baseline)],
    [GPTQ 4-bit (GPU)], [~4 GB VRAM], [~40 tok/s], [5.85 (+1%)],
    [GGUF q4_K_M (CPU)], [~4.1 GB RAM], [~15 tok/s], [5.86 (+1.2%)],
    [GGUF q2_K (CPU)], [~2.7 GB RAM], [~20 tok/s], [6.71 (+16%)],
  ),
  caption: [Model Size at Different Precision Levels],
  kind: table,
)

*Recommendation:* `q4_K_M` is the sweet spot - 4.1 GB fits in 8GB RAM, quality loss is \<2%, and it runs entirely on CPU via llama.cpp. Use `q8_0` if you have 8GB+ RAM to spare.

=== 9.9 Section Summary

This section surveys LLM quantization techniques for reducing model precision from FP32/FP16 to INT8/INT4, covering the core math of scale and zero-point computation for both symmetric and asymmetric quantization. It compares five methods — PTQ (dynamic/static), GPTQ (Hessian-based layer-wise quantization), AWQ (activation-aware weight quantization), and QAT (with learnable scale/zero-point using the Straight-Through Estimator) — and details the GGUF/llama.cpp pipeline for CPU-only deployment. A practical worked example shows that GGUF q4_K_M quantization reduces a 7B model from ~14 GB to ~4.1 GB with less than 2% perplexity degradation, making it runnable on consumer hardware with 8 GB RAM.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
