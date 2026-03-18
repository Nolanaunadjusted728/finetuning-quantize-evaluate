#import "../template.typ": *

#let Chapter8() = [
== 8 - Preference Alignment with DPO

#blockquote[
  *Notebook:* #link("https://github.com/sunnysavita10/Complete-LLM-Finetuning/blob/main/LLM%20Fine-Tuning-16-Preference-based-training/Preference_Aligned_Training_DPO_final.ipynb")[`Preference_Aligned_Training_DPO_final.ipynb`]
]

=== 8.1 The DPO Loss Function

==== Key Equation: DPO Training Loss

#figure($ cal(L)_("DPO")(pi_theta; pi_("ref")) = -bb(E)[log sigma(beta (log frac(pi_theta(y^+ | x), pi_("ref")(y^+ | x)) - log frac(pi_theta(y^- | x), pi_("ref")(y^- | x))))] $, caption: [DPO loss: direct preference optimization without reward model], kind: math.equation)

*Name and Context:* This is the #link("https://arxiv.org/abs/2305.18290")[Direct Preference Optimization (DPO)] training loss, introduced by Stanford University researchers (Rafailov et al., 2023). It replaces the complex RLHF pipeline (which requires a separate reward model and PPO training) with a single supervised loss function.

*Symbol Definitions:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Symbol*], [*Meaning*], [*Notes*]),
    [$pi_theta$], [The policy model being trained], [The preference-aligned model],
    [$pi_("ref")$], [The reference model], [The instruction-tuned model (frozen, used for comparison)],
    [$y^+$], [Chosen (preferred) response], [Human- or AI-selected as the better answer],
    [$y^-$], [Rejected response], [The response deemed worse],
    [$x$], [The input prompt], [The question or instruction],
    [$sigma$], [Sigmoid function], [Compresses values to range \[0, 1\]],
    [$beta$], [Scaling factor (temperature)], [Controls how strongly the model aligns to chosen responses; typical range: 0.1–0.5],
    [$bb(E)$], [Expectation], [We want to minimize the expected negative log-likelihood],
  ),
  caption: [DPO Loss Function Symbol Definitions],
  kind: table,
)

*Plain English Explanation:*

The DPO loss measures how much more the model prefers the chosen answer over the rejected answer, _relative to a reference model_. It does this by:

1. Computing how much the training model's probability for the chosen response exceeds the reference model's probability (the "chosen improvement")
2. Computing the same for the rejected response (the "rejected improvement")
3. Taking the difference: chosen improvement − rejected improvement
4. Scaling by β and passing through sigmoid to get a probability
5. Taking the negative log to create a loss to minimize

*Interpretation:*

- If the difference is *positive*, the model prefers the chosen response more than the rejected one → loss decreases → good
- If the difference is *negative*, the model is misaligned → loss increases → model adjusts
- Higher *β* forces stronger alignment to chosen responses
- Lower *β* allows more exploration

*Connection to Surrounding Content:*

DPO eliminates the need for a separate reward model (required by RLHF/PPO). The reference model π_ref is simply the instruction-tuned model from the previous stage, frozen during DPO training. The trained model π_θ starts as a copy of π_ref and is updated to prefer chosen responses.

#blockquote[
  *VRAM warning:* Vanilla DPO requires roughly *2× the VRAM* of SFT because both the active policy model and the frozen reference model must reside in GPU memory simultaneously. However, when using *PEFT/LoRA*, TRL's `DPOTrainer` supports `ref_model=None` — it computes reference log-probabilities by temporarily disabling the LoRA adapter, avoiding a separate reference model entirely. This is the recommended approach for memory-constrained setups:
  
  ```python
  trainer = DPOTrainer(
      model=model,        \# LoRA-adapted model
      ref_model=None,     \# No separate reference model — uses adapter toggling
      args=training_args,
      train_dataset=dpo_dataset,
  )
  ```
  
  Without this trick, expect DPO to need ~2× the VRAM of SFT. With `ref_model=None` and LoRA, DPO VRAM usage is comparable to SFT.
]

=== 8.2 DPO Data Format

The data has three columns:

#figure(
  table(
    columns: 2,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Column*], [*Description*]),
    [`prompt`], [The question/instruction given to the model],
    [`chosen`], [The preferred/correct response],
    [`rejected`], [The dispreferred/incorrect response],
  ),
  caption: [DPO Training Data Format],
  kind: table,
)

Example datasets: Anthropic HH-RLHF, UltraFeedback, various Hugging Face preference datasets.

#pagebreak(weak: true)

=== 8.3 LoRA Mathematics: Why Low-Rank Decomposition Works

The core idea behind LoRA (Low-Rank Adaptation) is that the weight updates during fine-tuning don't need the full dimensionality of the original weight matrices. Instead of updating all parameters, LoRA decomposes the update into two small matrices whose product approximates the full update.

#figure(
  image("../diagrams/05-lora-decomposition.png", width: 85%),
  caption: [LoRA Low-Rank Decomposition],
)

*The Key Equation:*

#figure($ W' = W_0 + Delta W = W_0 + B A $, caption: [LoRA weight update via low-rank matrix decomposition], kind: math.equation)

where the update $Delta W$ is decomposed as a product of two low-rank matrices $B$ and $A$.

*Symbol Definitions:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Symbol*], [*Meaning*], [*Notes*]),
    [$W_0$], [Frozen pretrained weight matrix], [Original weights from the base model, never modified during training],
    [$W'$], [Effective weight matrix after adaptation], [What the model actually uses at inference time],
    [$Delta W$], [The weight update learned during fine-tuning], [Decomposed as $B A$ instead of stored as a full matrix],
    [$B$], [Up-projection matrix of shape $(d times r)$], [Initialized to zeros so $Delta W = 0$ at the start of training; projects from rank-$r$ space back to $d$-dimensional space],
    [$A$], [Down-projection matrix of shape $(r times d)$], [Initialized with random Gaussian values; projects from $d$-dimensional input down to rank-$r$ space],
    [$d$], [Hidden dimension of the original weight matrix], [For LLaMA 2 7B, $d = 4096$],
    [$r$], [Rank of the decomposition], [Typically $r i n \{8, 16, 32, 64\}$, with $r l t.d o u b l e d$],
    [$alpha$], [LoRA scaling factor], [Controls the magnitude of the learned update],
  ),
  caption: [LoRA Low-Rank Decomposition Symbol Definitions],
  kind: table,
)

*Why This Works — The Parameter Efficiency Argument:*

A weight matrix $W i n bb(R)^(d times d)$ has $d^2$ parameters. The LoRA update $Delta W = B A$ requires only:

#figure($ (d times r) + (r times d) = 2 d r " parameters" $, caption: [LoRA parameter count: two low-rank matrices], kind: math.equation)

For concrete numbers with d = 4096 and r = 16:

- Full update: d² = 4096² = 16,777,216 parameters
- LoRA update: 2dr = 2 x 4096 x 16 = 131,072 parameters
- *Ratio: only 0.78% of the full parameter count*

*What "Rank" Means Intuitively:*

The rank $r$ controls how many independent directions the update can move in. Think of it this way: if your model needs to learn that "patient" should map closer to "diagnosis" and "treatment" in medical contexts, that's a small number of directional changes in the embedding space — not a complete overhaul of every weight.

The theoretical foundation comes from Aghajanyan et al. (2020), #link("https://arxiv.org/abs/2012.13255")["Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"]. Their key finding was that fine-tuning updates are *inherently low-rank* — most of the meaningful change happens in a surprisingly small subspace of the full parameter space. This means we're not losing much by restricting the update to a low-rank form; we're simply discarding directions that wouldn't have changed meaningfully anyway.

*The Scaling Factor:*

In practice, the update is scaled:

#figure($ Delta W = frac(alpha, r) times B A $, caption: [LoRA scaling factor applied to weight update], kind: math.equation)

The ratio $alpha / r$ acts as a learning rate modifier for the LoRA update. When you change $r$, the scaling ensures the magnitude of the update remains roughly consistent, so you don't need to re-tune the learning rate every time you change the rank.

*How $r$ and $alpha$ Affect Training:*

- *Higher $r$* = more expressive updates (the model can learn more complex adaptations), but requires more VRAM and more trainable parameters. Typical values: $r i n \{8, 16, 32, 64\}$.
- *Higher $alpha$* = larger magnitude updates. If $alpha$ is too high, training becomes unstable; too low, and the model barely changes. Typical values: $alpha i n \{16, 32\}$, often set to $alpha = 2 r$.
- A common starting point: $r = 16$, $alpha = 32$ (so $alpha / r = 2$).

#pagebreak()

*Connection to Full Fine-Tuning Efficiency:*

This mathematical decomposition is precisely why you can fine-tune a 7B parameter model by training only ~0.5-2% of parameters. For example, applying LoRA with $r = 16$ to all linear layers in LLaMA 2 7B yields roughly 40M trainable parameters out of 7B total — enough to specialize the model for a new domain or task while keeping VRAM requirements low enough to train on a single consumer GPU.

==== 8.3.1 LoRA Target Module Selection Guide

Throughout this document, different sections use different `target_modules` configurations — `["q_proj", "v_proj"]` in basic examples versus `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` in Unsloth. This choice significantly affects both training cost and model quality.

*What the modules are:*

In a standard transformer block, the linear layers fall into two groups:

#figure(
  image("../diagrams/06-transformer-lora-targets.png", width: 85%),
  caption: [Transformer Block — LoRA Target Modules],
)

- *Attention projections:* `q_proj` (query), `k_proj` (key), `v_proj` (value), `o_proj` (output) — these control how the model attends to different parts of the input
- *FFN/Multi-Layer Perceptron (MLP) projections:* `gate_proj`, `up_proj`, `down_proj` — these control the feed-forward computation that transforms representations between attention layers

*Empirical guidance (from QLoRA paper, Dettmers et al. 2023):*

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Target Modules*], [*Trainable Params (7B, r=16)*], [*Quality*], [*When to Use*]),
    [`q_proj`, `v_proj` only], [~20M (~0.3%)], [Good baseline], [Quick experiments, very limited VRAM],
    [All attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`], [~40M (~0.6%)], [Better], [Standard choice when VRAM allows],
    [All linear layers (attention + FFN)], [~80-160M (~1-2%)], [Best], [Recommended default — the QLoRA paper found this performs closest to full fine-tuning],
  ),
  caption: [Empirical guidance (from QLoRA paper, Dettmers et al. 2023)],
  kind: table,
)

*Key findings from the QLoRA paper:*
- Adapting *all linear layers* consistently outperforms adapting attention layers only
- The performance gap widens on harder tasks (reasoning, complex instruction following)
- Adding more target modules is generally more effective than increasing rank on fewer modules — `r=8` on all layers often beats `r=64` on attention-only

*Practical recommendation:* Start with all linear layers at `r=16`. If VRAM is too tight, reduce rank to `r=8` before removing target modules. Only fall back to `q_proj`/`v_proj`-only as a last resort on severely memory-constrained hardware.

#figure(
```python
# Recommended default: all linear layers
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Memory-constrained fallback: attention only
lora_config_lite = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Minimum viable LoRA
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```
, caption: [LoRA target module configurations: full linear layers vs attention-only fallback],
)

=== 8.4 Critical LoRA Adapter Stacking Problem

#blockquote[
  *Important practical lesson:* You *cannot* stack LoRA adapters on top of each other. Before adding a new LoRA adapter (e.g., for DPO training on an instruction-tuned LoRA model), you must first *merge and unload* the previous adapter into the base model weights.
]

*Why:* Stacking LoRA layers causes unstable loss during training. The PEFT library provides two key methods:

#figure(
  image("../diagrams/07-lora-stacking.png", width: 85%),
  caption: [LoRA Adapter Stacking: Wrong vs Right],
)

#figure(
```python
from peft import PeftModel

# Load the instruction-tuned LoRA model
model = PeftModel.from_pretrained(base_model, instruction_checkpoint)

# CRITICAL: Merge LoRA weights into base model, then unload the adapter
model = model.merge_and_unload()

# Now you can apply a NEW LoRA config for DPO training
model = get_peft_model(model, new_lora_config)
```
, caption: [Merge-and-unload LoRA adapter before applying new adapter for DPO training],
)

*`merge_and_unload()`* - Merges the LoRA adapter weights into the base model's weights and removes the adapter layer entirely, returning a standard model that can accept a new LoRA adapter.

=== 8.5 DPO Training with TRL

TRL's `DPOTrainer` handles the preference optimization loop. It requires the fine-tuned model, a frozen reference model (used to compute the Kullback-Leibler (KL) divergence penalty), and a dataset containing prompt/chosen/rejected triples.

#figure(
```python
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,        # Frozen reference model
    args=training_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)
trainer.train()
```
, caption: [DPOTrainer setup with frozen reference model for preference alignment training],
)

#pagebreak(weak: true)

=== 8.6 Worked Example: Safe Medical Response Alignment

*Scenario:* Your instruction-tuned pharma model gives correct but sometimes unsafe responses - e.g., recommending dosages without mentioning contraindications, or not advising "consult your doctor." DPO aligns the model to prefer safe, complete responses.

*Training data (DPO format):*

#figure(
```json
[
  {
    "prompt": "What dosage of metformin should I take for type 2 diabetes?",
    "chosen": "Metformin dosing for type 2 diabetes is individualized by a healthcare provider based on your kidney function, other medications, and blood glucose levels. The typical starting dose is 500mg once or twice daily with meals, gradually titrated to a maximum of 2,000-2,550mg/day. However, I strongly recommend consulting your physician before starting or adjusting any medication, as they will consider your complete medical history and current lab results.",
    "rejected": "The standard dose of metformin is 500mg twice daily. You can increase it to 2000mg per day if needed. Take it with food to reduce stomach upset."
  },
  {
    "prompt": "Can I take ibuprofen with my blood pressure medication?",
    "chosen": "NSAIDs like ibuprofen can interact with several blood pressure medications, potentially reducing their effectiveness and increasing cardiovascular risk. Specifically:\n- With ACE inhibitors/ARBs: can reduce antihypertensive effect and increase risk of kidney injury\n- With diuretics: can cause fluid retention, counteracting the diuretic effect\n- With beta-blockers: may slightly increase blood pressure\n\nPlease consult your doctor or pharmacist before combining these medications. They may recommend acetaminophen as a safer alternative for pain management.",
    "rejected": "Yes, you can take ibuprofen with blood pressure medication. Just don't take too much. The usual dose is 200-400mg every 4-6 hours."
  },
  {
    "prompt": "Is it safe to stop taking atorvastatin if my cholesterol levels are normal now?",
    "chosen": "Do not stop taking atorvastatin without consulting your prescribing physician. While your cholesterol levels may be normal now, that is likely *because* of the medication. Abruptly discontinuing statins can cause:\n- Rebound increase in LDL cholesterol\n- Increased cardiovascular risk, especially in patients with a history of coronary events\n- Loss of pleiotropic benefits (anti-inflammatory, endothelial stabilization)\n\nYour doctor may consider adjusting the dose or discontinuation based on your overall cardiovascular risk profile, but this decision should always be made under medical supervision.",
    "rejected": "If your cholesterol is normal, you probably don't need the medication anymore. You can try stopping it and check your levels again in a few months."
  }
]
```
, caption: [DPO preference data: chosen safe responses vs rejected unsafe responses],
)

#pagebreak()

*Before DPO (instruction-tuned model):*

#figure(
```
Prompt: "What dosage of metformin should I take for type 2 diabetes?"
Output: "The standard dose of metformin is 500mg twice daily, which can be increased to
         a maximum of 2000mg per day. Take it with meals to minimize GI side effects."
         ← factually correct but gives direct dosing advice without safety caveats
```
, caption: [Instruction-tuned model before DPO: correct facts but no safety disclaimers],
)

*After DPO:*

#figure(
```
Prompt: "What dosage of metformin should I take for type 2 diabetes?"
Output: "Metformin dosing is individualized by a healthcare provider based on your kidney
         function, other medications, and blood glucose levels. The typical starting dose
         is 500mg once or twice daily with meals. However, I strongly recommend consulting
         your physician before starting or adjusting any medication."
         ← same knowledge, but now includes safety disclaimers and defers to physicians
```
, caption: [DPO-aligned model: adds safety caveats and defers to physician recommendation],
)

*What changed:* The model's knowledge is unchanged - the DPO step didn't teach new facts. Instead, it reshaped the model's _preferences_: it now favors responses that include safety warnings, recommend medical consultation, mention contraindications, and avoid giving direct medical advice without context.

*Inference after DPO — testing the preference-aligned model:*

#figure(
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the DPO-aligned model (after merge-and-unload)
model = AutoModelForCausalLM.from_pretrained("dpo-merged-checkpoint")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test with a safety-sensitive prompt
prompt = "What dosage of metformin should I take for type 2 diabetes?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.4,
    do_sample=True,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Expected: includes safety disclaimers, recommends consulting a physician,
# mentions individualized dosing — NOT a direct dosage recommendation
```
, caption: [DPO-aligned model inference: generating safe medical responses],
)

#blockquote[
  *Evaluating alignment quality:* Compare the DPO-aligned model's outputs against the pre-DPO instruction-tuned model on the same prompts. Look for: (1) safety caveats present, (2) physician consultation recommended, (3) no direct unsupervised medical advice. A/B testing with domain experts is the gold standard for alignment evaluation.
]

#pagebreak(weak: true)

=== 8.7 The GRPO Loss Function

==== Key Equation: GRPO Training Objective

#figure($ cal(L)_("GRPO")(theta) = -frac(1, G) sum_(i=1)^(G) [ min(frac(pi_theta(o_i|q), pi_("old")(o_i|q)) hat(A)_i,   "clip"(frac(pi_theta(o_i|q), pi_("old")(o_i|q)), 1-epsilon, 1+epsilon) hat(A)_i ) - beta   D_("KL")(pi_theta || pi_("ref")) ] $, caption: [GRPO objective: group-relative policy optimization], kind: math.equation)

*Name and Context:* This is the #link("https://arxiv.org/abs/2402.03300")[Group Relative Policy Optimization (GRPO)] training objective, introduced by DeepSeek (Shao et al., 2024). Unlike DPO which uses pairwise chosen/rejected comparisons, GRPO generates a _group_ of completions for each prompt and uses relative reward ranking within the group to compute advantages.

*Symbol Definitions:*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Symbol*], [*Meaning*], [*Notes*]),
    [$pi_theta$], [The policy model being trained], [Updated via gradient descent],
    [$pi_("old")$], [The policy model from the previous iteration], [Used for importance sampling ratio],
    [$pi_("ref")$], [The reference model], [Frozen baseline to prevent drift (same as in DPO)],
    [$o_i$], [The $i$-th generated output in the group], [One of $G$ completions for the same prompt],
    [$q$], [The input prompt], [The question or instruction],
    [$G$], [Group size], [Number of completions generated per prompt (typically 4-16)],
    [$hat(A)_i$], [Normalized advantage for output $i$], [How much better/worse this output is relative to the group],
    [$epsilon$], [Clipping parameter], [Prevents excessively large policy updates (typically 0.1-0.2)],
    [$beta$], [KL penalty coefficient], [Controls how far the model can drift from the reference (typically 0.04)],
    [$D_("KL")$], [KL divergence], [Regularization term preventing policy collapse],
  ),
  caption: [Symbol Definitions],
  kind: table,
)

*Plain English Explanation:*

GRPO works by generating multiple completions for the same prompt, scoring them with reward functions, and reinforcing the better ones. The process is:

1. For each prompt $q$, generate $G$ completions (the "group")
2. Score each completion using reward functions (e.g., correct answer format, mathematical accuracy)
3. Compute the *advantage* $hat(A)_i$ by normalizing rewards within the group: $hat(A)_i = frac(r_i - "mean"(r), "std"(r))$
4. Update the policy to increase the probability of high-advantage outputs and decrease low-advantage ones
5. Apply *clipping* (the min/clip operation) to prevent destructively large updates
6. Subtract a *KL penalty* to keep the model from drifting too far from the reference

#pagebreak(weak: true)

*Key Difference from DPO:*

DPO requires curated pairs of chosen/rejected responses. GRPO eliminates this requirement - it generates its own comparison group and learns from reward signals directly. This makes GRPO particularly effective for *reasoning tasks* (e.g., math, coding) where a verifier can automatically check correctness, enabling the model to discover its own reasoning strategies rather than imitating human-written chain-of-thought.

*Connection to Surrounding Content:*

GRPO is used by DeepSeek for training reasoning models like DeepSeek-R1. Unsloth provides optimized GRPO training with long-context support (Section 12.13) and a step-by-step tutorial for training your own reasoning model (Section 12.14).

=== 8.8 Additional Preference Alignment Techniques

#figure(
  table(
    columns: 4,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Technique*], [*Full Name*], [*How It Works*], [*Best For*]),
    [#link("https://arxiv.org/abs/2203.02155")[_RLHF (PPO)_]], [Reinforcement Learning from Human Feedback], [Train a reward model on human preferences, then use PPO to optimize the policy], [Maximum control over alignment, used by OpenAI for ChatGPT],
    [#link("https://arxiv.org/abs/2305.18290")[_DPO_]], [Direct Preference Optimization], [Directly optimize on chosen/rejected pairs without a reward model], [Simpler implementation, no reward model needed],
    [#link("https://arxiv.org/abs/2402.01306")[_KTO_]], [Kahneman-Tversky Optimization], [Uses prospect theory with *unpaired* binary feedback (thumbs up/down per response, no chosen/rejected pairs needed)], [Low-data scenarios where paired preference data is unavailable],
    [#link("https://arxiv.org/abs/2310.12036")[_IPO_]], [Identity Preference Optimization], [Regularized DPO that prevents overfitting to preference data by using identity mapping instead of log-sigmoid], [When preference dataset is noisy or small],
    [#link("https://arxiv.org/abs/2403.07691")[_ORPO_]], [Odds Ratio Preference Optimization], [Combines SFT and preference alignment in a single training stage], [Simplified pipeline, fewer training stages],
    [#link("https://arxiv.org/abs/2402.03300")[_GRPO_]], [Group Relative Policy Optimization], [Uses group-level comparisons instead of pairwise], [Reasoning RL tasks (used by DeepSeek); see Section 12 for Unsloth's GRPO support],
  ),
  caption: [Additional Preference Alignment Techniques],
  kind: table,
)

#pagebreak(weak: true)

=== 8.9 The RLHF Pipeline: Reward Modeling + PPO

While DPO has become popular for its simplicity, RLHF with PPO remains the dominant alignment technique at frontier labs (OpenAI, Anthropic, Google). Understanding the full RLHF pipeline is essential because: (1) it gives you maximum control over alignment behavior, (2) many production systems still use it, and (3) DPO's limitations (sensitivity to data quality, no iterative improvement) sometimes make RLHF the better choice.

*The three-stage RLHF pipeline:*

#figure(
  image("../diagrams/24-rlhf-three-stage-pipeline.png", width: 85%),
  caption: [RLHF three-stage pipeline: SFT model, reward model, PPO optimization],
)

==== Stage 2: Reward Model Training

The reward model learns to predict human preferences. Given a prompt $x$ and two responses $(y_w, y_l)$ where $y_w$ is preferred, the reward model is trained with the Bradley-Terry loss:

#figure($ cal(L)_("RM")(phi) = -bb(E)_((x, y_w, y_l)) [ log sigma ( r_phi(x, y_w) - r_phi(x, y_l) ) ] $, caption: [Bradley-Terry reward model loss], kind: math.equation)

_Symbol definitions:_

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Symbol*], [*Meaning*], [*Notes*]),
    [$r_phi(x, y)$], [Reward model output], [Scalar score for response $y$ given prompt $x$],
    [$y_w$], [Preferred (winning) response], [Ranked higher by human annotators],
    [$y_l$], [Dispreferred (losing) response], [Ranked lower by human annotators],
    [$sigma$], [Sigmoid function], [Converts score difference to probability],
  ),
  caption: [Stage 2: Reward Model Training],
  kind: table,
)

*Plain English:* The reward model learns to assign higher scores to responses that humans prefer. The loss pushes the score gap between preferred and dispreferred responses to be positive and large.

==== Stage 3: PPO Optimization

The policy model (initialized from the SFT model) is optimized to maximize the reward while staying close to the original SFT model via a KL penalty:

#figure($ cal(L)_("PPO")(theta) = bb(E)_(x tilde.op cal(D),  y tilde.op pi_theta(y|x)) [ r_phi(x, y) - beta dot.c D_("KL")(pi_theta(y|x) || pi_("SFT")(y|x)) ] $, caption: [PPO objective: reward maximization with KL penalty], kind: math.equation)

The PPO clipped surrogate objective prevents destructively large policy updates:

#figure($ cal(L)^("CLIP")(theta) = bb(E)_t [ min ( frac(pi_theta(a_t|s_t), pi_(theta_("old"))(a_t|s_t)) hat(A)_t,   "clip"(frac(pi_theta(a_t|s_t), pi_(theta_("old"))(a_t|s_t)), 1-epsilon, 1+epsilon ) hat(A)_t ) ] $, caption: [PPO clipped surrogate objective], kind: math.equation)

_Symbol definitions:_

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Symbol*], [*Meaning*], [*Notes*]),
    [$pi_theta$], [Policy model being trained], [Initialized from SFT model],
    [$pi_("SFT")$], [Original SFT model], [Frozen, used as KL anchor],
    [$r_phi(x, y)$], [Reward model score], [Learned scalar reward],
    [$beta$], [KL penalty coefficient], [Controls deviation from SFT model (typically 0.01–0.2)],
    [$hat(A)_t$], [Advantage estimate], [How much better an action is than expected (computed via Generalized Advantage Estimation (GAE))],
    [$epsilon$], [Clipping parameter], [Prevents large updates (typically 0.1–0.2)],
  ),
  caption: [Stage 3: PPO Optimization],
  kind: table,
)

*Plain English:* PPO generates responses from the current policy, scores them with the reward model, and updates the policy to produce higher-scoring responses — but the KL penalty prevents the model from drifting too far from the SFT baseline (which would cause "reward hacking" — exploiting the reward model's weaknesses rather than genuinely improving).

*Why RLHF is harder than DPO (but sometimes necessary):*

#figure(
  table(
    columns: 3,
    align: left,
    stroke: 0.4pt + luma(200),
    fill: (_, row) => if row == 0 { rgb("#1a5276").lighten(85%) } else if calc.odd(row) { rgb("#f5f8fc") } else { white },
    table.header([*Dimension*], [*DPO*], [*RLHF (PPO)*]),
    [Components needed], [Policy model + frozen reference], [Policy model + reward model + value model + reference model],
    [VRAM requirement], [~2x SFT (or ~1x with LoRA ref trick)], [~4x SFT (four models in memory)],
    [Hyperparameter sensitivity], [Moderate (beta, learning rate)], [High (KL coeff, clip range, value loss coeff, GAE lambda, multiple learning rates)],
    [Iterative improvement], [No — learns from fixed dataset], [Yes — can generate new data and re-score with reward model],
    [Reward hacking risk], [Low (no explicit reward model)], [Moderate — policy can exploit reward model weaknesses],
    [Best for], [Static preference datasets, simpler setups], [Iterative alignment, complex reward signals, maximum control],
  ),
  caption: [Why RLHF is harder than DPO (but sometimes necessary)],
  kind: table,
)

==== 8.9.1 How ChatGPT Was Aligned (InstructGPT Pipeline)

The ChatGPT alignment process (described in the #link("https://arxiv.org/abs/2203.02155")[InstructGPT paper], Ouyang et al. 2022) had three phases:

1. *Human Demonstration Phase:* Human labelers wrote ideal responses to prompts, creating SFT training data
2. *Preference Comparison Phase:* Labelers ranked model outputs from best to worst for ~40,000 prompts, creating comparison pairs for reward model training
3. *PPO Optimization:* The reward model was used to fine-tune the policy model via Proximal Policy Optimization

*Data sources for preferences:* ChatGPT user interaction logs, human labeler contractors (from Kenya, Philippines, US)

*Key insight from the course:* DPO eliminates the need for the separate reward model step, simplifying the entire pipeline from 3 stages to 2. However, RLHF remains preferred when you need iterative refinement or when your reward signal is complex (e.g., combining safety scores, helpfulness ratings, and factuality checks into a single reward).

=== 8.10 Section Summary

DPO provides a closed-form alternative to RLHF for preference alignment, eliminating the need for a separate reward model and PPO training loop. Despite sometimes being called "supervised" because it avoids explicit RL, DPO is a preference optimization method - it uses a contrastive loss over chosen/rejected pairs, not standard cross-entropy on labeled examples. GRPO extends this further by generating its own comparison groups and learning from reward signals directly. For maximum control, the full RLHF pipeline (reward modeling + PPO) remains the dominant approach at frontier labs. The critical practical lesson: always merge-and-unload previous LoRA adapters before applying new ones.

#line(length: 100%, stroke: 0.5pt + luma(200))
]
