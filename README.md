# Precision-Weighted Training for Language Models

**When Loss and Quality Diverge**

Standard language model training treats every token and every layer equally. This repository provides two composable training-time interventions, inspired by [Predictive Coding](https://en.wikipedia.org/wiki/Predictive_coding) theory from neuroscience, that reshape the learning signal to produce qualitatively better output — even when aggregate loss metrics show no difference.

**Read the full paper: [paper/precision-weighted-training.md](paper/precision-weighted-training.md)**

## What This Does

### 1. Per-token precision-weighted gain (`src/gain_functions.py`)

Instead of treating every token's loss equally, the gain function scales each token's contribution based on how surprising it is relative to the batch:

```
gain_i = clamp(1 + scale * (loss_i - mean_loss) / variance, min, max)
```

- Tokens the model is uncertain about (high loss relative to batch mean) get **amplified** gradient
- Tokens it already predicts well get **attenuated** gradient
- The `1/variance` term is the **precision** — it self-regulates: noisy batches (high variance) get conservative, near-uniform gain; consistent batches (low variance) get stronger redistribution
- **Mean-normalized by construction**: since `mean(loss_i - mean_loss) = 0`, the average gain is exactly 1.0. Total gradient magnitude is preserved — gain redistributes the learning signal, it does not amplify or suppress it. This property is critical; our experiments showed that gain functions where the mean drifts away from 1.0 cause training degeneration

### 2. Per-layer divergence gradient scaling (`src/layer_gain.py`)

After each forward pass, each transformer block's "divergence" is measured — how much it changed the representation: `||output - input|| / ||input||`. After backward, gradients are scaled by this divergence:

- **High-divergence layers** (actively revising representations) get amplified gradients
- **Low-divergence layers** (stable, not changing much) get attenuated gradients
- Like token gain, layer gain is **mean-normalized**: a layer with average divergence gets scale 1.0

Together: token gain reshapes **what** the model learns from; layer gain reshapes **where** in the network the learning lands.

## Key Results

In a controlled 1.2B-parameter comparison (3.9B tokens, both models trained on identical data in identical order):

| Metric | Baseline | Gain-trained |
|---|---|---|
| Smoothed val loss | 3.946 | 3.950 (**indistinguishable**, diff 0.004) |
| Blind A/B preference | 40.1% of decisive | **59.9% of decisive** (p = 2.80 × 10⁻⁸) |
| Compute overhead | — | **none** (identical wall-clock throughput) |

Across 1,181 judgments from a 42-judge blind panel (29 humans — the author plus 28 volunteers — and 13 foundation-model judges spanning eleven vendors), the gain-trained model is preferred by humans (60.5% decisive) and foundation models (59.0% decisive) within 1.5 points of each other. The direction survives every sensitivity filter we apply (FMs only, humans only, exclude human speed-clickers, exclude tie-biased judges, exclude partial completions, exclude all of the above simultaneously); the strictest filter leaves 27 engaged judges and 864 judgments with **63.1% decisive gain preference at p = 5.3 × 10⁻¹¹**. The preference is strongest on open-ended tasks (creative 71.3%, world knowledge 74.5%, instruction following 68.3%, conversational 64.1%) and narrowest on factual recall, the one category with a small baseline lean (46.2% decisive) — consistent with the mechanism's theoretical prediction that precision weighting favors generalization over rote memorization.

**Full methodology, per-category breakdowns, and discussion in the [paper](paper/precision-weighted-training.md).**

## Repository Structure

```
src/
  gain_functions.py         # Per-token gain functions (precision, linear, focal, sigmoid, uniform)
  layer_gain.py             # Per-layer divergence gradient scaler
eval/
  ab_compare.py             # Blind A/B comparison Flask webapp (see note below)
configs/
  eval_questions_1.2B.json  # 32 evaluation prompts across 7 categories
paper/
  precision-weighted-training.md  # Full paper with all experimental results
```

## Quick Start

### Installing

```bash
git clone https://github.com/troycorbinz/precision-weighted-training.git
cd precision-weighted-training
pip install torch  # PyTorch 2.0+ required
```

### Per-token precision-weighted gain (standalone)

Portable to most PyTorch transformer training loops with minimal changes. The per-token gain function requires only switching to `reduction='none'` and one multiply; the layer-gain scaler requires small architecture-specific edits (see porting checklist below).

**Per-token gain** — the only change is using `reduction='none'` to get per-token losses, then multiplying by the gain before taking the mean:

```python
import torch.nn.functional as F
from src.gain_functions import create_gain_function

# Configure — "precision" is the recommended variant (see paper Section 3.1)
config = {
    "training": {
        "gain_function": "precision",
        "gain_config": {"scale": 1.0}
    }
}
gain_fn = create_gain_function(config)

# In your training step — replace:
#   loss = F.cross_entropy(logits, targets)
# with:
per_token_loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),  # (batch*seq, vocab)
    targets.view(-1),                   # (batch*seq,)
    reduction='none'                    # key change: get per-token losses
)
gain = gain_fn(per_token_loss, logits.view(-1, logits.size(-1)), targets.view(-1))
loss = (per_token_loss * gain).mean()

loss.backward()
optimizer.step()
```

The gain function is **detached** (no additional gradients flow through it) and works with any optimizer (Adam, AdamW, SGD, Muon, etc.).

### Per-layer divergence gradient scaling (standalone)

This requires two small additions to your code:

**Step 1: Record divergence during the forward pass.**

In your transformer model's forward method, initialize the divergence list before running the blocks, and record each block's divergence:

```python
class YourTransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
        # ...

    def forward(self, x):
        # Reset divergence tracking at the start of each forward pass
        self._layer_divergences = []

        for block in self.blocks:
            x_in = x
            x = block(x)

            # Record how much this block changed the representation.
            # .item() moves the scalar to CPU; detach + no_grad ensure
            # this measurement doesn't affect the backward graph.
            if self.training:
                with torch.no_grad():
                    div = (x - x_in).norm() / (x_in.norm() + 1e-8)
                    self._layer_divergences.append(div.item())

        return x
```

**Step 2: Scale gradients after backward, before the optimizer step.**

```python
from src.layer_gain import LayerGainScaler

config = {
    "training": {
        "layer_gain": {
            "enabled": True,
            "strength": 0.5,       # 0 = no effect, 1.0 = full scaling
            "min_scale": 0.1,      # safety floor (no layer gets <10% gradient)
            "max_scale": 3.0,      # safety ceiling
            "exclude_layers": [0]  # layer 0 is always a structural outlier (see paper Section 3.2)
        }
    }
}
layer_scaler = LayerGainScaler(config)

# In your training step:
loss.backward()
layer_scaler.scale_gradients(model)  # scale gradients in-place
optimizer.step()
```

**Important:** The scaler looks up parameters by name prefix `blocks.{i}.` to match them to layer indices. If your model uses a different attribute name (e.g. `model.layers`, `model.transformer.h`), you'll need to update the prefix in `scale_gradients()`. Search for `scale_map[f"blocks.{i}."]` in `layer_gain.py`.

### Using both together (recommended)

The two mechanisms are independent and compose naturally. Token gain happens before backward; layer gain happens after. Here is a complete training step:

```python
import torch
import torch.nn.functional as F
from src.gain_functions import create_gain_function
from src.layer_gain import LayerGainScaler

# --- Setup (once) ---
config = {
    "training": {
        "gain_function": "precision",
        "gain_config": {"scale": 1.0},
        "layer_gain": {
            "enabled": True,
            "strength": 0.5,
            "min_scale": 0.1,
            "max_scale": 3.0,
            "exclude_layers": [0]
        }
    }
}
gain_fn = create_gain_function(config)
layer_scaler = LayerGainScaler(config)

# --- Training step ---
optimizer.zero_grad()

# Forward pass (model records _layer_divergences internally — see Step 1 above)
logits = model(input_ids)

# Per-token gain-weighted loss
logits_flat = logits.view(-1, logits.size(-1))
targets_flat = targets.view(-1)
per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
gain = gain_fn(per_token_loss, logits_flat, targets_flat)
loss = (per_token_loss * gain).mean()

# Backward
loss.backward()

# Layer-gain gradient scaling (after backward, before optimizer)
layer_scaler.scale_gradients(model)

# Optional: gradient clipping, then step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Optional: log stats to W&B or similar
# gain_fn.stats()        -> {"gain/mean": ..., "gain/std": ..., ...}
# layer_scaler.stats()   -> {"layer_gain/div_mean": ..., "layer_gain/scale_mean": ..., ...}
```

### Config reference

```json
{
  "training": {
    "gain_function": "precision",
    "gain_config": {
      "scale": 1.0,
      "clamp_min": 0.1,
      "clamp_max": 5.0
    },
    "layer_gain": {
      "enabled": true,
      "strength": 0.5,
      "min_scale": 0.1,
      "max_scale": 3.0,
      "exclude_layers": [0]
    }
  }
}
```

**Available gain functions:**
| Name | Description | Mean-normalized? | Recommended? |
|---|---|---|---|
| `"precision"` | PC-inspired: `1 + (1/var) * (loss - mean)` | Yes (by construction) | **Yes** |
| `"linear"` | `loss / mean(loss)`, clamped | Yes (by construction) | Viable alternative |
| `"focal"` | `(1 - p_correct)^gamma` (Lin et al. 2017) | No | No (causes degeneration) |
| `"sigmoid"` | S-curve centered on batch mean | No (drifts) | No (unstable) |
| `"none"` | Uniform gain = 1.0 (standard CE) | N/A | Baseline control |

## Porting Checklist

If you're integrating into an existing codebase, here is the complete list of changes:

**Per-token gain (minimal, architecture-independent):**
1. Switch your CE loss to `reduction='none'` to get per-token losses
2. Mask out padding / ignored tokens (the gain function handles this via `loss > 0`)
3. Multiply per-token loss by the detached gain weights before reducing to a scalar
4. Call `.mean()` on the result — gain is mean-normalized, so total gradient magnitude is preserved

**Per-layer divergence scaling (requires architecture-specific edits):**

5. Initialize `model._layer_divergences = []` at the start of each forward pass
6. After each transformer block, record `(x_out - x_in).norm() / (x_in.norm() + 1e-8)` under `torch.no_grad()`
7. Call `layer_scaler.scale_gradients(model)` after `loss.backward()` and after AMP unscale (if using mixed precision), but before gradient clipping and `optimizer.step()`
8. Update the layer-name prefix in `layer_gain.py` if your blocks are not named `blocks.{i}.` (search for `scale_map[f"blocks.{i}."]`)

Steps 1–4 are a ~5-line change in your loss function. Steps 5–8 require touching your model's forward method and training loop.

## A/B Evaluation Webapp

`eval/ab_compare.py` is the blind comparison webapp used in the paper. The **batch-serving mode** works standalone with Flask:

```bash
pip install flask
python eval/ab_compare.py --batch <path_to_batch_eval.json>
# Open http://localhost:8400
```

The interactive and batch-generation modes have model-loading imports specific to CLLM and won't run outside that environment. They are included as reference for the evaluation methodology.

### Judge flow

1. `/` — judge enters a display name (or pseudonym), then completes an optional 5-question demographic survey (LLM usage frequency, background, primary language, age band, prior participation in language-model studies). All questions are optional; each can be left blank.
2. The webapp assigns a deterministic per-judge RNG (seeded from `judge_id`) that draws the `(a_gen_idx, b_gen_idx, left_is_a)` pairing for each question. This avoids the modular-collision failure where linear `(q*k + jid) % n_gens` schemes produce identical responses for judges whose IDs differ by a multiple of `n_gens`.
3. The judge is shown 32 blind pairs side-by-side and picks Left / Right / Tie.
4. Results are stored in `ab_results.json` and demographics in `ab_demographics.json` (local JSON files, not a database).

### Admin report (`/_report`)

A dashboard auto-refreshing every 30 seconds, covering:

- **Aggregate** — total judgments, total judges, decisive (A+B) count, A/B/tie counts, B% of decisive, two-sided binomial p-value.
- **Sensitivity** — the headline B% re-computed under multiple filters:
  - **FMs only** and **Humans only** — splits the panel to check whether the signal lives in both populations or is carried by one. Directly addresses the "foundation-model judges share training priors" critique.
  - **Exclude human speed-clickers** (median vote interval <15s; FMs exempt as fast-by-nature) — removes inattentive humans.
  - **Exclude tie-biased judges** (>80% ties) — removes judges who couldn't or wouldn't discriminate.
  - **Exclude partial completions** (n<32) — removes judges who didn't complete, whose prompt coverage is skewed toward early questions.
  - **Exclude all of the above** — the strictest filter.
- **Per-judge** — one row per judge: type classification (FM/Human based on a name-marker tuple), vote counts, tie rate, B% of decisive, median seconds between votes, first/last vote timestamps. Orange highlights flag tie rate >80% or median speed <15s.
- **Demographics** — counts for each of the 5 survey fields.
- **Per-question coverage** — 32 rows (one per prompt) showing total votes, A/B/tie breakdown, B% of decisive, and a prompt snippet. Any question below the max vote count is highlighted to surface coverage gaps.
- **Per-answer coverage** — `n_questions × n_models × n_generations` rows showing how often each specific generation was drawn into a pairing and how often it won or tied. Lets you verify that the per-judge RNG is producing adequate spread across generations and surface any generation that was never shown.
- **Recent activity** — last 15 votes with judge, question #, and result.

### Interpreting the sensitivity rows

If the headline direction survives all filters with comparable effect magnitude, the result is not explained by low-quality judges, partial coverage, or population-specific bias. A headline that matches the signed direction of the strictest filter is a stronger claim than either row alone.

## Requirements

- Python 3.10+
- PyTorch 2.0+

No other dependencies for the core method (`src/`). The A/B evaluation webapp additionally requires Flask.

## Citation

If you use this work, please cite:

```
Corbin, T. (2026). Precision-Weighted Training for Language Models: When Loss
and Quality Diverge. Working paper. https://github.com/troycorbinz/precision-weighted-training
```

## License

MIT License. See [LICENSE](LICENSE) for details.
