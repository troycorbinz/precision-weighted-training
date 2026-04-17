"""
layer_gain.py — Per-layer gradient scaling based on representation divergence.

After backward(), scales each transformer block's parameter gradients by how
much that block's representations changed during the forward pass (divergence).
High-divergence layers get amplified gradients (they're actively revising
representations — genuine learning is happening). Low-divergence layers
get attenuated gradients (they're stable — don't disrupt what works).

Implements the "depth of surprise" concept from Predictive Coding:
shallow surprise (early layers) vs deep surprise (late layers) drive
different learning responses.

Composes with per-token gain functions:
  - Token gain (gain_functions.py): reshapes WHAT the model learns from
  - Layer gain (this file): reshapes WHERE in the model the learning happens

Model requirements:
  - Your model must store per-block divergences in a list attribute called
    ``model._layer_divergences`` during the forward pass (see README for how).
  - Your model's transformer blocks must be in an attribute called ``model.blocks``
    (an nn.ModuleList or similar), with parameters named ``blocks.0.``, ``blocks.1.``, etc.
  - If your model uses a different naming convention (e.g. ``model.layers`` or
    ``model.transformer.h``), update the ``scale_map`` prefix construction in
    ``scale_gradients()`` to match.

Usage:
    scaler = LayerGainScaler(config)
    # ... after loss.backward() and (optionally) grad_scaler.unscale_() ...
    scaler.scale_gradients(model)
    # ... then gradient clipping and optimizer.step() ...
"""

import torch


class LayerGainScaler:
    """Scale per-block gradients by forward-pass representation divergence.

    Divergence is measured during the forward pass as
    ``||x_out - x_in|| / ||x_in||`` for each transformer block and stored
    on the model as ``model._layer_divergences`` (a plain Python list of floats).
    You must add the divergence recording to your model's forward pass — see
    the README for a minimal code snippet.

    The scaling is mean-normalized: a layer with average divergence gets
    scale 1.0. Above-average divergence → scale > 1 (amplified). Below-average
    → scale < 1 (attenuated). This preserves total gradient magnitude while
    redirecting it to where the model is actively revising representations.

    Config (under ``training.layer_gain``):
        enabled (bool): Toggle layer gain scaling. Default False.
        strength (float): How aggressively to scale. 0 = no effect (all 1.0),
            1.0 = full linear scaling. Default 0.5.
        min_scale (float): Safety floor. Default 0.1.
        max_scale (float): Safety ceiling. Default 3.0.
        exclude_layers (list[int]): Layer indices to exclude from divergence
            normalization. Excluded layers still train normally with full
            gradients (scale=1.0) but don't participate in the mean
            calculation that determines other layers' scales. Default [0]
            (layer 0 is the embedding-to-representation bridge and is
            always a structural outlier).
    """

    def __init__(self, config: dict):
        cfg = config.get("training", {}).get("layer_gain", {})
        self.enabled = cfg.get("enabled", False)
        self.strength = cfg.get("strength", 0.5)
        self.min_scale = cfg.get("min_scale", 0.1)
        self.max_scale = cfg.get("max_scale", 3.0)
        self.exclude_layers = set(cfg.get("exclude_layers", [0]))
        self._last_stats: dict = {}

    def scale_gradients(self, model) -> None:
        """Scale each block's gradients by its divergence.

        Must be called after ``loss.backward()`` and ``precision.unscale()``,
        before gradient clipping and ``optimizer.step()``.
        """
        if not self.enabled:
            return

        divergences = getattr(model, '_layer_divergences', None)
        if not divergences or len(divergences) == 0:
            return

        divs = torch.tensor(divergences)
        n_layers = len(model.blocks)

        # Compute normalization from included layers only (exclude structural outliers)
        included_mask = torch.ones(n_layers, dtype=torch.bool)
        for idx in self.exclude_layers:
            if 0 <= idx < n_layers:
                included_mask[idx] = False

        included_divs = divs[included_mask]
        if len(included_divs) == 0 or included_divs.mean() < 1e-12:
            return
        mean_div = included_divs.mean()

        # Normalize included layers: center at 1.0
        # Excluded layers get scale = 1.0 (train normally, no modulation)
        scales = torch.ones(n_layers)
        normalized_included = included_divs / mean_div
        scales_included = (1.0 + self.strength * (normalized_included - 1.0)).clamp(
            self.min_scale, self.max_scale
        )
        scales[included_mask] = scales_included

        # Build param-name-prefix → scale mapping
        scale_map: dict[str, float] = {}

        for i in range(n_layers):
            scale_map[f"blocks.{i}."] = scales[i].item()

        # Optional: if your model groups blocks into larger modules (e.g. block
        # attention residual groups), you can scale those group-level parameters
        # by the mean scale of their constituent blocks. This section handles
        # models with a `block_attn_res` attribute; skip/adapt for your architecture.
        attn_res_group_size = getattr(model, 'attn_res_group_size', 0)
        if attn_res_group_size > 0 and hasattr(model, 'block_attn_res'):
            for g in range(len(model.block_attn_res)):
                start = g * attn_res_group_size
                end = min(start + attn_res_group_size, n_layers)
                group_scale = scales[start:end].mean().item()
                scale_map[f"block_attn_res.{g}."] = group_scale

        # Apply gradient scaling
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            for prefix, s in scale_map.items():
                if name.startswith(prefix):
                    param.grad.mul_(s)
                    break
            # Parameters not matching any prefix (embeddings, ln_f, lm_head)
            # keep their gradients unchanged (implicit scale = 1.0)

        # Record stats for W&B (mean/std from included layers only)
        self._last_stats = {
            "layer_gain/div_mean": mean_div.item(),
            "layer_gain/div_std": included_divs.std().item(),
            "layer_gain/div_min": included_divs.min().item(),
            "layer_gain/div_max": included_divs.max().item(),
            "layer_gain/scale_mean": scales.mean().item(),
            "layer_gain/scale_std": scales.std().item(),
            "layer_gain/scale_min": scales.min().item(),
            "layer_gain/scale_max": scales.max().item(),
        }

        # Per-layer detail: individual divergence and scale for each block.
        # Creates W&B line plots like layer_gain/div_layer_00, div_layer_01, ...
        # allowing visualization of divergence profiles and emergent grouping.
        for i in range(n_layers):
            self._last_stats[f"layer_gain/div_layer_{i:02d}"] = divergences[i]
            self._last_stats[f"layer_gain/scale_layer_{i:02d}"] = scales[i].item()

    def stats(self) -> dict:
        """Return cached stats from the last scale_gradients() call."""
        return self._last_stats
