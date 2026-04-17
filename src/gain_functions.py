"""
gain_functions.py — Per-token loss gain functions for precision-weighted training.

Gain functions re-weight per-token cross-entropy loss before reduction,
allowing the model to learn harder from surprising tokens and coast through
predictable ones.  Each function maps per-token loss → per-token gain weight.

All gain weights are **detached** — they scale loss magnitude but do not
contribute their own gradients through the gain computation. This means the
gain function reshapes the loss landscape without adding any learnable parameters
or changing the backward graph.

Usage:
    gain_fn = create_gain_function(config)
    # In your training loop, replace loss = F.cross_entropy(logits, targets):
    per_token_loss = F.cross_entropy(logits, targets, reduction='none')
    weights = gain_fn(per_token_loss, logits, targets)
    loss = (per_token_loss * weights).mean()

Config keys (nested under a "training" dict):
    training.gain_function: "none" | "linear" | "focal" | "sigmoid" | "precision"
    training.gain_config: dict of function-specific params (passed to the gain class)

See the paper (paper/precision-weighted-training.md) for full experimental results
comparing these variants.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class GainFunction(ABC):
    """Base class for per-token loss gain functions."""

    def __init__(self, config: dict):
        self.config = config
        self._last_stats: dict = {}

    @abstractmethod
    def __call__(
        self,
        per_token_loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token gain weights.

        Args:
            per_token_loss: (N,) per-token CE loss (detached is fine, but not required)
            logits: (N, V) raw logits — needed by focal loss for p_correct
            targets: (N,) target token ids

        Returns:
            (N,) gain weights, detached, same device as per_token_loss.
        """
        ...

    def stats(self) -> dict:
        """Return cached stats from the last __call__ for W&B logging."""
        return self._last_stats

    def _record_stats(self, gain: torch.Tensor, per_token_loss: torch.Tensor):
        """Compute and cache gain + loss distribution stats."""
        with torch.no_grad():
            # Filter out zero-loss tokens (padding)
            mask = per_token_loss > 0
            if mask.any():
                g = gain[mask]
                l = per_token_loss[mask]
                self._last_stats = {
                    "gain/mean": g.mean().item(),
                    "gain/std": g.std().item(),
                    "gain/max": g.max().item(),
                    "gain/min": g.min().item(),
                    "loss/per_token_mean": l.mean().item(),
                    "loss/per_token_std": l.std().item(),
                    "loss/per_token_p90": l.quantile(0.9).item(),
                }
            else:
                self._last_stats = {}


class UniformGain(GainFunction):
    """Baseline (paper variant A0): gain = 1.0 for all tokens.

    Equivalent to standard cross-entropy. Useful as a control that still
    logs per-token loss distribution stats for comparison with other variants.
    """

    def __call__(self, per_token_loss, logits, targets):
        gain = torch.ones_like(per_token_loss)
        self._record_stats(gain, per_token_loss)
        return gain


class LinearNormalizedGain(GainFunction):
    """Simple normalized gain (paper variant A1): gain = loss / mean(loss), clamped.

    Mean-normalized so total gradient magnitude is preserved.
    High-loss (surprising) tokens get gain > 1, low-loss (easy) tokens get gain < 1.
    This was the first successful variant in our experiments and motivated the
    theoretically grounded PrecisionWeightedGain below.

    Config:
        clamp_min (float): Minimum gain. Default 0.1
        clamp_max (float): Maximum gain. Default 5.0
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.clamp_min = config.get("clamp_min", 0.1)
        self.clamp_max = config.get("clamp_max", 5.0)

    def __call__(self, per_token_loss, logits, targets):
        with torch.no_grad():
            mask = per_token_loss > 0
            if mask.any():
                mean_loss = per_token_loss[mask].mean()
                gain = torch.where(
                    mask,
                    (per_token_loss / mean_loss.clamp(min=1e-8)).clamp(
                        self.clamp_min, self.clamp_max
                    ),
                    torch.ones_like(per_token_loss),
                )
            else:
                gain = torch.ones_like(per_token_loss)
            self._record_stats(gain, per_token_loss)
        return gain.detach()


class FocalGain(GainFunction):
    """Focal loss variant (paper variant A2): gain = (1 - p_correct)^gamma.

    Downweights easy tokens (model already confident), upweights hard tokens.
    Originally Lin et al. 2017 for object detection; adapted for LM training.

    WARNING: This variant is NOT mean-normalized. In our experiments it caused
    degenerate output (heavy repetition) because it systematically suppresses
    gradient on high-confidence tokens, which at early training means foundational
    language tokens like "the", "is", "a". Included for completeness and comparison.

    Config:
        gamma (float): Focusing parameter. Default 2.0
        clamp_min (float): Minimum gain. Default 1e-4
        clamp_max (float): Maximum gain. Default 10.0
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.gamma = config.get("gamma", 2.0)
        self.clamp_min = config.get("clamp_min", 1e-4)
        self.clamp_max = config.get("clamp_max", 10.0)

    def __call__(self, per_token_loss, logits, targets):
        with torch.no_grad():
            # Compute p_correct = softmax(logits)[target]
            probs = F.softmax(logits, dim=-1)  # (N, V)
            p_correct = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (N,)
            gain = ((1.0 - p_correct) ** self.gamma).clamp(
                self.clamp_min, self.clamp_max
            )
            self._record_stats(gain, per_token_loss)
        return gain.detach()


class SigmoidGain(GainFunction):
    """Sigmoid gain (paper variant A3): smooth S-curve centered on batch mean loss.

    Tokens with loss above the batch mean get gain > 1 (boosted).
    Tokens below get gain < 1 (suppressed).  The steepness parameter k
    controls how sharp the transition is.

    WARNING: This variant is NOT reliably mean-normalized. The gain/mean drifts
    during training (amplifying early, suppressive late), which destabilized
    our experiments. Included for completeness and comparison.

    Config:
        k (float): Steepness of the sigmoid. Default 5.0
        scale (float): Output range is [1-scale, 1+scale]. Default 0.5
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.k = config.get("k", 5.0)
        self.scale = config.get("scale", 0.5)

    def __call__(self, per_token_loss, logits, targets):
        with torch.no_grad():
            mask = per_token_loss > 0
            if mask.any():
                mean_loss = per_token_loss[mask].mean()
                # Sigmoid centered on mean, scaled to [1-scale, 1+scale]
                # When loss == mean: sigmoid(0) = 0.5 → gain = 1.0
                # When loss >> mean: sigmoid(+inf) = 1.0 → gain = 1 + scale
                # When loss << mean: sigmoid(-inf) = 0.0 → gain = 1 - scale
                raw = torch.sigmoid(self.k * (per_token_loss - mean_loss))
                gain = torch.where(
                    mask,
                    1.0 - self.scale + 2.0 * self.scale * raw,
                    torch.ones_like(per_token_loss),
                )
            else:
                gain = torch.ones_like(per_token_loss)
            self._record_stats(gain, per_token_loss)
        return gain.detach()


class PrecisionWeightedGain(GainFunction):
    """Predictive Coding-inspired: gain = 1 + precision * centered_error.

    Based on the precision-weighting framework from Predictive Coding
    (Rao & Ballard 1999, Millidge et al. 2022).  Precision is the inverse
    variance of the per-token loss within the batch — a principled measure
    of how *reliable* the error signal is.

    - High-variance batch (noisy, unreliable) → low precision → conservative gain
    - Low-variance batch (consistent signal) → high precision → stronger redistribution
    - Naturally mean-normalized (centered error sums to 0, so gain centers at 1.0)
    - Naturally bounded by batch statistics — no arbitrary clamp range needed

    In neuroscience, precision weighting is mediated by dopamine and linked to
    attention and salience — the brain amplifies reliable prediction errors
    and discounts noisy ones, which is exactly what this function does.

    Config:
        scale (float): Scales the precision term to control redistribution
            strength.  Default 1.0.  Lower values (0.5) are more conservative.
        epsilon (float): Prevents division by zero in variance.  Default 1e-6.
        clamp_min (float): Safety floor for gain.  Default 0.1.
        clamp_max (float): Safety ceiling for gain.  Default 5.0.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.scale = config.get("scale", 1.0)
        self.epsilon = config.get("epsilon", 1e-6)
        self.clamp_min = config.get("clamp_min", 0.1)
        self.clamp_max = config.get("clamp_max", 5.0)

    def __call__(self, per_token_loss, logits, targets):
        with torch.no_grad():
            mask = per_token_loss > 0
            if mask.any():
                valid_loss = per_token_loss[mask]
                mean_loss = valid_loss.mean()
                var_loss = valid_loss.var() + self.epsilon

                # Precision = inverse variance (how reliable is this batch's signal?)
                precision = 1.0 / var_loss

                # Centered error: how far each token deviates from batch mean
                error = per_token_loss - mean_loss

                # Gain = 1 + scaled precision-weighted deviation
                # Mean-normalized by construction: mean(error) = 0 → mean(gain) = 1
                gain = torch.where(
                    mask,
                    (1.0 + self.scale * precision * error).clamp(
                        self.clamp_min, self.clamp_max
                    ),
                    torch.ones_like(per_token_loss),
                )
            else:
                gain = torch.ones_like(per_token_loss)
            self._record_stats(gain, per_token_loss)
        return gain.detach()


# -- Registry and factory --

GAIN_REGISTRY: dict[str, type[GainFunction]] = {
    "none": UniformGain,
    "linear": LinearNormalizedGain,
    "focal": FocalGain,
    "sigmoid": SigmoidGain,
    "precision": PrecisionWeightedGain,
}


def create_gain_function(config: dict) -> GainFunction | None:
    """Create a gain function from model config.

    Reads ``training.gain_function`` from the config dict.

    - Key **absent** (production config): returns None — avoids the memory
      cost of ``reduction='none'`` on large models.
    - Key **present** (even ``"none"``): returns the corresponding gain
      function so that per-token loss stats are logged for comparison.
    """
    training_cfg = config.get("training", {})
    if "gain_function" not in training_cfg:
        return None
    name = training_cfg["gain_function"]
    gain_cfg = training_cfg.get("gain_config", {})
    cls = GAIN_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown gain function '{name}'. Available: {list(GAIN_REGISTRY.keys())}"
        )
    return cls(gain_cfg)
