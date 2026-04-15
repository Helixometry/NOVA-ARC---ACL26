"""
nova_arc/emotion_lens.py
------------------------
Hyperbolic Emotion Lens (HEL) — learnable radial calibration.

Motivation
----------
Non-verbal vocalizations (NVV) and verbal speech (UVS) differ in emotion
*intensity* even when the same emotion is expressed.  HEL corrects this
mismatch by applying a learnable power-law warp to the *radius* of each
hyperbolic embedding, leaving the *direction* unchanged:

    r̃ = r^α

where α is initialised to 1.0 (identity) and is jointly optimised during
training.  When α < 1 the warp compresses high-intensity embeddings toward
the origin; when α > 1 it expands them.

Procedure (per frame):
  1. log-map  b_t  to tangent space
  2. decompose into radius r and unit direction n̂
  3. r̃ = r^α  (power-law warp)
  4. exp-map  r̃ · n̂  back to Poincaré ball
"""

import torch
import torch.nn as nn

from nova_arc.hyperbolic import expmap0, logmap0, poincare_clip

EPS = 1e-8


class HyperbolicEmotionLens(nn.Module):
    """
    Hyperbolic Emotion Lens (HEL).

    Args:
        curvature (float):    Poincaré ball curvature c  (default 1.0)
        alpha_init (float):   initial value of the learnable exponent α  (default 1.0)
        stabilizer (float):   ε added to radius before the warp  (default 1e-8)
    """

    def __init__(
        self,
        curvature:   float = 1.0,
        alpha_init:  float = 1.0,
        stabilizer:  float = 1e-8,
    ):
        super().__init__()
        self.c   = curvature
        self.eps = stabilizer

        # α is a scalar learnable parameter
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """
        Apply radial calibration to hyperbolic frame embeddings.

        Args:
            b:  (B, T, d)  or  (B, d)  hyperbolic embeddings in Poincaré ball

        Returns:
            b̃:  same shape as b,  calibrated embeddings in Poincaré ball
        """
        # 1. Map to tangent space
        v = logmap0(b, self.c)                               # (..., d)

        # 2. Decompose into radius and unit direction
        r   = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)  # (..., 1)
        n   = v / r                                              # unit direction

        # 3. Learnable power-law warp on radius
        #    Use softplus(α) + eps so α stays positive
        alpha_pos = torch.nn.functional.softplus(self.alpha) + self.eps
        r_warped  = r.pow(alpha_pos)                          # r^α

        # 4. Reconstruct tangent vector and map back to ball
        v_warped = r_warped * n
        b_tilde  = expmap0(v_warped, self.c)

        return poincare_clip(b_tilde, self.c)

    def extra_repr(self) -> str:
        return f"c={self.c}, alpha={self.alpha.item():.4f}"
