"""
nova_arc/pooling.py
-------------------
Hyperbolic Attention Pooling.

Aggregates a sequence of calibrated hyperbolic frame embeddings {b̃_t}
into a single utterance-level representation u♭.

Procedure:
  1. Log-map all calibrated frames to the tangent space
  2. Compute a scalar attention weight for each frame using a learnable
     vector w:  a_t = exp(w^T b̃_t^tang) / Σ_t' exp(w^T b̃_{t'}^tang)
  3. Take the weighted sum in the tangent space  →  u♭
  4. Also keep the hyperbolic counterpart  b̃ = exp_0(u♭)  for prototype
     matching and optimal transport.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nova_arc.hyperbolic import logmap0, expmap0, poincare_clip

EPS = 1e-8


class HyperbolicAttentionPooling(nn.Module):
    """
    Attention pooling over a sequence of hyperbolic frame embeddings.

    Args:
        dim (int):        embedding dimension (bottleneck dim d_b)
        curvature (float): Poincaré ball curvature c  (default 1.0)
    """

    def __init__(self, dim: int, curvature: float = 1.0):
        super().__init__()
        self.c = curvature
        # Learnable attention query vector w  (d_b,)
        self.w = nn.Parameter(torch.randn(dim))
        nn.init.normal_(self.w, std=0.01)

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, b_tilde: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            b_tilde:  (B, T, d)  calibrated hyperbolic frame embeddings
            mask:     (B, T)     bool mask — True for valid frames, False for padding
                                 (optional, only needed for variable-length batches)

        Returns:
            u_tang:  (B, d)  utterance representation in tangent space  (u♭)
            u_hyp:   (B, d)  hyperbolic counterpart  b̃ = exp_0(u♭)
        """
        # 1. Map to tangent space
        v = logmap0(b_tilde, self.c)    # (B, T, d)

        # 2. Compute attention scores  a_t = w · v_t
        scores = torch.matmul(v, self.w)   # (B, T)

        # Mask padding frames with -inf before softmax
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)   # (B, T)

        # Handle all-padding edge case (all -inf → nan after softmax)
        attn = torch.nan_to_num(attn, nan=0.0)

        # 3. Weighted sum in tangent space
        u_tang = (attn.unsqueeze(-1) * v).sum(dim=1)   # (B, d)

        # 4. Hyperbolic counterpart
        u_hyp = expmap0(u_tang, self.c)                # (B, d)
        u_hyp = poincare_clip(u_hyp, self.c)

        return u_tang, u_hyp

    def extra_repr(self) -> str:
        return f"dim={self.w.shape[0]}, c={self.c}"
