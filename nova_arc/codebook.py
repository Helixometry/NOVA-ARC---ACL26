"""
nova_arc/codebook.py
--------------------
Hyperbolic Vector-Quantized (HVQ) Prosody Codebook.

Standard VQ codebook adapted to operate in the Poincaré ball:
  - Nearest-codeword assignment uses Poincaré distance (not Euclidean).
  - Straight-through gradient estimator (stop-gradient on codeword selection).
  - Codebook loss  +  commitment loss  (β controls the commitment weight).

Forward returns:
  quantized  : (B, T, d_hyp)  codeword embeddings for each frame
  vq_loss    : scalar  commitment + codebook loss
  indices    : (B, T)  int64  assigned codeword index for each frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nova_arc.hyperbolic import (
    poincare_dist_matrix,
    expmap0,
    logmap0,
    poincare_clip,
)


class HyperbolicVQCodebook(nn.Module):
    """
    Hyperbolic Vector-Quantized codebook.

    Args:
        codebook_size (int):  number of codewords  K   (default 256)
        dim (int):            embedding dimension   d   (same as hyperbolic latent dim)
        commitment_weight (float):  β in commitment loss  (default 0.25)
        vq_loss_weight (float):     weight on the full VQ loss  (default 1.0)
        curvature (float):          Poincaré ball curvature c   (default 1.0)
    """

    def __init__(
        self,
        codebook_size:     int   = 256,
        dim:               int   = 256,
        commitment_weight: float = 0.25,
        vq_loss_weight:    float = 1.0,
        curvature:         float = 1.0,
    ):
        super().__init__()
        self.K    = codebook_size
        self.dim  = dim
        self.beta = commitment_weight
        self.lam  = vq_loss_weight
        self.c    = curvature

        # Codewords are stored in Euclidean space and mapped to the ball on the fly.
        # Initialised near the origin for stability.
        self.codebook = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.codebook.weight, -0.1, 0.1)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _codewords_in_ball(self) -> torch.Tensor:
        """Return codeword embeddings projected into the Poincaré ball."""
        return expmap0(self.codebook.weight, self.c)  # (K, d)

    # ── forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Quantize hyperbolic frame embeddings.

        Args:
            x:  (B, T, d)  hyperbolic frame embeddings in Poincaré ball

        Returns:
            quantized:  (B, T, d)  nearest codewords (straight-through gradient)
            vq_loss:    scalar     codebook + commitment loss
            indices:    (B, T)     int64 codeword indices
        """
        B, T, d = x.shape

        # Flatten to (B*T, d) for distance computation
        x_flat = x.reshape(B * T, d)                      # (N, d)
        cw     = self._codewords_in_ball()                 # (K, d)

        # ── Assignment: nearest codeword by Poincaré distance ──
        dist_mat = poincare_dist_matrix(x_flat, cw, self.c)  # (N, K)
        indices  = dist_mat.argmin(dim=-1)                    # (N,)
        quantized_flat = cw[indices]                          # (N, d)

        # ── VQ losses (in tangent space for clean gradients) ──
        x_tang  = logmap0(x_flat,        self.c)   # (N, d)
        q_tang  = logmap0(quantized_flat, self.c)  # (N, d)

        # Codebook loss: move codewords toward encoder output (stop-grad on encoder)
        codebook_loss   = F.mse_loss(q_tang, x_tang.detach())
        # Commitment loss: move encoder output toward codewords (stop-grad on codeword)
        commitment_loss = F.mse_loss(x_tang, q_tang.detach())

        vq_loss = self.lam * (codebook_loss + self.beta * commitment_loss)

        # ── Straight-through estimator ──
        # Pass gradients to encoder as if quantized == x
        quantized_flat_st = x_flat + (quantized_flat - x_flat).detach()

        # Reshape back
        quantized = quantized_flat_st.reshape(B, T, d)
        indices   = indices.reshape(B, T)

        return quantized, vq_loss, indices
