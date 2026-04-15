"""
nova_arc/prototypes.py
----------------------
Hyperbolic Emotion Prototype Bank.

For each emotion class c, the prototype μ^(c) is the Fréchet mean of all
source-domain utterance embeddings in that class:

    μ^(c) = argmin_{z ∈ 𝔻} Σ_{i: y_i=c}  d_c²(z, b̃_i^S)

Prototypes are recomputed once per epoch from the running set of source
embeddings accumulated during that epoch, then frozen for the next epoch.
"""

import torch
from typing import Dict, Optional

from nova_arc.hyperbolic import frechet_mean, poincare_clip

EPS = 1e-8


class PrototypeBank:
    """
    Manages per-class hyperbolic emotion prototypes.

    Usage (inside the training loop):
        bank = PrototypeBank(num_classes=5, dim=128, curvature=1.0)

        # --- during each source batch ---
        bank.accumulate(embeddings, labels)   # collect source embeddings

        # --- at the end of each epoch ---
        bank.refresh()                        # recompute Fréchet means
        prototypes = bank.prototypes          # (C, d) tensor

        # --- use for OT cost matrix ---
        cost = poincare_dist_matrix(prototypes, target_embeddings)

    Args:
        num_classes (int):   number of emotion classes C
        dim (int):           embedding dimension (bottleneck dim d_b)
        curvature (float):   Poincaré ball curvature c
        frechet_iters (int): Riemannian GD iterations for Fréchet mean
    """

    def __init__(
        self,
        num_classes:   int,
        dim:           int,
        curvature:     float = 1.0,
        frechet_iters: int   = 20,
    ):
        self.C             = num_classes
        self.dim           = dim
        self.c             = curvature
        self.frechet_iters = frechet_iters

        # Current prototypes — initialised near origin
        self._prototypes: torch.Tensor = torch.zeros(num_classes, dim)

        # Buffers to accumulate source embeddings within an epoch
        self._buffer: Dict[int, list] = {k: [] for k in range(num_classes)}

    # ── accumulate ──────────────────────────────────────────────────────────

    def accumulate(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Collect source embeddings from the current batch.

        Args:
            embeddings:  (B, d)  hyperbolic utterance embeddings (u_hyp)
            labels:      (B,)    integer class labels 0 … C-1
        """
        embeddings = embeddings.detach().cpu()
        labels     = labels.cpu()
        for emb, lbl in zip(embeddings, labels):
            self._buffer[int(lbl)].append(emb)

    # ── refresh ──────────────────────────────────────────────────────────────

    def refresh(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Recompute all class prototypes from accumulated embeddings.
        Clears the buffer after recomputation.

        Args:
            device:  move the resulting prototype tensor to this device

        Returns:
            (C, d)  updated prototype tensor
        """
        protos = []
        for c_idx in range(self.C):
            pts = self._buffer[c_idx]
            if len(pts) == 0:
                # No samples seen for this class: keep previous prototype
                protos.append(self._prototypes[c_idx])
            elif len(pts) == 1:
                protos.append(poincare_clip(pts[0], self.c))
            else:
                stacked = torch.stack(pts, dim=0)          # (N, d)
                mu      = frechet_mean(
                    stacked,
                    c       = self.c,
                    n_iter  = self.frechet_iters,
                )
                protos.append(mu)

        self._prototypes = torch.stack(protos, dim=0)      # (C, d)

        # Clear buffers
        self._buffer = {k: [] for k in range(self.C)}

        if device is not None:
            self._prototypes = self._prototypes.to(device)

        return self._prototypes

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def prototypes(self) -> torch.Tensor:
        """(C, d) current prototype tensor."""
        return self._prototypes

    def to(self, device: torch.device) -> "PrototypeBank":
        self._prototypes = self._prototypes.to(device)
        return self

    def state_dict(self) -> dict:
        return {"prototypes": self._prototypes}

    def load_state_dict(self, state: dict) -> None:
        self._prototypes = state["prototypes"]
