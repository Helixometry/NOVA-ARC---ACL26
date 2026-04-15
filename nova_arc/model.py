"""
nova_arc/model.py
-----------------
NOVA-ARC — fully modular, component-switchable model.

Design philosophy
-----------------
Every layer is optional.  The user controls the architecture entirely through
NOVAARCConfig flags.  Components can be mixed freely:

    Geometry:
        use_hyperbolic=True   →  full Poincaré-ball pipeline
        use_hyperbolic=False  →  pure Euclidean pipeline (all hyperbolic ops removed)

    Components (each can be turned on/off independently):
        use_codebook  →  HVQ prosody codebook
        use_hel       →  Hyperbolic Emotion Lens (radial calibration)
        use_ot        →  Optimal-transport domain adaptation

    Pooling (choose one):
        pooling="attention"  →  learnable attention pooling  (default)
        pooling="mean"       →  simple mean pooling
        pooling="max"        →  max pooling

Examples
--------
Full NOVA-ARC (paper setting)::

    config = NOVAARCConfig(num_classes=5)
    model  = NOVAARC(encoder=enc, config=config)

Euclidean baseline (no hyperbolic, no OT)::

    config = NOVAARCConfig(num_classes=5,
                           use_hyperbolic=False,
                           use_codebook=False,
                           use_hel=False,
                           use_ot=False)

Source-only hyperbolic (no OT adaptation)::

    config = NOVAARCConfig(num_classes=5, use_ot=False)

Ablate HEL only::

    config = NOVAARCConfig(num_classes=5, use_hel=False)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nova_arc.hyperbolic import expmap0, logmap0, mobius_add, poincare_clip
from nova_arc.encoders   import BaseEncoder
from nova_arc.codebook   import HyperbolicVQCodebook
from nova_arc.emotion_lens import HyperbolicEmotionLens
from nova_arc.pooling    import HyperbolicAttentionPooling
from nova_arc.prototypes import PrototypeBank
from nova_arc.transport  import compute_ot
from nova_arc.losses     import nova_arc_loss


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class NOVAARCConfig:
    """
    Complete configuration for NOVA-ARC.

    Required
    --------
    num_classes (int):
        Number of emotion / target classes in YOUR dataset.
        The model builds its classifier head dynamically from this value.

    Architecture toggles
    --------------------
    use_hyperbolic (bool):
        True  → all embeddings live in the Poincaré ball (default, paper setting).
        False → pure Euclidean pipeline; all expmap/logmap ops are identity.

    use_codebook (bool):
        True  → include HVQ prosody codebook (discrete cues).
        False → skip; continuous embedding used directly.

    use_hel (bool):
        True  → apply Hyperbolic Emotion Lens (radial calibration).
        False → skip; useful for ablation.

    use_ot (bool):
        True  → include Sinkhorn OT domain adaptation during training.
        False → source-only training (standard supervised).

    pooling (str):
        "attention"  → learnable attention pooling over frames (default).
        "mean"       → simple mean over time.
        "max"        → max over time.

    Dimensions
    ----------
    hyperbolic_dim (int):
        Output embedding dimension d_hyp.  The encoder output is projected to
        this dimension before any hyperbolic operations.

    curvature (float):
        Poincaré ball curvature c.  Use 1.0 (paper setting).
        Only relevant when use_hyperbolic=True.

    Codebook
    --------
    codebook_size (int):      number of codewords K  (default 256)
    commitment_weight (float): β for VQ commitment loss  (default 0.25)
    vq_loss_weight (float):   scale on total VQ loss  (default 1.0)

    Prototype bank
    --------------
    frechet_iters (int):
        Riemannian GD steps for Fréchet mean.
        Only used when use_hyperbolic=True and use_ot=True.

    Optimal transport
    -----------------
    sinkhorn_eps (float):  regularisation ε  (default 0.05)
    sinkhorn_iters (int):  Sinkhorn iterations  (default 50)

    Loss weights
    ------------
    lambda_opt (float):  weight on OT cost  L_OPT  (default 0.1)
    lambda_ce (float):   weight on OT soft-CE  L_OT-CE  (default 0.1)

    Training
    --------
    lr (float):     Adam learning rate  (default 1e-4)
    epochs (int):   training epochs  (default 30)
    device (str):   "cuda" / "cpu" / "mps"
    """

    # ── Required ─────────────────────────────────────────────────────────────
    num_classes: int   # no default — must be set by user

    # ── Architecture toggles ─────────────────────────────────────────────────
    use_hyperbolic: bool = True
    use_codebook:   bool = True
    use_hel:        bool = True
    use_ot:         bool = True
    pooling:        str  = "attention"   # "attention" | "mean" | "max"

    # ── Dimensions ───────────────────────────────────────────────────────────
    hyperbolic_dim: int   = 256
    curvature:      float = 1.0

    # ── Codebook ─────────────────────────────────────────────────────────────
    codebook_size:      int   = 256
    commitment_weight:  float = 0.25
    vq_loss_weight:     float = 1.0

    # ── Prototype bank ───────────────────────────────────────────────────────
    frechet_iters: int = 20

    # ── OT ───────────────────────────────────────────────────────────────────
    sinkhorn_eps:   float = 0.05
    sinkhorn_iters: int   = 50

    # ── Loss weights ─────────────────────────────────────────────────────────
    lambda_opt: float = 0.1
    lambda_ce:  float = 0.1

    # ── Training ─────────────────────────────────────────────────────────────
    lr:     float = 1e-4
    epochs: int   = 30
    device: str   = "cpu"

    def __post_init__(self):
        if self.pooling not in ("attention", "mean", "max"):
            raise ValueError(
                f"pooling must be 'attention', 'mean', or 'max', got '{self.pooling}'"
            )
        if not self.use_hyperbolic and (self.use_codebook or self.use_hel):
            # Codebook and HEL require hyperbolic geometry — auto-disable them
            self.use_codebook = False
            self.use_hel      = False


# ── Internal pooling helpers ─────────────────────────────────────────────────

class _MeanPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask=None):
        # x: (B, T, d)
        if mask is not None:
            # mask: True for valid frames
            m = mask.unsqueeze(-1).float()
            return (x * m).sum(1) / m.sum(1).clamp(min=1e-8)
        return x.mean(dim=1)


class _MaxPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        return x.max(dim=1).values


# ── Main model ───────────────────────────────────────────────────────────────

class NOVAARC(nn.Module):
    """
    NOVA-ARC model.

    All components are built dynamically from NOVAARCConfig, so the
    architecture adapts completely to the user's settings.

    Args:
        encoder (BaseEncoder):
            Any encoder that inherits BaseEncoder and returns (B, T, d_enc).
            Built-in: WavLMEncoder, Wav2Vec2Encoder, Voc2VecEncoder, MMSEncoder.
            Custom: subclass BaseEncoder and implement forward().

        config (NOVAARCConfig):
            Full hyperparameter + component config.

    Examples::

        # Full NOVA-ARC (paper setting)
        from nova_arc import NOVAARC, NOVAARCConfig
        from nova_arc.encoders import WavLMEncoder

        model = NOVAARC(
            encoder = WavLMEncoder(freeze=True),
            config  = NOVAARCConfig(num_classes=5),
        )

        # Euclidean baseline — no hyperbolic, no OT
        model = NOVAARC(
            encoder = WavLMEncoder(freeze=True),
            config  = NOVAARCConfig(
                num_classes    = 5,
                use_hyperbolic = False,
                use_ot         = False,
            ),
        )

        # Custom encoder
        class MyEncoder(BaseEncoder):
            def __init__(self):
                super().__init__(output_dim=512)
                self.backbone = ...
            def forward(self, waveforms):
                return self.backbone(waveforms)   # (B, T, 512)

        model = NOVAARC(
            encoder = MyEncoder(),
            config  = NOVAARCConfig(num_classes=8, hyperbolic_dim=512),
        )
    """

    def __init__(self, encoder: BaseEncoder, config: NOVAARCConfig):
        super().__init__()

        self.config  = config
        self.encoder = encoder

        d_enc = encoder.output_dim
        d     = config.hyperbolic_dim
        C     = config.num_classes
        c     = config.curvature

        # ── Linear projection (always present) ───────────────────────────────
        self.proj = nn.Linear(d_enc, d)

        # ── Optional HVQ Codebook ─────────────────────────────────────────────
        self.codebook = (
            HyperbolicVQCodebook(
                codebook_size     = config.codebook_size,
                dim               = d,
                commitment_weight = config.commitment_weight,
                vq_loss_weight    = config.vq_loss_weight,
                curvature         = c,
            )
            if config.use_codebook else None
        )

        # ── Optional HEL ─────────────────────────────────────────────────────
        self.hel = (
            HyperbolicEmotionLens(curvature=c)
            if config.use_hel else None
        )

        # ── Pooling ───────────────────────────────────────────────────────────
        if config.pooling == "attention":
            if config.use_hyperbolic:
                self.pooling_layer = HyperbolicAttentionPooling(dim=d, curvature=c)
            else:
                # Euclidean attention pooling
                self.pooling_layer = _EuclideanAttentionPooling(dim=d)
        elif config.pooling == "mean":
            self.pooling_layer = _MeanPooling()
        else:  # "max"
            self.pooling_layer = _MaxPooling()

        # ── Classifier ───────────────────────────────────────────────────────
        self.classifier = nn.Linear(d, C)

        # ── Prototype bank + OT (only when use_ot=True) ───────────────────────
        if config.use_ot:
            self.prototype_bank = PrototypeBank(
                num_classes   = C,
                dim           = d,
                curvature     = c,
                frechet_iters = config.frechet_iters,
            )
        else:
            self.prototype_bank = None

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map Euclidean tensor to Poincaré ball (only when use_hyperbolic).

        A raw Linear layer outputs vectors whose norm grows as √d (≈11 for d=128).
        expmap0 uses tanh(norm), which saturates to 1 for norm >> 1 and pushes
        every point to the boundary → logmap0 / Poincaré-dist return NaN/Inf.

        We apply a soft rescaling  x / (‖x‖ + 1)  that:
          - leaves small-norm vectors nearly unchanged
          - smoothly compresses large norms to < 1
          - is differentiable everywhere
        """
        if self.config.use_hyperbolic:
            norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            x    = x / (norm + 1.0)          # soft normalise → norm < 1
            return poincare_clip(expmap0(x, self.config.curvature), self.config.curvature)
        return x

    def _pool(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Apply the chosen pooling and return a flat (B, d) vector in tangent space.
        """
        if self.config.use_hyperbolic and self.config.pooling == "attention":
            u_tang, _ = self.pooling_layer(x, mask)
            return u_tang
        elif self.config.pooling == "attention":
            return self.pooling_layer(x, mask)
        else:
            return self.pooling_layer(x, mask)

    # ── encode ────────────────────────────────────────────────────────────────

    def encode(
        self,
        waveforms: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the full encoding pipeline for a batch of waveforms.

        Args:
            waveforms: (B, T_samples)  float32, 16 kHz, normalised to [-1, 1]
            mask:      (B, T_frames)   bool, True = valid frame  (optional)

        Returns:
            u_rep:   (B, d)   utterance representation (tangent space if hyperbolic)
            u_hyp:   (B, d)   utterance in Poincaré ball (same as u_rep if Euclidean)
            vq_loss: scalar   VQ loss (0.0 when use_codebook=False)
        """
        c       = self.config.curvature
        vq_loss = torch.tensor(0.0, device=waveforms.device)

        # 1. SSL encoder
        feats = self.encoder(waveforms)        # (B, T, d_enc)

        # 2. Project to model dim
        feats = self.proj(feats)               # (B, T, d)

        # 3. Map to Poincaré ball (no-op when Euclidean)
        b = self._to_ball(feats)               # (B, T, d)

        # 4. Optional: HVQ Codebook
        if self.codebook is not None:
            quantized, vq_loss, _ = self.codebook(b)
            b = poincare_clip(mobius_add(b, quantized, c), c)

        # 5. Optional: HEL calibration
        if self.hel is not None:
            b = self.hel(b)

        # 6. Pool to utterance level
        u_rep = self._pool(b, mask)            # (B, d)

        # 7. Hyperbolic utterance embedding for OT / prototypes
        if self.config.use_hyperbolic:
            u_hyp = poincare_clip(expmap0(u_rep, c), c)
        else:
            u_hyp = u_rep

        return u_rep, u_hyp, vq_loss

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        waveforms: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Args:
            waveforms: (B, T_samples)
            mask:      (B, T_frames)  optional

        Returns:
            logits:  (B, num_classes)
            u_hyp:   (B, d)           utterance embedding (for OT)
            vq_loss: scalar
        """
        u_rep, u_hyp, vq_loss = self.encode(waveforms, mask)
        logits = self.classifier(u_rep)
        return logits, u_hyp, vq_loss

    # ── class prior ──────────────────────────────────────────────────────────

    @staticmethod
    def _class_prior(labels: torch.Tensor, C: int) -> torch.Tensor:
        counts = torch.bincount(labels, minlength=C).float()
        return counts / counts.sum().clamp(min=1e-8)

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        source_loader: DataLoader,
        target_loader: Optional[DataLoader] = None,
        *,
        verbose: bool = True,
    ) -> "NOVAARC":
        """
        Train NOVA-ARC.

        Each DataLoader yields ``(waveforms, labels)`` where:
          - waveforms: ``(B, T_samples)`` float32 tensor, 16 kHz, normalised
          - labels:    ``(B,)`` int64 tensor, class indices 0…num_classes-1

        When ``use_ot=False``, ``target_loader`` is not used and can be omitted.

        Args:
            source_loader: DataLoader for labelled source domain
            target_loader: DataLoader for unlabelled target domain
                           (required when use_ot=True)
            verbose:       print per-epoch loss breakdown

        Returns:
            self
        """
        cfg    = self.config
        device = torch.device(cfg.device)
        self.to(device)

        if cfg.use_ot and target_loader is None:
            raise ValueError(
                "target_loader is required when use_ot=True. "
                "Pass target_loader or set use_ot=False in NOVAARCConfig."
            )

        if cfg.use_ot:
            self.prototype_bank.to(device)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=cfg.lr,
        )

        # Pre-compute class prior from full source label set
        if cfg.use_ot:
            all_labels = torch.cat([lbl for _, lbl in source_loader])
            class_prior = self._class_prior(all_labels, cfg.num_classes).to(device)

        for epoch in range(1, cfg.epochs + 1):
            self.train()
            stats: Dict[str, float] = {
                "total": 0.0, "L_S": 0.0, "L_OPT": 0.0, "L_OT_CE": 0.0
            }
            n = 0

            src_iter = iter(source_loader)
            tgt_iter = iter(target_loader) if cfg.use_ot else None

            while True:
                try:
                    src_wav, src_lbl = next(src_iter)
                except StopIteration:
                    break

                src_wav = src_wav.to(device)
                src_lbl = src_lbl.to(device)

                src_logits, src_u_hyp, src_vq = self(src_wav)

                if cfg.use_ot:
                    # Get target batch
                    try:
                        tgt_wav, _ = next(tgt_iter)
                    except StopIteration:
                        tgt_iter = iter(target_loader)
                        tgt_wav, _ = next(tgt_iter)

                    tgt_wav    = tgt_wav.to(device)
                    tgt_logits, tgt_u_hyp, tgt_vq = self(tgt_wav)

                    # Accumulate source embeddings for prototype refresh
                    self.prototype_bank.accumulate(src_u_hyp.detach(), src_lbl)

                    # OT with current prototypes
                    protos  = self.prototype_bank.prototypes.to(device)
                    ot_out  = compute_ot(
                        prototypes  = protos,
                        target_hyp  = tgt_u_hyp.detach(),
                        class_prior = class_prior,
                        curvature   = cfg.curvature,
                        eps         = cfg.sinkhorn_eps,
                        n_iter      = cfg.sinkhorn_iters,
                    )
                    soft_labels = ot_out["soft_labels"].to(device)
                    ot_cost     = ot_out["ot_cost"]

                    losses = nova_arc_loss(
                        source_logits = src_logits,
                        source_labels = src_lbl,
                        target_logits = tgt_logits,
                        soft_labels   = soft_labels,
                        ot_cost       = ot_cost,
                        lambda_opt    = cfg.lambda_opt,
                        lambda_ce     = cfg.lambda_ce,
                    )
                    total = losses["total"] + src_vq + tgt_vq

                    stats["L_OPT"]   += losses["L_OPT"].item()
                    stats["L_OT_CE"] += losses["L_OT_CE"].item()

                else:
                    # Source-only training
                    import torch.nn.functional as _F
                    total = _F.cross_entropy(src_logits, src_lbl) + src_vq
                    losses = {"L_S": total.detach()}

                optimizer.zero_grad()
                total.backward()
                optimizer.step()

                stats["total"] += total.item()
                stats["L_S"]   += losses["L_S"].item()
                n += 1

            if cfg.use_ot:
                self.prototype_bank.refresh(device=device)

            if verbose and n > 0:
                avg = {k: v / n for k, v in stats.items()}
                if cfg.use_ot:
                    print(
                        f"Epoch {epoch:3d}/{cfg.epochs} | "
                        f"total={avg['total']:.4f}  "
                        f"L_S={avg['L_S']:.4f}  "
                        f"L_OPT={avg['L_OPT']:.4f}  "
                        f"L_OT_CE={avg['L_OT_CE']:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:3d}/{cfg.epochs} | "
                        f"total={avg['total']:.4f}  "
                        f"L_S={avg['L_S']:.4f}"
                    )

        return self

    # ── evaluate ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Compute accuracy and weighted-F1 on a labelled DataLoader.

        Args:
            loader: yields (waveforms, labels)

        Returns:
            {"accuracy": float, "weighted_f1": float}
        """
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise ImportError("pip install scikit-learn")

        device = device or torch.device(self.config.device)
        self.eval().to(device)

        preds, gts = [], []
        for wav, lbl in loader:
            logits, _, _ = self(wav.to(device))
            preds.append(logits.argmax(-1).cpu())
            gts.append(lbl)

        preds = torch.cat(preds).numpy()
        gts   = torch.cat(gts).numpy()

        return {
            "accuracy":    float((preds == gts).mean()),
            "weighted_f1": float(f1_score(gts, preds, average="weighted")),
        }

    # ── predict ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return predicted class indices for all samples.

        Args:
            loader: yields (waveforms,) or (waveforms, labels)

        Returns:
            (N,) int64 predicted class indices
        """
        device = device or torch.device(self.config.device)
        self.eval().to(device)

        preds = []
        for batch in loader:
            wav = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits, _, _ = self(wav.to(device))
            preds.append(logits.argmax(-1).cpu())

        return torch.cat(preds)

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save weights + prototypes + config to a .pt file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ckpt = {
            "model_state": self.state_dict(),
            "config":      self.config.__dict__,
        }
        if self.prototype_bank is not None:
            ckpt["prototype_bank"] = self.prototype_bank.state_dict()
        torch.save(ckpt, path)

    @classmethod
    def load(
        cls,
        path:         str,
        encoder:      BaseEncoder,
        map_location: Optional[str] = None,
    ) -> "NOVAARC":
        """
        Load a saved model.

        Args:
            path:         path to .pt file written by save()
            encoder:      same encoder architecture used during training
            map_location: e.g. "cpu" or "cuda"
        """
        ckpt   = torch.load(path, map_location=map_location or "cpu")
        config = NOVAARCConfig(**ckpt["config"])
        model  = cls(encoder=encoder, config=config)
        model.load_state_dict(ckpt["model_state"])
        if "prototype_bank" in ckpt and model.prototype_bank is not None:
            model.prototype_bank.load_state_dict(ckpt["prototype_bank"])
        return model


# ── Euclidean attention pooling (used when use_hyperbolic=False) ──────────────

class _EuclideanAttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # x: (B, T, d)
        scores = x @ self.w                          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn   = torch.softmax(scores, dim=-1)
        attn   = torch.nan_to_num(attn, nan=0.0)
        return (attn.unsqueeze(-1) * x).sum(dim=1)   # (B, d)
