"""
nova_arc/model.py
-----------------
NOVA-ARC core model.

Design philosophy
-----------------
The library does NOT handle feature extraction.
Users extract frame-level features however they like — WavLM, wav2vec 2.0,
voc2vec, librosa MFCCs, openSMILE, or any other tool — and pass them in as
tensors.  NOVA-ARC handles everything from the projection into hyperbolic
space onward.

DataLoader contract
-------------------
Each DataLoader must yield  (features, labels)  where:
    features : (B, T, input_dim)   float32  frame-level feature tensor
    labels   : (B,)                int64    class indices  0 … num_classes-1

For the unlabelled target loader, pass labels as -1 (they are ignored).

Minimal usage
-------------
    from nova_arc import NOVAARC, NOVAARCConfig

    config = NOVAARCConfig(num_classes=5, input_dim=768)
    model  = NOVAARC(config)
    model.fit(source_loader, target_loader)
    model.evaluate(test_loader)
    model.save("checkpoint.pt")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nova_arc.hyperbolic import expmap0, logmap0, mobius_add, poincare_clip
from nova_arc.codebook    import HyperbolicVQCodebook
from nova_arc.emotion_lens import HyperbolicEmotionLens
from nova_arc.pooling     import HyperbolicAttentionPooling
from nova_arc.prototypes  import PrototypeBank
from nova_arc.transport   import compute_ot
from nova_arc.losses      import nova_arc_loss


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class NOVAARCConfig:
    """
    Full configuration for NOVA-ARC.

    Only  num_classes  and  input_dim  are required.
    Everything else has sensible defaults matching the ACL 2026 paper.

    Parameters
    ----------
    num_classes : int
        Number of emotion (or any target) classes in your dataset.

    input_dim : int
        Dimension of the frame-level features your DataLoader provides.
        Must match the feature extractor you used:
            WavLM / wav2vec 2.0 / voc2vec → 768
            MMS-1B                         → 1024
            librosa 40-dim MFCCs           → 40
            any custom features            → whatever your extractor outputs

    Architecture toggles
    --------------------
    use_hyperbolic : bool
        True  → full Poincaré ball pipeline  (default, paper setting)
        False → pure Euclidean; also auto-disables codebook and HEL

    use_codebook : bool
        True  → Hyperbolic VQ prosody codebook  (default)
        False → skip; ablation

    use_hel : bool
        True  → Hyperbolic Emotion Lens radial calibration  (default)
        False → skip; ablation

    use_ot : bool
        True  → Sinkhorn OT domain adaptation  (default)
        False → source-only supervised training; target_loader not needed

    pooling : str
        "attention"  learnable attention pooling over frames  (default)
        "mean"       simple mean pooling
        "max"        max pooling

    Dimensions
    ----------
    hidden_dim : int
        Internal Poincaré ball embedding dimension d  (default 256)

    curvature : float
        Poincaré ball curvature c  (κ = -c, paper uses c = 1.0)

    Codebook
    --------
    codebook_size : int       number of codewords K  (default 256)
    commitment_weight : float β for VQ commitment loss  (default 0.25)
    vq_loss_weight : float    scale on total VQ loss  (default 1.0)

    Prototype bank
    --------------
    frechet_iters : int
        Riemannian GD steps for Fréchet mean  (default 20)

    Optimal Transport
    -----------------
    sinkhorn_eps : float    regularisation ε  (default 0.05)
    sinkhorn_iters : int    Sinkhorn iterations  (default 50)
    lambda_opt : float      weight on L_OPT  (default 1.0)
    lambda_ce : float       weight on L_OT-CE  (default 1.0)

    Training
    --------
    lr : float      Adam learning rate  (default 1e-4)
    epochs : int    training epochs  (default 30)
    device : str    "cuda" / "cpu" / "mps"
    """

    # ── Required ─────────────────────────────────────────────────────────────
    num_classes : int
    input_dim   : int

    # ── Architecture toggles ─────────────────────────────────────────────────
    use_hyperbolic : bool = True
    use_codebook   : bool = True
    use_hel        : bool = True
    use_ot         : bool = True
    pooling        : str  = "attention"   # "attention" | "mean" | "max"

    # ── Dimensions ───────────────────────────────────────────────────────────
    hidden_dim : int   = 256
    curvature  : float = 1.0

    # ── Codebook ─────────────────────────────────────────────────────────────
    codebook_size      : int   = 256
    commitment_weight  : float = 0.25
    vq_loss_weight     : float = 1.0

    # ── Prototype bank ───────────────────────────────────────────────────────
    frechet_iters : int = 20

    # ── Optimal Transport ─────────────────────────────────────────────────────
    sinkhorn_eps   : float = 0.05
    sinkhorn_iters : int   = 50
    lambda_opt     : float = 1.0
    lambda_ce      : float = 1.0

    # ── Training ─────────────────────────────────────────────────────────────
    lr     : float = 1e-4
    epochs : int   = 30
    device : str   = "cpu"

    def __post_init__(self):
        if self.pooling not in ("attention", "mean", "max"):
            raise ValueError(
                f"pooling must be 'attention', 'mean', or 'max'  —  got '{self.pooling}'"
            )
        if not self.use_hyperbolic:
            # codebook and HEL require hyperbolic geometry
            self.use_codebook = False
            self.use_hel      = False


# ── Internal pooling helpers ──────────────────────────────────────────────────

class _MeanPool(nn.Module):
    def forward(self, x, mask=None):
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            return (x * m).sum(1) / m.sum(1).clamp(min=1e-8)
        return x.mean(dim=1)

class _MaxPool(nn.Module):
    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        return x.max(dim=1).values

class _EuclideanAttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
    def forward(self, x, mask=None):
        scores = x @ self.w
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        return (attn.unsqueeze(-1) * x).sum(dim=1)


# ── Main model ────────────────────────────────────────────────────────────────

class NOVAARC(nn.Module):
    """
    NOVA-ARC model.

    Takes pre-extracted frame features as input — no encoder is built-in.
    Users extract features with any tool (WavLM, wav2vec 2.0, librosa, etc.)
    and pass them directly via their DataLoader.

    Parameters
    ----------
    config : NOVAARCConfig
        Full model + training configuration.

    Example
    -------
    >>> from nova_arc import NOVAARC, NOVAARCConfig
    >>> config = NOVAARCConfig(num_classes=5, input_dim=768)
    >>> model  = NOVAARC(config)
    >>> model.fit(source_loader, target_loader)
    >>> model.evaluate(test_loader)
    """

    def __init__(self, config: NOVAARCConfig):
        super().__init__()
        self.config = config

        d   = config.hidden_dim
        C   = config.num_classes
        c   = config.curvature
        inp = config.input_dim

        # ── Input projection  (input_dim → hidden_dim) ───────────────────────
        self.proj = nn.Linear(inp, d)

        # ── Optional: HVQ Codebook ────────────────────────────────────────────
        self.codebook = (
            HyperbolicVQCodebook(
                codebook_size     = config.codebook_size,
                dim               = d,
                commitment_weight = config.commitment_weight,
                vq_loss_weight    = config.vq_loss_weight,
                curvature         = c,
            ) if config.use_codebook else None
        )

        # ── Optional: Hyperbolic Emotion Lens ─────────────────────────────────
        self.hel = (
            HyperbolicEmotionLens(curvature=c)
            if config.use_hel else None
        )

        # ── Pooling ───────────────────────────────────────────────────────────
        if config.pooling == "attention":
            self.pool = (
                HyperbolicAttentionPooling(dim=d, curvature=c)
                if config.use_hyperbolic else _EuclideanAttnPool(d)
            )
        elif config.pooling == "mean":
            self.pool = _MeanPool()
        else:
            self.pool = _MaxPool()

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Linear(d, C)

        # ── Prototype bank (optional, used by OT) ────────────────────────────
        self.prototype_bank = (
            PrototypeBank(
                num_classes   = C,
                dim           = d,
                curvature     = c,
                frechet_iters = config.frechet_iters,
            ) if config.use_ot else None
        )

    # ── geometry helpers ─────────────────────────────────────────────────────

    def _to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Soft-normalise then expmap into Poincaré ball."""
        if not self.config.use_hyperbolic:
            return x
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x    = x / (norm + 1.0)          # soft normalise → norm < 1
        return poincare_clip(expmap0(x, self.config.curvature), self.config.curvature)

    def _pool_frames(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """Pool (B, T, d) → (B, d) in tangent space."""
        if self.config.use_hyperbolic and self.config.pooling == "attention":
            u_tang, _ = self.pool(x, mask)
            return u_tang
        return self.pool(x, mask)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        features: torch.Tensor,           # (B, T, input_dim)
        mask: Optional[torch.Tensor] = None,  # (B, T) bool, True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        features : (B, T, input_dim)
            Pre-extracted frame-level features.
        mask : (B, T) bool, optional
            True for valid frames, False for padding.

        Returns
        -------
        logits  : (B, num_classes)
        u_hyp   : (B, hidden_dim)   utterance in Poincaré ball  (for OT)
        vq_loss : scalar
        """
        c       = self.config.curvature
        vq_loss = torch.tensor(0.0, device=features.device)

        # 1. Project input features to hidden_dim
        x = self.proj(features)           # (B, T, d)

        # 2. Map to Poincaré ball
        x = self._to_ball(x)              # (B, T, d)

        # 3. HVQ Codebook
        if self.codebook is not None:
            q, vq_loss, _ = self.codebook(x)
            x = poincare_clip(mobius_add(x, q, c), c)

        # 4. HEL calibration
        if self.hel is not None:
            x = self.hel(x)

        # 5. Pool to utterance level
        u = self._pool_frames(x, mask)    # (B, d)

        # 6. Hyperbolic utterance embedding (for OT / prototype bank)
        u_hyp = (
            poincare_clip(expmap0(u, c), c)
            if self.config.use_hyperbolic else u
        )

        # 7. Classify
        logits = self.classifier(u)       # (B, C)

        return logits, u_hyp, vq_loss

    # ── class prior ──────────────────────────────────────────────────────────

    @staticmethod
    def _class_prior(labels: torch.Tensor, C: int) -> torch.Tensor:
        counts = torch.bincount(labels.clamp(min=0), minlength=C).float()
        return counts / counts.sum().clamp(min=1e-8)

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        source_loader  : DataLoader,
        target_loader  : Optional[DataLoader] = None,
        *,
        verbose        : bool = True,
    ) -> "NOVAARC":
        """
        Train NOVA-ARC.

        Parameters
        ----------
        source_loader : DataLoader
            Yields (features, labels) — labelled source domain.
            features : (B, T, input_dim)
            labels   : (B,)  int64, class indices 0 … num_classes-1

        target_loader : DataLoader, optional
            Yields (features, labels) — unlabelled target domain.
            features : (B, T, input_dim)
            labels   : (B,)  ignored (can be -1)
            Required when use_ot=True.

        verbose : bool
            Print per-epoch loss breakdown.

        Returns
        -------
        self
        """
        cfg    = self.config
        device = torch.device(cfg.device)
        self.to(device)

        if cfg.use_ot:
            if target_loader is None:
                raise ValueError(
                    "target_loader is required when use_ot=True.\n"
                    "Either pass target_loader or set use_ot=False in NOVAARCConfig."
                )
            self.prototype_bank.to(device)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=cfg.lr,
        )

        # Pre-compute class prior from source labels
        if cfg.use_ot:
            all_labels  = torch.cat([lbl for _, lbl in source_loader])
            class_prior = self._class_prior(all_labels, cfg.num_classes).to(device)

        for epoch in range(1, cfg.epochs + 1):
            self.train()
            stats = {"total": 0.0, "L_S": 0.0, "L_OPT": 0.0, "L_OT_CE": 0.0}
            n = 0

            src_iter = iter(source_loader)
            tgt_iter = iter(target_loader) if cfg.use_ot else None

            while True:
                # ── source batch ──────────────────────────────────────────────
                try:
                    src_feat, src_lbl = next(src_iter)
                except StopIteration:
                    break

                src_feat = src_feat.to(device)
                src_lbl  = src_lbl.to(device)

                src_logits, src_u_hyp, src_vq = self(src_feat)

                if cfg.use_ot:
                    # ── target batch ──────────────────────────────────────────
                    try:
                        tgt_feat, _ = next(tgt_iter)
                    except StopIteration:
                        tgt_iter    = iter(target_loader)
                        tgt_feat, _ = next(tgt_iter)

                    tgt_feat = tgt_feat.to(device)
                    tgt_logits, tgt_u_hyp, tgt_vq = self(tgt_feat)

                    # accumulate source embeddings for prototype refresh
                    self.prototype_bank.accumulate(src_u_hyp.detach(), src_lbl)

                    # OT with current prototypes
                    protos = self.prototype_bank.prototypes.to(device)
                    ot_out = compute_ot(
                        prototypes  = protos,
                        target_hyp  = tgt_u_hyp.detach(),
                        class_prior = class_prior,
                        curvature   = cfg.curvature,
                        eps         = cfg.sinkhorn_eps,
                        n_iter      = cfg.sinkhorn_iters,
                    )
                    soft_labels = ot_out["soft_labels"].to(device)

                    losses = nova_arc_loss(
                        source_logits = src_logits,
                        source_labels = src_lbl,
                        target_logits = tgt_logits,
                        soft_labels   = soft_labels,
                        ot_cost       = ot_out["ot_cost"],
                        lambda_opt    = cfg.lambda_opt,
                        lambda_ce     = cfg.lambda_ce,
                    )
                    total = losses["total"] + src_vq + tgt_vq

                    stats["L_OPT"]   += losses["L_OPT"].item()
                    stats["L_OT_CE"] += losses["L_OT_CE"].item()

                else:
                    # source-only training
                    total  = F.cross_entropy(src_logits, src_lbl) + src_vq
                    losses = {"L_S": total.detach()}

                optimizer.zero_grad()
                total.backward()
                optimizer.step()

                stats["total"] += total.item()
                stats["L_S"]   += losses["L_S"].item()
                n += 1

            # end of epoch: refresh prototypes
            if cfg.use_ot:
                self.prototype_bank.refresh(device=device)

            if verbose and n > 0:
                avg = {k: v / n for k, v in stats.items()}
                if cfg.use_ot:
                    print(
                        f"Epoch {epoch:3d}/{cfg.epochs}  |  "
                        f"total={avg['total']:.4f}  "
                        f"L_S={avg['L_S']:.4f}  "
                        f"L_OPT={avg['L_OPT']:.4f}  "
                        f"L_OT_CE={avg['L_OT_CE']:.4f}"
                    )
                else:
                    print(
                        f"Epoch {epoch:3d}/{cfg.epochs}  |  "
                        f"total={avg['total']:.4f}  "
                        f"L_S={avg['L_S']:.4f}"
                    )
        return self

    # ── evaluate ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        loader : DataLoader,
        device : Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Compute accuracy and weighted-F1 on a labelled DataLoader.

        Parameters
        ----------
        loader : DataLoader
            Yields (features, labels).

        Returns
        -------
        {"accuracy": float, "weighted_f1": float}
        """
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise ImportError("pip install scikit-learn")

        device = device or torch.device(self.config.device)
        self.eval().to(device)

        preds, gts = [], []
        for feat, lbl in loader:
            logits, _, _ = self(feat.to(device))
            preds.append(logits.argmax(-1).cpu())
            gts.append(lbl)

        preds = torch.cat(preds).numpy()
        gts   = torch.cat(gts).numpy()

        return {
            "accuracy"    : float((preds == gts).mean()),
            "weighted_f1" : float(f1_score(gts, preds, average="weighted", zero_division=0)),
        }

    # ── predict ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        loader : DataLoader,
        device : Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Return predicted class indices for all samples.

        Parameters
        ----------
        loader : DataLoader
            Yields (features,) or (features, labels) — labels are ignored.

        Returns
        -------
        (N,) int64 predicted class indices
        """
        device = device or torch.device(self.config.device)
        self.eval().to(device)

        preds = []
        for batch in loader:
            feat   = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits, _, _ = self(feat.to(device))
            preds.append(logits.argmax(-1).cpu())

        return torch.cat(preds)

    # ── save ─────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Save model weights + prototype bank + config to a .pt checkpoint.

        Parameters
        ----------
        path : str
            File path, e.g. "checkpoints/nova_arc.pt"
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ckpt = {
            "model_state" : self.state_dict(),
            "config"      : self.config.__dict__,
        }
        if self.prototype_bank is not None:
            ckpt["prototype_bank"] = self.prototype_bank.state_dict()
        torch.save(ckpt, path)
        print(f"Saved -> {path}")

    # ── load ─────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        path         : str,
        map_location : Optional[str] = None,
    ) -> "NOVAARC":
        """
        Load a checkpoint saved by  save().

        Parameters
        ----------
        path         : str   path to .pt file
        map_location : str   e.g. "cpu" or "cuda"

        Returns
        -------
        NOVAARC instance with loaded weights and config
        """
        ckpt   = torch.load(path, map_location=map_location or "cpu")
        config = NOVAARCConfig(**ckpt["config"])
        model  = cls(config)
        model.load_state_dict(ckpt["model_state"])
        if "prototype_bank" in ckpt and model.prototype_bank is not None:
            model.prototype_bank.load_state_dict(ckpt["prototype_bank"])
        print(f"Loaded <- {path}")
        return model
