"""
nova_arc/losses.py
------------------
Loss functions for NOVA-ARC training.

Three losses are combined:

    L_total = L_S  +  λ_OPT · L_OPT  +  λ_CE · L_OT-CE

  L_S       — standard cross-entropy on source domain
  L_OPT     — hyperbolic OT transport cost  <C_hyp, P>_F
  L_OT-CE   — soft cross-entropy on target domain using OT pseudo-labels

All losses are computed in a single forward call via `nova_arc_loss`.
"""

import torch
import torch.nn.functional as F


EPS = 1e-8


# ── Individual losses ─────────────────────────────────────────────────────────

def source_ce_loss(
    logits: torch.Tensor,   # (B_S, C)  source classifier logits
    labels: torch.Tensor,   # (B_S,)    integer class labels
) -> torch.Tensor:
    """Standard cross-entropy loss on the source domain."""
    return F.cross_entropy(logits, labels)


def ot_transport_cost(ot_cost: torch.Tensor) -> torch.Tensor:
    """
    OT transport cost  <C_hyp, P>_F.
    Already computed by transport.compute_ot; just return it.
    """
    return ot_cost


def ot_soft_ce_loss(
    logits:      torch.Tensor,   # (B_T, C)  target classifier logits
    soft_labels: torch.Tensor,   # (B_T, C)  soft pseudo-labels from OT
) -> torch.Tensor:
    """
    Soft cross-entropy on the target domain using OT pseudo-labels.

        L_OT-CE = - Σ_j Σ_c ŷ_{j,c}^soft · log p_{j,c}

    Args:
        logits:      (B_T, C)
        soft_labels: (B_T, C)  each row is a probability distribution
    """
    log_probs = F.log_softmax(logits, dim=-1)           # (B_T, C)
    loss      = -(soft_labels * log_probs).sum(dim=-1)  # (B_T,)
    return loss.mean()


# ── Combined loss ─────────────────────────────────────────────────────────────

def nova_arc_loss(
    source_logits:  torch.Tensor,   # (B_S, C)
    source_labels:  torch.Tensor,   # (B_S,)
    target_logits:  torch.Tensor,   # (B_T, C)
    soft_labels:    torch.Tensor,   # (B_T, C)  from OT
    ot_cost:        torch.Tensor,   # scalar
    lambda_opt:     float = 0.1,
    lambda_ce:      float = 0.1,
) -> dict:
    """
    Compute the full NOVA-ARC training loss.

        L_total = L_S  +  λ_OPT · L_OPT  +  λ_CE · L_OT-CE

    Args:
        source_logits:  (B_S, C)  logits from source samples
        source_labels:  (B_S,)    ground-truth labels for source
        target_logits:  (B_T, C)  logits from target samples
        soft_labels:    (B_T, C)  OT pseudo-labels for target samples
        ot_cost:        scalar    OT transport cost from compute_ot()
        lambda_opt:     weight on OT cost
        lambda_ce:      weight on OT-CE loss

    Returns:
        dict with keys:
          "total":   combined scalar loss (use for backward)
          "L_S":     source cross-entropy
          "L_OPT":   OT transport cost
          "L_OT_CE": soft CE on target
    """
    L_S     = source_ce_loss(source_logits, source_labels)
    L_OPT   = ot_transport_cost(ot_cost)
    L_OT_CE = ot_soft_ce_loss(target_logits, soft_labels)

    total = L_S + lambda_opt * L_OPT + lambda_ce * L_OT_CE

    return {
        "total":   total,
        "L_S":     L_S.detach(),
        "L_OPT":   L_OPT.detach(),
        "L_OT_CE": L_OT_CE.detach(),
    }
