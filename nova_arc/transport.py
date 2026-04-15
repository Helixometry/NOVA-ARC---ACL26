"""
nova_arc/transport.py
---------------------
Optimal Transport via Sinkhorn Algorithm in Hyperbolic Space.

Given:
  - Source class prototypes  μ^(c)  (C, d)  — hyperbolic
  - Target utterance embeddings  u_hyp  (M, d)  — hyperbolic

We solve the regularised OT problem:

    min_{P} <C_hyp, P>_F  −  ε H(P)

    subject to:  P 1_M = a  (source marginal = class prior)
                 P^T 1_C = b  (target marginal = uniform)

where C_hyp[c,j] = d_P(μ^(c), u_j^T)  is the Poincaré distance cost,
and H(P) = −Σ_{c,j} P_{c,j} log P_{c,j}  is the entropy regulariser.

Sinkhorn iterations (log-domain for numerical stability):
    u ← a / (K v)
    v ← b / (K^T u)
where K = exp(−C / ε).

After solving, soft pseudo-labels for each target sample are:
    ŷ_j^soft = P[:, j] / Σ_c P[c, j]   (i.e. the normalised column of P)
"""

import torch
import torch.nn.functional as F

from nova_arc.hyperbolic import poincare_dist_matrix

EPS = 1e-8


# ── Sinkhorn in log-domain ───────────────────────────────────────────────────

def sinkhorn_log(
    cost:      torch.Tensor,   # (C, M)  cost matrix
    a:         torch.Tensor,   # (C,)    source marginal (class prior)
    b:         torch.Tensor,   # (M,)    target marginal (uniform)
    eps:       float = 0.05,
    n_iter:    int   = 50,
) -> torch.Tensor:
    """
    Sinkhorn algorithm in log-domain for numerical stability.

    Args:
        cost:   (C, M)  pairwise cost matrix
        a:      (C,)    source marginal; sums to 1
        b:      (M,)    target marginal; sums to 1
        eps:    regularisation strength ε
        n_iter: number of Sinkhorn iterations

    Returns:
        P:  (C, M)  optimal transport plan
    """
    log_a = a.log().clamp(min=-1e9)   # (C,)
    log_b = b.log().clamp(min=-1e9)   # (M,)

    # Normalise cost so the kernel doesn't underflow to all-zeros when
    # Poincaré distances are large (e.g. early training, boundary saturation).
    cost  = cost / cost.max().clamp(min=1e-8)

    # log kernel:  log K = -C / ε
    log_K = -cost / eps                # (C, M)

    # Dual variables in log-domain
    log_u = torch.zeros_like(log_a)   # (C,)
    log_v = torch.zeros_like(log_b)   # (M,)

    for _ in range(n_iter):
        # log_u = log_a - logsumexp(log_K + log_v, dim=1)
        log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
        # log_v = log_b - logsumexp(log_K^T + log_u, dim=1)
        log_v = log_b - torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)

    # Transport plan: P = diag(u) K diag(v)
    log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)  # (C, M)
    P     = log_P.exp()
    return P


# ── Main OT module ────────────────────────────────────────────────────────────

def compute_ot(
    prototypes:   torch.Tensor,   # (C, d)  source class prototypes (hyperbolic)
    target_hyp:   torch.Tensor,   # (M, d)  target utterance embeddings (hyperbolic)
    class_prior:  torch.Tensor,   # (C,)    source class frequencies (normalised)
    curvature:    float = 1.0,
    eps:          float = 0.05,
    n_iter:       int   = 50,
) -> dict:
    """
    Solve OT between source prototypes and target embeddings.

    Args:
        prototypes:  (C, d)  hyperbolic source class prototypes
        target_hyp:  (M, d)  hyperbolic target utterance embeddings
        class_prior: (C,)    normalised class frequencies on source domain
        curvature:   Poincaré ball curvature c
        eps:         Sinkhorn regularisation ε
        n_iter:      Sinkhorn iterations

    Returns:
        dict with keys:
          "cost_matrix":    (C, M)  Poincaré distance matrix
          "transport_plan": (C, M)  OT plan P
          "soft_labels":    (M, C)  soft pseudo-labels for target samples
          "ot_cost":        scalar  <C_hyp, P>_F
    """
    C = prototypes.shape[0]
    M = target_hyp.shape[0]

    # Cost matrix: Poincaré distances between each prototype and target embedding
    cost_matrix = poincare_dist_matrix(prototypes, target_hyp, curvature)  # (C, M)

    # Marginals
    a = class_prior.to(cost_matrix.device)                    # (C,)
    b = torch.full((M,), 1.0 / M, device=cost_matrix.device)  # (M,) uniform

    # Normalise marginals (safety)
    a = a / a.sum().clamp(min=EPS)
    b = b / b.sum().clamp(min=EPS)

    # Solve OT
    P = sinkhorn_log(cost_matrix.detach(), a, b, eps=eps, n_iter=n_iter)  # (C, M)

    # OT cost  (differentiable w.r.t. cost_matrix via P)
    ot_cost = (cost_matrix * P.detach()).sum()

    # Soft pseudo-labels: normalise each column of P  → (M, C)
    col_sums    = P.sum(dim=0, keepdim=True).clamp(min=EPS)  # (1, M)
    soft_labels = (P / col_sums).T                            # (M, C)

    return {
        "cost_matrix":    cost_matrix,
        "transport_plan": P,
        "soft_labels":    soft_labels,
        "ot_cost":        ot_cost,
    }
