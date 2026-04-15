"""
nova_arc/hyperbolic.py
----------------------
All Poincaré ball operations used throughout NOVA-ARC.

Every function is a pure PyTorch operation and works on batched tensors.
Curvature c > 0  corresponds to sectional curvature κ = -c.
The paper uses κ = -1.0  →  c = 1.0  (default everywhere).
"""

import torch
import torch.nn.functional as F

# Small epsilon for numerical stability
EPS = 1e-8


# ─── Boundary clip ──────────────────────────────────────────────────────────

def poincare_clip(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Project points back inside the open Poincaré ball  ||x|| < 1/√c.
    Applied after every operation that might push points to the boundary.
    """
    max_norm = (1.0 - EPS) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale


# ─── Exponential and logarithmic maps at the origin ─────────────────────────

def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Exponential map at the origin:  tangent space  →  Poincaré ball.

        exp_0^c(v) = tanh(√c ‖v‖) · v / (√c ‖v‖)

    Args:
        v:  (..., d)  tangent vector(s)
        c:  curvature (positive scalar)

    Returns:
        (..., d)  point(s) in the Poincaré ball
    """
    sqrt_c = c ** 0.5
    norm   = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    return poincare_clip(torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm), c)


def logmap0(x: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Logarithmic map at the origin:  Poincaré ball  →  tangent space.

        log_0^c(x) = (1/√c) · arctanh(√c ‖x‖) · x / ‖x‖

    Args:
        x:  (..., d)  point(s) in the Poincaré ball
        c:  curvature (positive scalar)

    Returns:
        (..., d)  tangent vector(s)
    """
    sqrt_c = c ** 0.5
    norm   = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    # clamp norm to stay strictly inside the ball before arctanh
    safe_norm = (sqrt_c * norm).clamp(max=1.0 - EPS)
    return torch.arctanh(safe_norm) * x / (sqrt_c * norm)


# ─── Möbius addition ────────────────────────────────────────────────────────

def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Möbius addition in the Poincaré ball:

        x ⊕ y = [(1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 - c‖x‖²)y]
                ─────────────────────────────────────────────
                     1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²

    Args:
        x, y:  (..., d)  points in the Poincaré ball
        c:     curvature

    Returns:
        (..., d)  result of Möbius addition, clipped to ball
    """
    x2  = (x * x).sum(dim=-1, keepdim=True)          # ‖x‖²
    y2  = (y * y).sum(dim=-1, keepdim=True)          # ‖y‖²
    xy  = (x * y).sum(dim=-1, keepdim=True)          # ⟨x, y⟩
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return poincare_clip(num / den.clamp(min=EPS), c)


# ─── Poincaré distance ──────────────────────────────────────────────────────

def poincare_dist(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """
    Poincaré distance between two points (same shape):

        d_c(x, y) = (2/√c) · arctanh(√c ‖-x ⊕ y‖)

    Args:
        x, y:  (..., d)
        c:     curvature

    Returns:
        (...)  scalar distance(s)
    """
    sqrt_c = c ** 0.5
    diff   = mobius_add(-x, y, c)
    norm   = diff.norm(dim=-1).clamp(min=EPS, max=(1.0 - EPS) / sqrt_c)
    return (2.0 / sqrt_c) * torch.arctanh(sqrt_c * norm)


def poincare_dist_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
) -> torch.Tensor:
    """
    All-pairs Poincaré distance matrix.

    Args:
        x:  (N, d)  first set of points
        y:  (M, d)  second set of points
        c:  curvature

    Returns:
        (N, M)  pairwise distances
    """
    # Expand for broadcasting: (N,1,d) and (1,M,d)
    x_ = x.unsqueeze(1)   # (N, 1, d)
    y_ = y.unsqueeze(0)   # (1, M, d)
    diff = mobius_add(-x_, y_, c)              # (N, M, d)
    sqrt_c = c ** 0.5
    norm   = diff.norm(dim=-1).clamp(min=EPS, max=(1.0 - EPS) / sqrt_c)  # (N, M)
    return (2.0 / sqrt_c) * torch.arctanh(sqrt_c * norm)


# ─── Fréchet mean ────────────────────────────────────────────────────────────

def frechet_mean(
    points:   torch.Tensor,
    c:        float = 1.0,
    weights:  torch.Tensor = None,
    n_iter:   int = 20,
    lr:       float = 0.1,
) -> torch.Tensor:
    """
    Fréchet mean of a set of points in the Poincaré ball via
    Riemannian gradient descent.

        μ* = argmin_z Σ_i w_i · d_c²(z, x_i)

    Args:
        points:   (N, d)  points in Poincaré ball
        c:        curvature
        weights:  (N,)  non-negative weights (uniform if None)
        n_iter:   number of gradient steps
        lr:       step size

    Returns:
        (d,)  Fréchet mean in the Poincaré ball
    """
    N, d = points.shape
    if weights is None:
        weights = points.new_ones(N) / N
    else:
        weights = weights / weights.sum().clamp(min=EPS)

    # Initialise: weighted Euclidean mean mapped into ball
    mu = expmap0(
        (weights.unsqueeze(-1) * logmap0(points, c)).sum(dim=0), c
    ).detach().requires_grad_(False)

    for _ in range(n_iter):
        # log_{mu}(x_i)  ≈  logmap0((-mu) ⊕ x_i)
        log_pts = logmap0(mobius_add(-mu.unsqueeze(0), points, c), c)  # (N, d)
        grad    = (weights.unsqueeze(-1) * log_pts).sum(dim=0)          # (d,)
        mu      = poincare_clip(expmap0(logmap0(mu, c) + lr * grad, c), c)

    return mu.detach()
