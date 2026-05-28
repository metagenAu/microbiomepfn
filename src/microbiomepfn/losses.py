"""
Loss functions.

Pretraining loss is a weighted sum of:

  L_count   — negative binomial (or zero-inflated NB) log-likelihood per cell,
              computed only on masked cells (the MLM target).
  L_y       — outcome prediction loss when y exists (MSE or BCE depending on y_kind),
              computed only on query samples.
  L_effect  — auxiliary regression of per-taxon predicted effect magnitudes onto
              the true effect L2 norm (a weak supervision signal that nudges the
              taxon representations toward something effect-aware). Disabled by
              default — turn on with effect_weight > 0.
"""
from __future__ import annotations
from typing import Optional, Dict
import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Negative binomial / ZINB log-likelihoods
# ---------------------------------------------------------------------------
def nb_log_prob(y: torch.Tensor, log_mu: torch.Tensor,
                log_theta: torch.Tensor) -> torch.Tensor:
    """
    Negative binomial log-likelihood with parameterization:
        E[y]   = mu = exp(log_mu)
        Var[y] = mu + mu^2 / theta,   theta = exp(log_theta)
    Returns log P(y | mu, theta), same shape as y.
    """
    mu = log_mu.exp().clamp(min=1e-8)
    theta = log_theta.exp().clamp(min=1e-8)
    # log P(y) = lgamma(y + theta) - lgamma(theta) - lgamma(y + 1)
    #          + theta * (log_theta - log(theta + mu))
    #          + y     * (log_mu    - log(theta + mu))
    log_theta_plus_mu = torch.log(theta + mu)
    y_f = y.float()
    log_p = (
        torch.lgamma(y_f + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y_f + 1.0)
        + theta * (log_theta - log_theta_plus_mu)
        + y_f * (log_mu - log_theta_plus_mu)
    )
    return log_p


def zinb_log_prob(y: torch.Tensor, log_mu: torch.Tensor,
                  log_theta: torch.Tensor, zi_logit: torch.Tensor) -> torch.Tensor:
    """
    Zero-inflated negative binomial. P(y=0) = pi + (1-pi) * NB(0); P(y>0) = (1-pi)*NB(y).
    Numerically stable via log-sum-exp / log-softplus.
    """
    nb_lp = nb_log_prob(y, log_mu, log_theta)
    log_pi = F.logsigmoid(zi_logit)              # log(pi)
    log_1mpi = F.logsigmoid(-zi_logit)            # log(1 - pi)

    # For y == 0: log( pi + (1-pi) * NB(0) ) = logaddexp(log_pi, log_1mpi + nb_lp_at_0)
    nb_lp_at_zero = nb_log_prob(torch.zeros_like(y), log_mu, log_theta)
    log_p_zero = torch.logaddexp(log_pi, log_1mpi + nb_lp_at_zero)

    # For y > 0: log_1mpi + nb_lp(y)
    log_p_nonzero = log_1mpi + nb_lp

    is_zero = (y == 0).float()
    return is_zero * log_p_zero + (1.0 - is_zero) * log_p_nonzero


# ---------------------------------------------------------------------------
# Master loss assembly
# ---------------------------------------------------------------------------
def compute_loss(out: Dict[str, torch.Tensor],
                 counts: torch.Tensor,
                 visible_cell: torch.Tensor,
                 y: Optional[torch.Tensor] = None,
                 y_kind: Optional[str] = None,
                 visible_sample: Optional[torch.Tensor] = None,
                 true_effects: Optional[torch.Tensor] = None,
                 use_zinb: bool = True,
                 y_weight: float = 0.3,
                 effect_weight: float = 0.0,
                 ) -> Dict[str, torch.Tensor]:
    """
    Returns dict with 'loss', 'loss_count', 'loss_y', 'loss_effect', 'n_masked'.

    counts:        (N, T) int
    visible_cell:  (N, T) bool — True = visible (not masked). We compute count loss
                                  on the masked cells (~visible_cell).
    y:             (N,) or None
    y_kind:        'cont' / 'binary' / None
    visible_sample:(N,) bool — True = context sample (use for ICL); compute y loss
                                on the queries (~visible_sample).
    true_effects:  (P, T) ground-truth effect matrix from prior; used only if
                  effect_weight > 0.
    """
    device = out['log_mu'].device

    # ---- Count loss on masked cells ----
    masked = ~visible_cell                                        # (N, T)
    if use_zinb:
        cell_lp = zinb_log_prob(counts, out['log_mu'],
                                out['log_theta'], out['zi_logit'])
    else:
        cell_lp = nb_log_prob(counts, out['log_mu'], out['log_theta'])
    # Mean negative log-likelihood per masked cell
    n_masked = masked.sum().clamp(min=1)
    loss_count = -(cell_lp * masked.float()).sum() / n_masked

    # ---- y loss on query samples ----
    loss_y = torch.tensor(0.0, device=device)
    if y is not None and y_kind is not None and visible_sample is not None:
        query = ~visible_sample                                   # (N,)
        n_query = query.sum().clamp(min=1)
        if y_kind == 'cont':
            yt = y.float()
            # Normalize y for stable scaling
            yt_mean = yt[visible_sample].mean() if visible_sample.any() else yt.mean()
            yt_std = yt[visible_sample].std().clamp(min=1e-3) if visible_sample.any() else yt.std().clamp(min=1e-3)
            yt_norm = (yt - yt_mean) / yt_std
            loss_y_per = (out['y_pred'] - yt_norm) ** 2
            loss_y = (loss_y_per * query.float()).sum() / n_query
        elif y_kind == 'binary':
            yt = y.float()
            loss_y_per = F.binary_cross_entropy_with_logits(
                out['y_pred'], yt, reduction='none')
            loss_y = (loss_y_per * query.float()).sum() / n_query

    # ---- Effect loss (auxiliary) ----
    loss_effect = torch.tensor(0.0, device=device)
    if effect_weight > 0 and true_effects is not None:
        # true_effects: (P, T). Use L2 norm per taxon as target.
        target = torch.linalg.norm(true_effects, dim=0)           # (T,)
        target = (target - target.mean()) / (target.std() + 1e-3)
        loss_effect = F.mse_loss(out['effect_pred'], target)

    total = loss_count + y_weight * loss_y + effect_weight * loss_effect

    return dict(
        loss=total,
        loss_count=loss_count.detach(),
        loss_y=loss_y.detach(),
        loss_effect=loss_effect.detach(),
        n_masked=n_masked.detach(),
    )
