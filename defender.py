"""Defender: ARA decision (eq:ara_type_response - eq:defender_optimal) and
type-belief update (eq:weight_update, eq:type_likelihood).

Everything is vectorized over
  decision delta (6) x trajectory (R) x quadrature node (Q) x attack (nA) x
  grid level j (6),
and over the audit choice beta (2) when utilities are evaluated.
"""
from __future__ import annotations

import numpy as np

import attacker as att
import reliability as rel
from config import DELTA_GRID

NT = 4  # types: honest, cautious, moderate, aggressive


class Context:
    """What the defender knows at the start of interval tau."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def type_response(ctx, omega_d, u_xa, rem):
    """p_D(a | delta_d, type k, x) for every candidate delta.

    omega_d: (6d, R, 6j) perceived beliefs per candidate delta.
    u_xa: (X, nA) tail probabilities of report x + a, for X values of x.
    rem: (NT,) remaining budget per type (MWh).
    Returns P (NT, 6d, X, nA): probability of each attack level."""
    cfg, A = ctx.cfg, ctx.A
    flagged = u_xa[..., None] < DELTA_GRID                      # (X, nA, 6j)
    pi = np.einsum("drj,xaj->drxa", omega_d, flagged.astype(float))
    nd, R, X, nA = pi.shape
    P = np.zeros((NT, nd, X, nA))
    P[0, :, :, 0] = 1.0
    for k in range(1, NT):
        feas = A <= rem[k] + 1e-9
        idx = att.best_response(pi, A, ctx.gbar, cfg.C_A[k], cfg.Pi_pen[k], cfg.q, feas)
        oh = np.zeros((nd, R, X, nA))
        np.put_along_axis(oh, idx[..., None], 1.0, -1)
        P[k] = oh.mean(1)
    return P


def perceived(ctx, deltas):
    """Simulated attacker beliefs after perceiving each delta in `deltas`."""
    cfg = ctx.cfg
    return att.perceive(ctx.Omega[None], np.log(deltas)[:, None], ctx.eta_def[None],
                        cfg.sigma_perc, cfg.lam)


def decide(ctx, nu):
    """Return (beta, delta_index, psi table (2, 6))."""
    cfg, A = ctx.cfg, ctx.A
    tp = ctx.tail
    xn, wq = tp.quadrature(0, cfg.n_quad)                       # adjusted scale
    u_xa = tp.u(0, xn[:, None] + A[None, :] / ctx.prof1)        # (Q, nA)
    omega_d = perceived(ctx, DELTA_GRID)                        # (6, R, 6)
    P = type_response(ctx, omega_d, u_xa, ctx.rem)              # (NT, 6, Q, nA)
    Pm = np.einsum("k,kdqa->dqa", nu, P)                        # mixture over types
    s = (u_xa[None] >= DELTA_GRID[:, None, None]).astype(float)  # (6, Q, nA)
    m1 = ctx.m1_adj
    imp_err = (m1 - xn) * ctx.prof1                              # raw MWh (Q,)
    beta = np.array([0.0, 1.0])[:, None, None, None]
    D = A * s[None] + (1 - beta) * (1 - s[None]) * imp_err[None, None, :, None]   # (2,6,Q,nA)
    rho = rel.rho(D, ctx.bhat, ctx.pev, ctx.cw, ctx.R, ctx.edist)
    flag = 1 - s[None]
    U = -(cfg.w1 * rho
          + cfg.w2 * ctx.gbar * A * (1 - beta * flag)
          + cfg.C_D * beta * flag
          + cfg.C_F * ((1 - beta) * flag + (cfg.n_part - 1) * DELTA_GRID[None, :, None, None]))
    psi = np.einsum("dqa,bdqa,q->bd", Pm, U, wq)
    b, d = np.unravel_index(np.argmax(psi), psi.shape)          # ties: beta 0, small delta
    return int(b), int(d), psi


def observe(ctx, nu, d_idx, Y1, S, xi, X1_if_audited):
    """Likelihood of the outcome under each type (eq:type_likelihood) and the
    posterior over the attempted manipulation.

    Returns lik (NT,), Ea_k (NT,) expected attempt per type, post_a (NT, nA)."""
    cfg, A = ctx.cfg, ctx.A
    tp = ctx.tail
    nA = len(A)
    delta = DELTA_GRID[d_idx]
    omega = perceived(ctx, np.array([delta]))                    # (1, R, 6)
    if xi:
        x_adj = np.array([X1_if_audited / ctx.prof1])
    else:
        x_adj = (Y1 - A) / ctx.prof1                              # candidate true values
    u_xa = tp.u(0, x_adj[:, None] + A[None, :] / ctx.prof1)     # (X, nA)
    P = type_response(ctx, omega, u_xa, ctx.rem)[:, 0]          # (NT, X, nA)
    P = (1 - cfg.eps) * P + cfg.eps / nA
    if xi:
        a_obs = int(np.argmin(np.abs(A - (Y1 - X1_if_audited))))
        lik = P[:, 0, a_obs]
        post = np.zeros((NT, nA)); post[:, a_obs] = 1.0
    else:
        f = tp.density(0, x_adj)                                 # (nA,)
        w = P[:, np.arange(nA), np.arange(nA)] * f[None]          # p(a | x = Y - a) f(Y - a)
        lik = w.sum(1)
        post = w / np.maximum(lik[:, None], 1e-300)
    Ea_k = post @ A
    return lik, Ea_k, post
