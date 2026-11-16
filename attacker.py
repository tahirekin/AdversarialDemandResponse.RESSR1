"""Attacker belief and decision (Sections 5.1, 6.1, 6.2).

These functions serve both the true attacker and the defender's model of
him (the defender's simulated belief trajectories): arrays broadcast over
leading axes.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from config import DELTA_GRID

LOG_GRID = np.log(DELTA_GRID)


def perceive(omega_post, log_delta, eta, sigma, lam):
    """eq:attacker_perception. omega_post (..., J); log_delta and eta
    broadcast against the leading axes. sigma = inf: feedback only."""
    lw = lam * np.log(np.maximum(omega_post, 1e-300))
    if np.isfinite(sigma):
        s = np.asarray(log_delta)[..., None] + np.asarray(eta)[..., None]
        lw = lw - 0.5 * ((s - LOG_GRID) / sigma) ** 2
    return np.exp(lw - logsumexp(lw, -1, keepdims=True))


def feedback(omega, u_t, S_t, eps_A):
    """eq:attacker_feedback for the report of interval t."""
    consistent = (u_t < DELTA_GRID) == (1 - S_t)
    lik = np.where(consistent, 1 - eps_A, eps_A)
    w = omega * lik
    return w / w.sum(-1, keepdims=True)


def flag_prob(omega, flagged_j):
    """eq:attacker_flag_prob. omega (..., J), flagged_j (..., nA, J) bool
    -> (..., nA)."""
    return np.einsum("...j,...aj->...a", omega, flagged_j.astype(float))


def best_response(pi, A, gbar, C_A, Pi_pen, q, feasible):
    """eq:attacker_eu / eq:attacker_problem. pi (..., nA) flag probability,
    A (nA,) attack levels in MWh, feasible (..., nA) or (nA,) bool.
    Returns the index of the optimal level (ties -> smaller attack)."""
    psi = (1 - q * pi) * gbar * A - C_A * A - q * pi * Pi_pen * (A > 0)
    psi = np.where(feasible, psi, -np.inf)
    return psi.argmax(-1)


def attack_grid(mean_consumption, cfg):
    """eq:attack_grid: fixed step g, levels up to L (MWh)."""
    g = cfg.g_frac * mean_consumption
    L = cfg.L_frac * mean_consumption
    n = int(np.floor(L / g + 1e-9))
    return g * np.arange(n + 1), L
