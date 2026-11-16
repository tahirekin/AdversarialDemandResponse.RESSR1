"""Synthetic participant-level consumption, eq:generator.

X_i = share * s_i * X_sys * exp(eps_i - sigma_i^2 / 2), eps_i AR(1) with
stationary s.d. sigma_i and autocorrelation rho_g.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import data
from config import Config


def structure(cfg: Config):
    """Shares s_i (sum 1) and variability class of each participant.
    Participant index 0 is the potentially strategic participant 1."""
    rng = np.random.default_rng(cfg.structure_seed)
    n = cfg.n_part
    others = rng.uniform(0.5, 1.5, n - 1)
    shares = np.concatenate([[cfg.s1], (1 - cfg.s1) * others / others.sum()])
    cls = np.concatenate([[cfg.attacker_class], rng.permutation(np.arange(n - 1) % 3)])
    sig = np.asarray(cfg.sigma_classes)[cls]
    return shares, cls, sig


def system_load(year: int, cfg: Config, scale_year: int) -> pd.DataFrame:
    """Season (Apr-Aug) of rescaled system load and prices. The scale is set
    so that the peak of the calibration summer `scale_year` equals
    cfg.peak_mw; it is applied to the calibration and the evaluation summer."""
    scale = cfg.peak_mw / data.summer(scale_year)["load"].max()
    df = data.season(year).copy()
    df["xsys"] = df["load"] * scale
    return df


def consumption(xsys: np.ndarray, cfg: Config, seed: int) -> np.ndarray:
    """(T, N) true consumption for a system-load path (no curtailment)."""
    shares, _, sig = structure(cfg)
    rng = np.random.default_rng(seed)
    T, n = len(xsys), cfg.n_part
    rho = cfg.rho_g
    zeta = rng.standard_normal((T, n)) * sig * np.sqrt(1 - rho ** 2)
    eps = np.empty((T, n))
    eps[0] = rng.standard_normal(n) * sig
    for t in range(1, T):
        eps[t] = rho * eps[t - 1] + zeta[t]
    return cfg.portfolio_share * shares * xsys[:, None] * np.exp(eps - 0.5 * sig ** 2)


def profiles(X: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    """(24, N) mean consumption by hour of day (hour-ending 1..24 -> 0..23)."""
    hod = hour_of_day(index)
    return np.stack([X[hod == k].mean(0) for k in range(24)])


def hour_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """0..23 for hour ending 1..24."""
    return np.asarray(((index.hour - 1) % 24))


def firm_levels(X: np.ndarray, index: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    he = hour_of_day(index) + 1
    m = np.isin(he, cfg.peak_hours)
    return cfg.firm_frac * X[m].mean(0)


def adjust(prof, sysprof, xsys, hod):
    """Adjustment factor for detection: the participant's hour-of-day mean
    scaled by the current system load relative to its own hour-of-day mean
    in the calibration summer. The reading divided by this factor removes
    both the daily shape and the system-wide level of the hour, which is
    measured by utility telemetry and not by participant reports.
    Works for a single hour (hod scalar -> (N,)) or many ((T,) -> (T, N))."""
    hod = np.asarray(hod)
    r = np.asarray(xsys) / sysprof[hod]
    return prof[hod] * (r[..., None] if np.ndim(r) else r)
