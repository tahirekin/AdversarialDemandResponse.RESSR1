"""Offline calibration for one evaluation summer, on the preceding summer.

Produces (cached in cache/calib_<E>.pkl):
  system-load model draws, portfolio-load model draws, mixture summary,
  membership probabilities, profiles, firm levels, forecast-error
  distribution and reserve, forecast weights c_{1,j}, diagnostics.
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

import data
import detection
import forecasting as fc
import generator as gen
import reliability as rel
from config import BASE, Config
from diagnostics import rhat_ess

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _summer_slice(index, year):
    return np.asarray((index >= pd.Timestamp(f"{year}-06-01 01:00")))


def steady_filter_means(params, series, D=None):
    """One-step predictive means (mixture over draws) for a fully observed
    series, steady-state gains. Returns (T,) with NaN for the first 50 h."""
    kf = fc.KF(params)
    P = kf.steady_state()
    K, Sv, _ = kf.gains(np.ones(1, bool), P)
    k = K[0]
    T = len(series)
    m = np.full(T, np.nan)
    a = fc.init_state(series[:fc.NLAG], kf.D)[:, :, 0]
    a = kf.predict_mean(a)
    for t in range(fc.NLAG, T):
        m[t] = a[:, 0].mean()
        a = a + k * (series[t] - a[:, 0])[:, None]
        a = kf.predict_mean(a)
    return m


CALIB_FIELDS = ("peak_mw", "n_part", "portfolio_share", "rho_g", "sigma_classes", "s1",
                "attacker_class", "structure_seed", "calib_seed", "firm_frac", "peak_hours",
                "lolp_target", "ev_price_pct", "ev_load_pct", "ev_lookback", "n_chains",
                "n_burn", "n_keep", "n_draws", "K", "sys_burn_mult", "sys_thin")


def calib_key(E, cfg):
    """Cache key: evaluation summer + settings that change the calibration.
    The base configuration keeps the plain name calib_<E>."""
    diff = [f"{k}={getattr(cfg, k)}" for k in CALIB_FIELDS if getattr(cfg, k) != getattr(BASE, k)]
    if not diff:
        return f"{E}"
    import hashlib
    return f"{E}_" + hashlib.md5(";".join(diff).encode()).hexdigest()[:8]


def calibrate(E: int, cfg: Config = BASE, force=False, verbose=True):
    path = os.path.join(CACHE, f"calib_{calib_key(E, cfg)}.pkl")
    if os.path.exists(path) and not force:
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    C = data.CALIB_OF[E]
    df = gen.system_load(C, cfg, scale_year=C)
    idx = df.index
    xsys = df["xsys"].to_numpy()
    price = df["price"].to_numpy()
    summ = _summer_slice(idx, C)
    storm = data.storm_mask(idx) if C == 2024 else np.zeros(len(idx), bool)
    hod = gen.hour_of_day(idx)

    # 1. system-load model on the calibration summer
    ys = xsys[summ]
    ms = ~storm[summ]
    sys_draws = fc.fit(ys, ms, cfg.n_chains, cfg.sys_burn_mult * cfg.n_burn, cfg.n_keep, seed=[E, 1],
                       thin=cfg.sys_thin)
    sys_p = fc.thin(sys_draws, cfg.n_draws)
    if verbose: print(f"[{E}] system model {time.time()-t0:.0f}s", flush=True)

    # 2. system-load forecast over the season, events
    msys = steady_filter_means(sys_p, xsys)
    ev = rel.events(price, xsys, msys, cfg, start=cfg.ev_lookback)
    ev &= ~storm

    # 3. participants in the calibration summer
    X = gen.consumption(xsys, cfg, seed=cfg.calib_seed + C)   # (T, N) uncurtailed
    use = summ & ~storm
    prof = gen.profiles(X[use], idx[use])                     # (24, N)
    F = gen.firm_levels(X[use], idx[use], cfg)
    XP = X.sum(1)

    # 4. portfolio-load model: event hours missing (curtailed)
    obs_mask = ~ev[summ] & ~storm[summ]
    port_draws = fc.fit(XP[summ], obs_mask, cfg.n_chains, cfg.n_burn, cfg.n_keep, seed=[E, 2])
    port_p = fc.thin(port_draws, cfg.n_draws)
    if verbose: print(f"[{E}] portfolio model {time.time()-t0:.0f}s", flush=True)

    # 5. mixture on profile-adjusted non-event readings
    sysprof = rel.hod_profile(xsys, hod, use)                  # (24,) system hour-of-day mean
    Yadj = X / gen.adjust(prof, sysprof, xsys, hod)
    Yfit = np.where((use & ~ev)[:, None], Yadj, np.nan)[summ]
    gmm_draws = detection.fit_gmm(Yfit, cfg.K, cfg.n_chains, cfg.n_burn, cfg.n_keep, seed=E)
    g = detection.gmm_summary(gmm_draws)
    r, _ = detection.membership_from_history(Yfit, g)
    if verbose: print(f"[{E}] mixture {time.time()-t0:.0f}s", flush=True)

    # 6. forecast error of the implied rest-of-system forecast
    kf = fc.KF(port_p)
    xs = XP[summ]
    P0 = kf.steady_state()
    K, Sv, _ = kf.gains(obs_mask, P0)
    a0 = kf.predict_mean(fc.init_state(XP[:np.flatnonzero(summ)[0]], kf.D))
    mX, _ = fc.filter_means(kf, K, a0, xs[:, None], obs_mask)
    mX = mX[:, :, 0].mean(1)
    e = (xsys[summ] - xs) - (msys[summ] - mX)
    e = e[~storm[summ]]
    edist = rel.ErrorDist(e)
    R = edist.reserve(cfg.lolp_target)

    # 7. forecast weights
    cw = kf.forecast_weights(400)
    ev_s = ev[summ]
    cw_ev, cw_ev_n = fc.event_weights(port_p, obs_mask, np.concatenate([ev_s[1:], [False]]),
                                      400, fallback=cw)

    diag = {"system": rhat_ess(sys_draws), "portfolio": rhat_ess(port_draws),
            "mixture": rhat_ess({f"{k}{j+1}": gmm_draws[k][..., j] for k in ("mu", "s2", "pi")
                                 for j in range(cfg.K)})}
    out = dict(key=calib_key(E, cfg), E=E, C=C, sysprof=sysprof, cw_ev=cw_ev, cw_ev_n=cw_ev_n, mean1=float(X[use, 0].mean()), sys_p=sys_p, port_p=port_p, gmm=g, r=r, prof=prof, F=F,
               edist=edist, R=R, cw=cw, diag=diag,
               ev_rate=float(ev[summ].mean()), e_sd=float(np.std(e)),
               classes_recovered=r.argmax(1), runtime=time.time() - t0)
    os.makedirs(CACHE, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"       # atomic: parallel jobs may build the same cache
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    if verbose: print(f"[{E}] done {time.time()-t0:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or data.EVAL_SUMMERS
    for E in years:
        calibrate(E, force=True)
