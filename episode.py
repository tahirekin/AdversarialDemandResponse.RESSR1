"""One evaluation episode: all five strategies on common random numbers.

Per interval tau (Section 3, Steps 1-5):
 1. defender commits d = (beta, delta) from information through tau-1;
 2. attacker perceives delta with noise, chooses attempt a, reports;
 3. detection, audit, imputation, settlement record;
 4. forecast update, committed supply / LOLP, settlement;
 5. belief updates (defender over types, attacker over delta).
Event hours: consumption curtailed to the firm level, no reading enters
the forecast or the baseline, no detection and no manipulation.
"""
from __future__ import annotations

import os
import pickle
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp

import attacker as att
import calibrate
import data
import defender as dfd
import detection
import forecasting as fc
import generator as gen
import reliability as rel
from config import BASE, DELTA_GRID, STRATEGIES, TYPES, Config

CACHE = calibrate.CACHE


# ----------------------------------------------------------------------
#  shared setup
# ----------------------------------------------------------------------
def episode_setup(E, start, cfg: Config, cal):
    """System-level quantities shared by all seeds, types and strategies."""
    key = (f"setup_{cal.get('key', str(E))}_{start}_W{cfg.W}_h{cfg.h}_J{cfg.J}"
           f"_w{cfg.warmup}_L{cfg.episode_len}.pkl")
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    df = gen.system_load(E, cfg, scale_year=cal["C"])
    idx = df.index
    xsys = df["xsys"].to_numpy(); price = df["price"].to_numpy()
    msys = calibrate.steady_filter_means(cal["sys_p"], xsys)
    ev = rel.events(price, xsys, msys, cfg, start=cfg.ev_lookback)
    s0 = int(np.flatnonzero(idx == pd.Timestamp(start) + pd.Timedelta(hours=1))[0])
    w0 = s0 - cfg.warmup
    t_end = s0 + cfg.episode_len
    hod = gen.hour_of_day(idx)
    expo = [rel.exposure_and_event_prob(price, ev, hod, t, cfg) for t in range(s0, t_end)]
    kf = fc.KF(cal["port_p"])
    storm = data.storm_mask(idx) if E == 2024 else np.zeros(len(idx), bool)
    mask = ~ev[w0:t_end] & ~storm[w0:t_end]
    K, Sv, _ = kf.gains(mask, kf.steady_state())
    out = dict(idx=idx, xsys=xsys, price=price, msys=msys, ev=ev, storm=storm, s0=s0, w0=w0, t_end=t_end,
               hod=hod, gbar=np.array([e[0] for e in expo]), pev=np.stack([e[1] for e in expo]),
               K=K, Sv=Sv)
    tmp = f"{path}.{os.getpid()}.tmp"       # atomic: parallel jobs may build the same cache
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, path)
    return out


def mixture_crps(y, mu, var):
    """CRPS of an equally weighted normal mixture."""
    sd = np.sqrt(var)
    z = (y - mu) / sd
    t1 = (2 * sd * norm.pdf(z) + (y - mu) * (2 * norm.cdf(z) - 1)).mean()
    dm = mu[:, None] - mu[None]
    s2 = np.sqrt(var[:, None] + var[None])
    z2 = dm / s2
    t2 = (2 * s2 * norm.pdf(z2) + dm * (2 * norm.cdf(z2) - 1)).mean()
    return t1 - 0.5 * t2


def mixture_quantiles(mu, var, ps=(0.05, 0.95)):
    sd = np.sqrt(var)
    out = []
    for p in ps:
        lo, hi = (mu - 6 * sd).min(), (mu + 6 * sd).max()
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if norm.cdf((mid - mu) / sd).mean() < p:
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out


# ----------------------------------------------------------------------
#  one strategy
# ----------------------------------------------------------------------
def run_strategy(strategy, true_type, cfg, cal, st, X, eta_att, eta_def, nu0=None,
                 trace=False, diag=None, refit=None, refit_every=24):
    """Simulate one episode for one strategy. Returns a list of row dicts."""
    g = cal["gmm"]
    prof = cal["prof"]; F = cal["F"]
    N = cfg.n_part
    kf = fc.KF(cal["port_p"])
    D = kf.D
    s0, w0, t_end = st["s0"], st["w0"], st["t_end"]
    ev, price, hod, idx = st["ev"], st["price"], st["hod"], st["idx"]
    K, Sv = st["K"], st["Sv"]
    edist, R = cal["edist"], cal["R"]
    cw = cal["cw_ev"]                        # weights at event hours (rho, bhat)
    mean1 = cal["mean1"]
    A, L = att.attack_grid(mean1, cfg)
    nA = len(A)
    k_true = TYPES.index(true_type)
    B_true = cfg.B_mult[k_true] * L
    ara = strategy in ("ara_static", "ara_dddas", "benchmark")
    if nu0 is None:
        nu0 = np.full(4, 0.25)
    if strategy == "benchmark":
        nu0 = np.eye(4)[k_true]
    nu = nu0.copy()
    B_types = np.asarray(cfg.B_mult) * L

    XP = X.sum(1)
    # filters: [corrected, truth, truth + imputation]; warm-up on true data
    assert cfg.h == 1, "the implementation assumes a commitment lead of h = 1"
    # state at w0-1 from the last 50 true values, predicted one step to w0
    a = kf.predict_mean(fc.init_state(XP[w0 - fc.NLAG:w0], D, 3))
    # membership log-likelihoods without the prior (pi is added when r is formed)
    llr = np.log(cal["r"] + 1e-300) - np.log(cal["gmm"]["pi"])
    obs_hist = np.zeros((t_end - w0, 3))     # filter inputs, for re-estimation checks
    adj_ep = {}                              # t -> corrected adjusted readings
    setrec, truerec = deque(maxlen=cfg.W), deque(maxlen=cfg.W)
    att_hist = deque(maxlen=cfg.W - 1)       # true attempts (non-event)
    est_hist = deque(maxlen=cfg.W - 1)       # defender's E_k[a] per type
    for t in range(w0, s0):                  # warm-up, truthful
        tt = t - w0
        if not ev[t] and not st["storm"][t]:
            a = a + K[tt][:, :, None] * (XP[t] - a[:, 0, :])[:, None, :]
            obs_hist[tt] = XP[t]
            setrec.append(X[t, 0]); truerec.append(X[t, 0])
            llr += detection.log_member_lik(X[t] / gen.adjust(prof, cal["sysprof"], st["xsys"][t], hod[t]), g)
        a = kf.predict_mean(a)
    omega = np.full(6, 1 / 6)                # attacker belief (post)
    Omega = np.full((cfg.n_traj, 6), 1 / 6)  # defender's simulated trajectories
    fb = {}                                  # t -> (u_1t, S_t)
    Dhat = {}                                # t -> defender's estimate of corrected - true
    rows = []
    for t in range(s0, t_end):
        tt = t - w0
        te = t - s0
        # --- predictions for hour t (made at t-1) -------------------------
        Mm = a[:, 0, :]                       # (D, 3)
        M, MX, Mi = Mm.mean(0)
        b_tot, b_iota = M - MX, Mi - MX
        lolp = dl = dla = 0.0
        if ev[t]:
            lolp = float(edist.sf(R - b_tot))
            dl = lolp - float(edist.sf(R))
            dla = lolp - float(edist.sf(R - b_iota))
        var = Sv[tt]
        row = dict(tau=te + 1, timestamp=idx[t], event=int(ev[t]), lolp=lolp, dlolp=dl,
                   dlolp_attack=dla, bias_total=b_tot, bias_attack=M - Mi,
                   forecast_mean=M, forecast_mean_x=MX, portfolio_true=XP[t])
        lo, hi = mixture_quantiles(Mm[:, 0], var)
        row.update(forecast_lo90=lo, forecast_hi90=hi,
                   crps=mixture_crps(XP[t], Mm[:, 0], var), crps_x=mixture_crps(XP[t], Mm[:, 1], var))
        fraud = 0.0; xi = 0; n_imp = 0; a_t = 0.0; flag1 = 0; d_idx = -1; beta = 0
        a_op = a_set = 0.0; dhat_v = np.nan; delta = np.nan
        if ev[t]:
            # settlement on the record, curtailed consumption; no reading
            PA = price[t] * max(0.0, np.mean(setrec) - F[0])
            P0 = price[t] * max(0.0, np.mean(truerec) - F[0])
            fraud = PA - P0
        else:
            r = np.exp((llr + np.log(g["pi"])) - logsumexp(llr + np.log(g["pi"]), 1, keepdims=True))
            tail = detection.TailProb(g, r)
            p_t = gen.adjust(prof, cal["sysprof"], st["xsys"][t], hod[t])      # detection (after reports)
            p_dec = gen.adjust(prof, cal["sysprof"], st["msys"][t], hod[t])    # decisions (forecast at tau-1)
            m_adj = tail.imputed()
            # ---- 1. defender commitment ----------------------------------
            if strategy == "no_defense":
                beta, delta = 0, 0.0
            elif strategy == "static_rule":
                beta, d_idx = 1, int(np.flatnonzero(DELTA_GRID == 0.05)[0]); delta = 0.05
            if ara:
                gbar = st["gbar"][te]; pev = st["pev"][te]
                bhat = np.zeros(cfg.J + 1)
                for s_, v in Dhat.items():
                    lag = t - s_ + np.arange(cfg.J + 1) - 1 + cfg.h
                    ok = lag < len(cw)
                    bhat[ok] += cw[lag[ok]] * v
                est = np.array(est_hist).sum(0) if est_hist else np.zeros(4)
                ctx = dfd.Context(cfg=cfg, A=A, tail=tail, prof1=p_dec[0], Omega=Omega,
                                  eta_def=eta_def[te], gbar=gbar, pev=pev, bhat=bhat,
                                  cw=cw[:cfg.J + 1], R=R, edist=edist, m1_adj=m_adj[0],
                                  rem=B_types - est)
                beta, d_idx, psi = dfd.decide(ctx, nu)
                delta = DELTA_GRID[d_idx]
            # ---- 2. attacker ---------------------------------------------
            ld = np.log(delta) if delta > 0 else np.log(DELTA_GRID[0])
            omega = att.perceive(omega, ld, eta_att[te], cfg.sigma_perc, cfg.lam)
            x1 = X[t, 0]
            if k_true > 0:
                u_c = tail.u(0, (x1 + A) / p_dec[0])
                pi = att.flag_prob(omega, (u_c[:, None] < DELTA_GRID))
                feas = A <= B_true - sum(att_hist) + 1e-9
                ia = att.best_response(pi, A, st["gbar"][te], cfg.C_A[k_true],
                                       cfg.Pi_pen[k_true], cfg.q, feas)
                a_t = float(A[ia])
            if diag is not None and ara:
                # defender's predicted attack distribution at the TRUE
                # consumption (check only; never used by the defender)
                u_x = tail.u(0, (x1 + A[None, :]) / p_dec[0])
                Pk = dfd.type_response(ctx, dfd.perceived(ctx, np.array([delta])), u_x, ctx.rem)[:, 0, 0]
                diag.append(dict(t=t, a=a_t, p_hit=float(Pk[k_true][np.argmin(np.abs(A - a_t))]),
                                 a_pred_mean=float(Pk[k_true] @ A), rem_est=float(ctx.rem[k_true]),
                                 rem_true=float(B_true - sum(att_hist)), delta=delta))
            Y = X[t].copy(); Y[0] += a_t
            # ---- 3. detection, audit, imputation -------------------------
            yadj = Y / p_t
            u = tail.u_all(yadj)
            flags = u < delta
            flag1 = int(flags[0]); S = 1 - flag1
            xi = int(beta * flag1)
            Yt = Y.copy()
            imp = flags.copy(); imp[0] = flags[0] and not xi
            Yt[imp] = m_adj[imp] * p_t[imp]
            if xi:
                Yt[0] = x1
            Yset = x1 if xi else Y[0]
            n_imp = int(imp.sum())
            a_op, a_set = a_t * S, a_t * (1 - xi)
            # ---- 4. forecast update --------------------------------------
            iota = np.where(imp, m_adj * p_t - X[t], 0.0).sum()
            obs = np.array([Yt.sum(), XP[t], XP[t] + iota])
            a = a + K[tt][:, :, None] * (obs - a[:, 0, :])[:, None, :]
            obs_hist[tt] = obs; adj_ep[t] = Yt / p_t
            llr += detection.log_member_lik(Yt / p_t, g)
            setrec.append(Yset); truerec.append(x1); att_hist.append(a_t)
            fb[t] = (float(u[0]), S)
            # ---- 5. defender belief update -------------------------------
            if ara:
                ctx.prof1 = p_t[0]               # after the interval: actual telemetry
                lik, Ea_k, post = dfd.observe(ctx, nu, d_idx, Y[0], S, xi, x1 if xi else None)
                nu_post = nu * lik; nu_post /= nu_post.sum()
                pa = (nu_post[:, None] * post).sum(0)
                Ea = float(pa @ A)
                est_hist.append(Ea_k)
                others = np.where(imp[1:], m_adj[1:] * p_t[1:] - Y[1:], 0.0).sum()
                if xi:
                    d1 = 0.0
                elif flag1:
                    d1 = m_adj[0] * p_t[0] - (Y[0] - Ea)
                else:
                    d1 = Ea
                Dhat[t] = d1 + others
                dhat_v = d1
                if strategy == "ara_dddas":
                    nu = nu_post
                Omega = dfd.perceived(ctx, np.array([delta]))[0]
        # ---- end of interval: delayed feedback ---------------------------
        tf = t - cfg.kappa
        if tf in fb:
            uf, Sf = fb[tf]
            omega = att.feedback(omega, uf, Sf, cfg.eps_A)
            if ara:
                Omega = att.feedback(Omega, uf, Sf, cfg.eps_A)
        a = kf.predict_mean(a)
        if refit is not None and (te + 1) % refit_every == 0 and t + 1 < t_end:
            # long-chain re-estimation check: refit both models on the data
            # available at the end of interval t, re-filter, continue
            port_p, g, llr_new = refit(t, obs_hist[:tt + 1, 0], adj_ep)
            if llr_new is not None:
                llr = llr_new
            kf = fc.KF(port_p)
            K, Sv, _ = kf.gains(~ev[w0:t_end] & ~st["storm"][w0:t_end], kf.steady_state())
            a0 = kf.predict_mean(fc.init_state(XP[w0 - fc.NLAG:w0], kf.D, 3))
            _, a = fc.filter_means(kf, K[:tt + 1], a0, obs_hist[:tt + 1],
                                   ~ev[w0:t + 1] & ~st["storm"][w0:t + 1])
        corr = cfg.C_F * n_imp
        row.update(strategy=strategy, true_type=true_type, delta=delta, beta=beta,
                   a_attempt=a_t, flagged=flag1, audited=xi, a_eff_op=a_op, a_eff_set=a_set,
                   n_imputed=n_imp, fraud=fraud, audit_cost=cfg.C_D * xi, correction_cost=corr,
                   utility=-(cfg.w1 * dl + cfg.w2 * fraud + cfg.C_D * xi + corr),
                   nu_honest=nu[0], nu_1=nu[1], nu_2=nu[2], nu_3=nu[3],
                   delta_hat=float(omega @ DELTA_GRID), dhat1=dhat_v)
        rows.append(row)
    return rows


def run_episode(E, start, seed, true_type, cfg: Config = BASE, strategies=STRATEGIES,
                episode_id=None):
    cal = calibrate.calibrate(E, cfg)
    st = episode_setup(E, start, cfg, cal)
    Xall = gen.consumption(st["xsys"], cfg, seed=1000 * E + seed)
    eta_att = np.random.default_rng([E, seed, int(pd.Timestamp(start).dayofyear), 2]).standard_normal(cfg.episode_len) * (
        cfg.sigma_perc if np.isfinite(cfg.sigma_perc) else 0.0)
    eta_def = np.random.default_rng([E, seed, int(pd.Timestamp(start).dayofyear), 3]).standard_normal(
        (cfg.episode_len, cfg.n_traj)) * (cfg.sigma_perc if np.isfinite(cfg.sigma_perc) else 0.0)
    rows = []
    for s in strategies:
        r = run_strategy(s, true_type, cfg, cal, st, Xall, eta_att, eta_def)
        for x in r:
            x.update(summer=E, episode_id=episode_id or start, seed=seed)
        rows += r
    return pd.DataFrame(rows)
