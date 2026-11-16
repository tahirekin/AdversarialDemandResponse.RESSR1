"""Long-chain re-estimation check (Section 7.2, response to R2.8).

For a subset of episodes, both statistical models are re-estimated with
long chains (4 x (2,000 + 2,000), as in the calibration) every 24 hours
during the episode, on all data available at that time: the calibration
summer and the evaluation summer up to the current hour (truthful before
the episode, corrected readings during it; event hours missing). The
filter state is rebuilt with the new draws and the episode continues.
The results are compared with the fixed-draw runs of version B.

  python3 tests/longchain_check.py          # run (about 40 min on 8 cores)
  python3 tests/longchain_check.py report   # summary only
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import calibrate  # noqa: E402
import data  # noqa: E402
import detection  # noqa: E402
import episode  # noqa: E402
import forecasting as fc  # noqa: E402
import generator as gen  # noqa: E402
import reliability as rel  # noqa: E402
from config import BASE, episode_starts  # noqa: E402

OUT = os.path.join(HERE, "results", "longchain")
STRATS = ["no_defense", "ara_static", "ara_dddas"]
EPS = [(2017, episode_starts(2017)[1]), (2017, episode_starts(2017)[4]),
       (2023, episode_starts(2023)[1]), (2023, episode_starts(2023)[4])]
GAP = 48   # missing hours inserted between calibration and evaluation summer


def calib_data(E, cfg):
    """Calibration-summer portfolio series and adjusted readings, as in
    calibrate.calibrate (same seeds)."""
    cal = calibrate.calibrate(E, cfg)
    C = cal["C"]
    df = gen.system_load(C, cfg, scale_year=C)
    idx = df.index
    xsys = df["xsys"].to_numpy(); price = df["price"].to_numpy()
    summ = np.asarray(idx >= pd.Timestamp(f"{C}-06-01 01:00"))
    storm = data.storm_mask(idx) if C == 2024 else np.zeros(len(idx), bool)
    msys = calibrate.steady_filter_means(cal["sys_p"], xsys)
    ev = rel.events(price, xsys, msys, cfg, start=cfg.ev_lookback) & ~storm
    X = gen.consumption(xsys, cfg, seed=cfg.calib_seed + C)
    hod = gen.hour_of_day(idx)
    adj = X / gen.adjust(cal["prof"], cal["sysprof"], xsys, hod)
    ok = summ & ~storm & ~ev
    return (X.sum(1)[summ], ok[summ], np.where(ok[:, None], adj, np.nan)[summ])


def make_refit(E, start, seed, cfg, cal, st, X, strategy):
    cP, cM, cA = calib_data(E, cfg)
    idx = st["idx"]
    j0 = int(np.flatnonzero(idx == pd.Timestamp(f"{E}-06-01 01:00"))[0])
    XP = X.sum(1)
    hod = gen.hour_of_day(idx)
    adj_true = X / gen.adjust(cal["prof"], cal["sysprof"], st["xsys"], hod)
    w0, s0, ev = st["w0"], st["s0"], st["ev"]
    count = [0]

    def refit(t, obs_corr, adj_ep):
        count[0] += 1
        # portfolio series: calibration summer | gap | evaluation summer to t
        ev_part = XP[j0:t + 1].copy()
        ev_part[w0 - j0:] = obs_corr                        # warm-up truthful, episode corrected
        mE = ~ev[j0:t + 1]
        obs = np.concatenate([cP, np.zeros(GAP), ev_part])
        mask = np.concatenate([cM, np.zeros(GAP, bool), mE])
        dr = fc.fit(obs, mask, cfg.n_chains, cfg.n_burn, cfg.n_keep, seed=[E, seed, count[0], 7])
        port_p = fc.thin(dr, cfg.n_draws)
        # mixture: calibration readings + evaluation summer to t
        A = adj_true[j0:t + 1].copy()
        A[~mE] = np.nan
        for tt, v in adj_ep.items():
            A[tt - j0] = v
        Yfit = np.vstack([cA, A])
        gd = detection.fit_gmm(Yfit, cfg.K, cfg.n_chains, cfg.n_burn, cfg.n_keep, seed=[E, seed, count[0]])
        g = detection.gmm_summary(gd)
        # membership log-likelihood from the episode-time data only (warm-up
        # onward), as in the fixed-draw run, plus the calibration readings
        _, ll = detection.membership_from_history(Yfit, g)
        llr = ll - np.log(g["pi"])
        return port_p, g, llr

    return refit


def job(args):
    E, start, strategy = args
    path = os.path.join(OUT, f"{E}_{start}_{strategy}.csv.gz")
    if os.path.exists(path):
        return path
    cfg = BASE
    seed = 1
    cal = calibrate.calibrate(E, cfg)
    st = episode.episode_setup(E, start, cfg, cal)
    X = gen.consumption(st["xsys"], cfg, seed=1000 * E + seed)
    doy = int(pd.Timestamp(start).dayofyear)
    s = cfg.sigma_perc
    ea = np.random.default_rng([E, seed, doy, 2]).standard_normal(cfg.episode_len) * s
    ed = np.random.default_rng([E, seed, doy, 3]).standard_normal((cfg.episode_len, cfg.n_traj)) * s
    refit = make_refit(E, start, seed, cfg, cal, st, X, strategy)
    rows = episode.run_strategy(strategy, "aggressive", cfg, cal, st, X, ea, ed, refit=refit)
    df = pd.DataFrame(rows); df["summer"] = E; df["episode_id"] = f"{E}_{start}"; df["seed"] = seed
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")
    return path


def report():
    lines = []
    rows = []
    for E, start in EPS:
        fx = pd.read_csv(os.path.join(HERE, "results", "B", "base", f"{E}_{start}_s1_aggressive.csv.gz"))
        for s in STRATS:
            p = os.path.join(OUT, f"{E}_{start}_{s}.csv.gz")
            if not os.path.exists(p):
                continue
            lc = pd.read_csv(p)
            f = fx[fx.strategy == s].reset_index(drop=True)
            ne = (f.event == 0).to_numpy()
            same = ((f.delta.fillna(-1) == lc.delta.fillna(-1)) & (f.beta == lc.beta)).to_numpy()[ne].mean()
            rows.append(dict(episode=f"{E}_{start}", strategy=s,
                             dLOLE_fixed=f.dlolp.sum(), dLOLE_long=lc.dlolp.sum(),
                             fraud_fixed=f.fraud.sum(), fraud_long=lc.fraud.sum(),
                             cost_fixed=-f.utility.sum(), cost_long=-lc.utility.sum(),
                             same_decision=same,
                             fc_mean_absdiff=(f.forecast_mean - lc.forecast_mean).abs().mean(),
                             crps_fixed=f.crps.mean(), crps_long=lc.crps.mean(),
                             nu3_end_fixed=f.nu_3.iloc[-1], nu3_end_long=lc.nu_3.iloc[-1]))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    lines.append(t.round(4).to_string(index=False))
    if len(t):
        d = t.cost_long - t.cost_fixed
        lines.append(f"\ntotal cost, long - fixed: mean {d.mean():.1f} $, max |diff| {d.abs().max():.1f} $; "
                     f"relative to cost: {np.mean(np.abs(d) / t.cost_fixed.abs().clip(lower=1)):.3f}")
        lines.append(f"decisions identical in {t.same_decision.mean():.3f} of non-event intervals; "
                     f"mean |forecast difference| {t.fc_mean_absdiff.mean():.3f} MW")
        for s in ("ara_dddas",):
            for a in (("ara_static",),):
                pass
        piv = t.pivot(index="episode", columns="strategy", values=["cost_fixed", "cost_long"])
        if {"ara_dddas", "ara_static"} <= set(t.strategy):
            df_ = piv["cost_fixed"]["ara_dddas"] - piv["cost_fixed"]["ara_static"]
            dl_ = piv["cost_long"]["ara_dddas"] - piv["cost_long"]["ara_static"]
            lines.append("ARA-DDDAS - ARA-static per episode, fixed: " + ", ".join(f"{v:.0f}" for v in df_)
                         + " | long: " + ", ".join(f"{v:.0f}" for v in dl_))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "logs", "longchain.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        jobs = [(E, s, st) for (E, s) in EPS for st in STRATS]
        with Pool(min(8, len(jobs))) as p:
            for path in p.imap_unordered(job, jobs):
                print("done", path, flush=True)
        report()
