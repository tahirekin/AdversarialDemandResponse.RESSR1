"""Tables and figures for Section 7.4, written from the result CSVs only.

  python3 tables.py B
Writes tables/<version>/*.csv and *.tex and figures to figures/<version>/.
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

import calibrate
import data
from config import BASE, DELTA_GRID, STRATEGIES, TYPES

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = {"no_defense": "No defense", "static_rule": "Static rule", "ara_static": "ARA-static",
         "ara_dddas": "ARA-DDDAS", "benchmark": "Complete information"}
METRICS = ["dLOLE", "dLOLE_attack", "fraud", "audit_cost", "correction_cost", "total_cost"]


def load(version, setting=None):
    pat = os.path.join(HERE, "results", version, setting or "*", "*.csv.gz")
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f"no results under {pat}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def episode_totals(df):
    keys = ["setting", "summer", "episode_id", "seed", "true_type", "strategy"]
    g = df.groupby(keys)
    t = pd.DataFrame({
        "dLOLE": g.dlolp.sum(), "dLOLE_attack": g.dlolp_attack.sum(), "fraud": g.fraud.sum(),
        "audit_cost": g.audit_cost.sum(), "correction_cost": g.correction_cost.sum(),
        "total_cost": -g.utility.sum(), "n_audits": g.audited.sum(), "attempts": g.a_attempt.sum(),
    }).reset_index()
    return t


def strategy_table(tot, by_summer=False):
    keys = ["true_type", "strategy"] + (["summer"] if by_summer else [])
    t = tot.groupby(keys)[METRICS].mean().reset_index()
    t["order"] = t.strategy.map(STRATEGIES.index)
    t["torder"] = t.true_type.map(TYPES.index)
    return t.sort_values(["torder"] + (["summer"] if by_summer else []) + ["order"]).drop(columns=["order", "torder"])


def paired_bootstrap(tot, a, b, metric="total_cost", n_boot=2000, seed=0):
    """Mean of (a - b) per replicate, CI by resampling whole episodes."""
    w = tot.pivot_table(index=["true_type", "summer", "episode_id", "seed"], columns="strategy",
                        values=metric).reset_index()
    w["diff"] = w[a] - w[b]
    out = []
    rng = np.random.default_rng(seed)
    for tt, d in w.groupby("true_type"):
        ep = d.groupby(["summer", "episode_id"])["diff"].mean().to_numpy()
        boots = rng.choice(ep, (n_boot, len(ep)), replace=True).mean(1)
        out.append(dict(true_type=tt, comparison=f"{LABEL[a]} - {LABEL[b]}", metric=metric,
                        mean=ep.mean(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                        n_episodes=len(ep)))
    return out


def paired_bootstrap_summers(tot, a, b, metric="total_cost", n_boot=2000, seed=0):
    """Two-level bootstrap: resample summers, then episodes within each
    drawn summer (more conservative; episodes of a summer share a
    calibration and the consumption path of a seed)."""
    w = tot.pivot_table(index=["true_type", "summer", "episode_id", "seed"], columns="strategy",
                        values=metric).reset_index()
    w["diff"] = w[a] - w[b]
    out = []
    rng = np.random.default_rng(seed)
    for tt, d in w.groupby("true_type"):
        ep = d.groupby(["summer", "episode_id"])["diff"].mean()
        groups = [g.to_numpy() for _, g in ep.groupby(level=0)]
        boots = np.empty(n_boot)
        for i in range(n_boot):
            gs = rng.integers(0, len(groups), len(groups))
            vals = np.concatenate([rng.choice(groups[g], len(groups[g]), replace=True) for g in gs])
            boots[i] = vals.mean()
        out.append(dict(true_type=tt, comparison=f"{LABEL[a]} - {LABEL[b]}", metric=metric,
                        mean=ep.mean(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                        n_summers=len(groups)))
    return out


def forecast_accuracy(df):
    rows = []
    d = df.copy()
    d["ape"] = (d.forecast_mean - d.portfolio_true).abs() / d.portfolio_true
    d["cover"] = (d.portfolio_true >= d.forecast_lo90) & (d.portfolio_true <= d.forecast_hi90)
    cases = [("Truthful data", (d.true_type == "honest") & (d.strategy == "no_defense"))]
    cases.append(("Manipulated, no defense", (d.true_type == "aggressive") & (d.strategy == "no_defense")))
    for s in STRATEGIES[1:]:
        cases.append((f"Manipulated, {LABEL[s]}", (d.true_type == "aggressive") & (d.strategy == s)))
    for name, m in cases:
        x = d[m]
        rows.append(dict(case=name, MAPE=100 * x.ape.mean(), CRPS=x.crps.mean(), coverage90=100 * x.cover.mean()))
    return pd.DataFrame(rows)


def posterior_true_type(df):
    d = df[df.strategy == "ara_dddas"].copy()
    col = {"honest": "nu_honest", "cautious": "nu_1", "moderate": "nu_2", "aggressive": "nu_3"}
    d["nu_true"] = [r[col[t]] for r, t in zip(d[list(col.values())].to_dict("records"), d.true_type)]
    q = d.groupby(["true_type", "tau"]).nu_true.agg(mean="mean", p10=lambda x: x.quantile(0.1),
                                                     p90=lambda x: x.quantile(0.9)).reset_index()
    return q


def traced(df, summer=2017, seed=1, true_type="aggressive", n=40):
    ep = sorted(df[df.summer == summer].episode_id.unique())[0]
    d = df[(df.summer == summer) & (df.episode_id == ep) & (df.seed == seed) &
           (df.true_type == true_type) & (df.strategy == "ara_dddas")].sort_values("tau")
    prev = np.array([0.25] * 4)
    rows = []
    for _, r in d.iterrows():
        after = np.array([r.nu_honest, r.nu_1, r.nu_2, r.nu_3])
        rows.append(dict(tau=r.tau, event=r.event, nu3_before=prev[3], nu0_before=prev[0],
                         delta=r.delta, beta=r.beta, attempt=r.a_attempt, flagged=r.flagged,
                         audited=r.audited, nu3_after=after[3], nu0_after=after[0]))
        prev = after
    return pd.DataFrame(rows).head(n)


def diagnostics():
    rows = []
    for E in data.EVAL_SUMMERS:
        path = os.path.join(calibrate.CACHE, f"calib_{E}.pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            cal = pickle.load(f)
        for model, dd in cal["diag"].items():
            rh = max(v[0] for v in dd.values()); es = min(v[1] for v in dd.values())
            rows.append(dict(summer=E, calibration=cal["C"], model=model, max_rhat=rh, min_ess=es))
        rows.append(dict(summer=E, calibration=cal["C"], model=f"e: {cal['edist'].family}",
                         max_rhat=np.nan, min_ess=np.nan))
    return pd.DataFrame(rows)


def write(df, name, outdir, floatfmt="%.4f"):
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, name + ".csv"), index=False)
    with open(os.path.join(outdir, name + ".tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format=lambda x: floatfmt % x, na_rep="--"))


def main(version="B"):
    outdir = os.path.join(HERE, "tables", version)
    df = load(version)
    base = df[df.setting == "base"]
    tot = episode_totals(df)
    tb = tot[tot.setting == "base"]
    tb2 = tb.assign(strategy_label=tb.strategy.map(LABEL))
    write(strategy_table(tb), "strategy_pooled", outdir)
    write(strategy_table(tb, by_summer=True), "strategy_by_summer", outdir)
    pairs = [("ara_dddas", "ara_static"), ("ara_dddas", "static_rule"), ("ara_dddas", "no_defense"),
             ("ara_static", "static_rule"), ("benchmark", "ara_dddas")]
    pb = []
    for a, b in pairs:
        for m in METRICS:
            pb += paired_bootstrap(tb, a, b, m)
    write(pd.DataFrame(pb), "paired_differences", outdir)
    write(forecast_accuracy(base), "forecast_accuracy", outdir)
    write(posterior_true_type(base), "posterior_true_type", outdir)
    write(traced(base), "traced_episode", outdir, "%.3f")
    write(diagnostics(), "convergence", outdir, "%.3f")
    sens = tot[(tot.true_type == "aggressive") & tot.summer.isin([2017, 2023])]
    write(sens.groupby(["setting", "strategy"])[METRICS].mean().reset_index(), "sensitivity", outdir)
    tb2.to_csv(os.path.join(outdir, "episode_totals.csv"), index=False)
    _figure(posterior_true_type(base), os.path.join(HERE, "figures", version))
    print(f"tables written to {outdir}")


def _figure(q, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for tt, d in q.groupby("true_type"):
        ax.plot(d.tau, d["mean"], label=tt)
        ax.fill_between(d.tau, d.p10, d.p90, alpha=0.15)
    ax.axhline(0.25, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("interval (hour of episode)"); ax.set_ylabel("posterior probability of true type")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_posterior_true_type.png"), dpi=200)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "B")
