"""Design checks before the version B batch (writes logs/design_checks.txt).

 1. bias decomposition: filtered b^a vs sum_j c_{1,j} a_op (eq:bias_weights)
 2. reserve transfer: P(e > R) on the evaluation summers, truthful data
 3. sign of b^a under no defense, all 30 episodes
 4. defender's attack prediction with the type known (benchmark), and the
    fraud identity Pi_fraud = p ev mean(a_set) when both baselines > F
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import calibrate  # noqa: E402
import data  # noqa: E402
import episode  # noqa: E402
import forecasting as fc  # noqa: E402
import generator as gen  # noqa: E402
from config import BASE, episode_starts  # noqa: E402

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def nd_job(args):
    E, start = args
    return episode.run_episode(E, start, 1, "aggressive", BASE, strategies=["no_defense"],
                               episode_id=f"{E}_{start}")


def bm_job(args):
    E, start, tt = args
    cal = calibrate.calibrate(E)
    st = episode.episode_setup(E, start, BASE, cal)
    X = gen.consumption(st["xsys"], BASE, seed=1000 * E + 1)
    doy = int(pd.Timestamp(start).dayofyear)
    s = BASE.sigma_perc
    ea = np.random.default_rng([E, 1, doy, 2]).standard_normal(BASE.episode_len) * s
    ed = np.random.default_rng([E, 1, doy, 3]).standard_normal((BASE.episode_len, BASE.n_traj)) * s
    diag = []
    rows = episode.run_strategy("benchmark", tt, BASE, cal, st, X, ea, ed, diag=diag)
    d = pd.DataFrame(diag); d["summer"] = E; d["true_type"] = tt; d["start"] = start
    r = pd.DataFrame(rows); r["summer"] = E; r["start"] = start
    r["price"] = st["price"][st["s0"]:st["t_end"]]
    return d, r


def check_bias(nd):
    say("== 1. bias decomposition (no defense, aggressive, seed 1) ==")
    out = []
    for (E, ep), d in nd.groupby(["summer", "episode_id"]):
        cal = calibrate.calibrate(E)
        a = d.a_eff_op.to_numpy()
        T = len(a)
        recs = {}
        for name, cw in (("rec", cal["cw_ev"]), ("rec_ss", cal["cw"])):
            rec = np.zeros(T)
            for t in range(T):
                for s in range(t):          # obs at s affects prediction of t (origin t-1)
                    j = t - 1 - s
                    if j < len(cw):
                        rec[t] += cw[j] * a[s]
            recs[name] = rec
        out.append(pd.DataFrame(dict(summer=E, ep=ep, b=d.bias_attack.to_numpy(),
                                     event=d.event.to_numpy(), **recs)))
    o = pd.concat(out)
    for name, m in (("all hours", np.ones(len(o), bool)), ("event hours", o.event.to_numpy() == 1)):
        x = o[m]
        say(f"{name}: corr={np.corrcoef(x.b, x.rec)[0, 1]:.3f}, mean filtered={x.b.mean():.3f} MW, "
            f"mean from c={x.rec.mean():.3f} MW, RMSE={np.sqrt(((x.b - x.rec) ** 2).mean()):.3f} MW, "
            f"slope={np.polyfit(x.rec, x.b, 1)[0]:.3f}  [steady-state weights: mean={x.rec_ss.mean():.3f}, "
            f"corr={np.corrcoef(x.b, x.rec_ss)[0, 1]:.3f}]")
    by = o[o.event == 1].groupby("summer").apply(lambda x: pd.Series(dict(b=x.b.mean(), rec_event_w=x.rec.mean(), rec_steady_w=x.rec_ss.mean())))
    say(by.round(3).to_string())
    say()


def check_reserve():
    say("== 2. reserve transfer: P(e > R), truthful data ==")
    for E in data.EVAL_SUMMERS:
        cal = calibrate.calibrate(E)
        st = episode.episode_setup(E, episode_starts(E)[0], BASE, cal)
        idx = st["idx"]
        summ = np.asarray(idx >= pd.Timestamp(f"{E}-06-01 01:00"))
        storm = data.storm_mask(idx) if E == 2024 else np.zeros(len(idx), bool)
        X = gen.consumption(st["xsys"], BASE, seed=1000 * E + 1)
        XP = X.sum(1)
        kf = fc.KF(cal["port_p"])
        i0 = np.flatnonzero(summ)[0]
        mask = ~st["ev"][i0:]
        K, Sv, _ = kf.gains(mask, kf.steady_state())
        a0 = kf.predict_mean(fc.init_state(XP[:i0], kf.D))
        mX, _ = fc.filter_means(kf, K, a0, XP[i0:, None], mask)
        mX = mX[:, :, 0].mean(1)
        e = (st["xsys"][i0:] - XP[i0:]) - (st["msys"][i0:] - mX)
        e = e[~storm[i0:]]
        evm = st["ev"][i0:][~storm[i0:]]
        R = cal["R"]
        say(f"{E} (calib {cal['C']}): R={R:.1f} MW, sd(e) calib={cal['e_sd']:.1f} eval={e.std():.1f}; "
            f"P(e>R) all hours={np.mean(e > R):.4f}, event hours={np.mean(e[evm] > R):.4f} (n={evm.sum()}); "
            f"model={cal['edist'].sf(R):.4f}")
    say()


def check_sign(nd):
    say("== 3. sign of b^a under no defense, 30 episodes ==")
    g = nd.groupby(["summer", "episode_id"])
    t = pd.DataFrame(dict(mean_all=g.bias_attack.mean(),
                          mean_event=nd[nd.event == 1].groupby(["summer", "episode_id"]).bias_attack.mean(),
                          min_event=nd[nd.event == 1].groupby(["summer", "episode_id"]).bias_attack.min(),
                          share_neg_all=g.bias_attack.apply(lambda x: (x < 0).mean()),
                          n_event=g.event.sum()))
    say(t.round(3).to_string())
    say(f"episodes with mean b^a > 0: {(t.mean_all > 0).sum()}/{len(t)}; event-hour b^a < 0 anywhere: "
        f"{(t.min_event < 0).sum()} episodes")
    say()


def check_prediction(dg, rows):
    say("== 4a. defender's attack prediction with the type known (benchmark) ==")
    for tt, d in dg.groupby("true_type"):
        base = 1 / 9
        say(f"{tt}: n={len(d)}, mean p(realized a)={d.p_hit.mean():.3f} (uniform {base:.3f}), "
            f"share p>=0.5={np.mean(d.p_hit >= 0.5):.3f}, mean realized a={d.a.mean():.3f}, "
            f"mean predicted a={d.a_pred_mean.mean():.3f}, corr={np.corrcoef(d.a, d.a_pred_mean)[0, 1]:.3f}; "
            f"budget est-true mean={np.mean(d.rem_est - d.rem_true):.2f} MWh")
    say()
    say("== 4b. fraud identity ==")
    errs = []
    for (E, s, _), r in rows.groupby(["summer", "start", "true_type"]):
        F1 = calibrate.calibrate(E)["F"][0]
        ne = r[r.event == 0]
        for i, row in r[r.event == 1].iterrows():
            prev = ne[ne.tau < row.tau].tail(BASE.W)
            if len(prev) < BASE.W:
                continue
            pred = row.price * prev.a_eff_set.mean()
            errs.append((row.fraud, pred))
    e = np.array(errs)
    if len(e):
        say(f"event hours checked: {len(e)}, max |fraud - p*mean(a_set)| = {np.abs(e[:, 0] - e[:, 1]).max():.4f} $ "
            f"(nonzero only where a baseline is below F), mean fraud = {e[:, 0].mean():.2f} $")
    say()


if __name__ == "__main__":
    eps = [(E, s) for E in data.EVAL_SUMMERS for s in episode_starts(E)]
    with Pool(8) as p:
        nd = pd.concat(p.map(nd_job, eps), ignore_index=True)
        bm = p.map(bm_job, [(E, s, tt) for (E, s) in [eps[1], eps[4], eps[13], eps[16]]
                            for tt in ("cautious", "moderate", "aggressive")])
    check_bias(nd)
    check_reserve()
    check_sign(nd)
    check_prediction(pd.concat([b[0] for b in bm]), pd.concat([b[1] for b in bm]))
    with open(os.path.join(HERE, "logs", "design_checks.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")
