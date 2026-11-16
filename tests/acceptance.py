"""Acceptance tests of design spec Section 9 on one 2017 check episode.

Run:  python3 tests/acceptance.py [start-date]
Writes logs/acceptance.txt and the check-episode CSVs.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import attacker as att  # noqa: E402
import calibrate  # noqa: E402
import episode  # noqa: E402
import generator as gen  # noqa: E402
from config import BASE, DELTA_GRID  # noqa: E402

E = 2017
START = sys.argv[1] if len(sys.argv) > 1 else "2017-07-06"
SEED = 1
out = []


def report(name, ok, detail=""):
    line = f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else "")
    print(line, flush=True)
    out.append(line)


def totals(df):
    return df.groupby("strategy")[["dlolp", "dlolp_attack", "fraud", "audit_cost",
                                   "correction_cost", "utility"]].sum()


t0 = time.time()
cfg = BASE
cal = calibrate.calibrate(E, cfg)
agg = episode.run_episode(E, START, SEED, "aggressive", cfg)
t_ep = time.time() - t0
hon = episode.run_episode(E, START, SEED, "honest", cfg)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
agg.to_csv(os.path.join(HERE, "logs", "check_aggressive.csv"), index=False)
hon.to_csv(os.path.join(HERE, "logs", "check_honest.csv"), index=False)

# 1. no attack and no flags -> no reliability change, no fraud
h0 = hon[hon.strategy == "no_defense"]
report("1 no attack, no flags: dlolp = 0 and fraud = 0",
       np.allclose(h0.dlolp, 0, atol=1e-12) and np.allclose(h0.fraud, 0, atol=1e-9),
       f"max|dlolp|={h0.dlolp.abs().max():.2e}, max|fraud|={h0.fraud.abs().max():.2e}")

# 2. honest participant: fraud = 0 whatever the defense
report("2 honest: fraud = 0 under every strategy", np.all(hon.fraud.abs() < 1e-9),
       f"max|fraud|={hon.fraud.abs().max():.2e}")

# 3. no defense, aggressive: b^a > 0 and monotone in L
nd = agg[agg.strategy == "no_defense"]
ba_pos = nd.bias_attack.mean() > 0 and nd.loc[nd.event == 1, "bias_attack"].min() > 0
res = {}
for Lf in (0.10, 0.20, 0.30):
    d = episode.run_episode(E, START, SEED, "aggressive", cfg.copy(L_frac=Lf), strategies=["no_defense"])
    res[Lf] = (d.dlolp_attack.sum(), d.fraud.sum())
mono = res[0.10][0] < res[0.20][0] < res[0.30][0] and res[0.10][1] < res[0.20][1] < res[0.30][1]
report("3a no defense, aggressive: bias_attack > 0 (h = 1)", ba_pos,
       f"mean={nd.bias_attack.mean():.3f} MW, min over event hours={nd.loc[nd.event == 1, 'bias_attack'].min():.3f}")
report("3b attack dLOLE and fraud increase with L (10/20/30%)", mono,
       "; ".join(f"L={k:.0%}: dLOLE_a={v[0]:.4f}, fraud=${v[1]:.0f}" for k, v in res.items()))

# 4. calibration of the tail probability on truthful readings
st = episode.episode_setup(E, START, cfg, cal)
X = gen.consumption(st["xsys"], cfg, seed=1000 * E + SEED)
import detection  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
g = cal["gmm"]
r = cal["r"]
tp = detection.TailProb(g, r)
rows = []
for t in range(st["s0"], st["t_end"]):
    if st["ev"][t]:
        continue
    rows.append(tp.u_all(X[t] / gen.adjust(cal["prof"], cal["sysprof"], st["xsys"][t], st["hod"][t])))
U = np.array(rows)
rates = {d: (U < d).mean() for d in DELTA_GRID}
ok4 = all(abs(v - d) < max(0.5 * d, 0.01) for d, v in rates.items())
report("4 truthful readings flagged at rate ~ delta", ok4,
       ", ".join(f"{d}: {v:.3f}" for d, v in rates.items()))

# 5. honest true type: posterior moves toward honest
hd = hon[hon.strategy == "ara_dddas"]
last = hd.iloc[-1]
report("5 honest: posterior moves toward honest", last.nu_honest > 0.25 and last.nu_honest > last.nu_3,
       f"final nu = ({last.nu_honest:.3f}, {last.nu_1:.3f}, {last.nu_2:.3f}, {last.nu_3:.3f})")

# 6. degenerate prior reproduces the benchmark
Xall = X
L_ = cfg.episode_len
sig = cfg.sigma_perc
eta_att = np.random.default_rng([E, SEED, int(pd.Timestamp(START).dayofyear), 2]).standard_normal(L_) * sig
eta_def = np.random.default_rng([E, SEED, int(pd.Timestamp(START).dayofyear), 3]).standard_normal((L_, cfg.n_traj)) * sig
r_deg = pd.DataFrame(episode.run_strategy("ara_dddas", "aggressive", cfg, cal, st, Xall, eta_att, eta_def,
                                          nu0=np.eye(4)[3]))
bm = agg[agg.strategy == "benchmark"].reset_index(drop=True)
cols = ["delta", "beta", "a_attempt", "flagged", "audited", "dlolp", "fraud", "utility"]
same6 = np.allclose(r_deg[cols].fillna(-1).to_numpy(float), bm[cols].fillna(-1).to_numpy(float))
report("6 ARA-DDDAS with degenerate prior == benchmark", same6)

# 7. ARA-static and ARA-DDDAS decide alike at tau = 1
a1 = agg[(agg.tau == agg[agg.event == 0].tau.min())]
d_s = a1[a1.strategy == "ara_static"][["delta", "beta"]].to_numpy()
d_d = a1[a1.strategy == "ara_dddas"][["delta", "beta"]].to_numpy()
report("7 ARA-static and ARA-DDDAS: same decision at tau = 1", np.array_equal(d_s, d_d),
       f"static={d_s.tolist()}, dddas={d_d.tolist()}")

# 8. nested attack grids
grids = {Lf: att.attack_grid(cal["mean1"], cfg.copy(L_frac=Lf))[0] for Lf in (0.10, 0.20, 0.30)}
nested = all(np.allclose(grids[a], grids[b][:len(grids[a])]) for a, b in ((0.10, 0.20), (0.20, 0.30)))
report("8 larger L only adds attack levels", nested, ", ".join(f"{k:.0%}: {len(v)} levels" for k, v in grids.items()))

# 9. reproducibility
agg2 = episode.run_episode(E, START, SEED, "aggressive", cfg)
num = agg.select_dtypes("number").columns
report("9 same seed, same result", np.allclose(agg[num].fillna(-9).to_numpy(), agg2[num].fillna(-9).to_numpy()))

# 10. the defender does not always pick the strictest level
dd = agg[(agg.strategy == "ara_dddas") & (agg.event == 0)]
share = (dd.delta == DELTA_GRID.max()).mean()
report("10 not the strictest flagging level in every interval", share < 1,
       f"share at delta=0.20: {share:.2f}; levels used: {dd.delta.value_counts().sort_index().to_dict()}")

out.append("")
out.append(f"Run time, one episode with all five strategies (incl. setup): {t_ep:.1f} s")
out.append(totals(agg).round(4).to_string())
out.append(totals(hon).round(4).to_string())
with open(os.path.join(HERE, "logs", "acceptance.txt"), "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out[-3:]))
