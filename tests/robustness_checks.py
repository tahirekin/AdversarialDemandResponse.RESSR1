"""Robustness of the main comparisons (writes logs/robustness_checks.txt):
(1) excluding the four episodes used to check the cost settings;
(2) a two-level bootstrap that resamples summers, then episodes."""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import tables as T
from config import episode_starts
TUNE = {f"2017_{episode_starts(2017)[1]}", f"2017_{episode_starts(2017)[4]}",
        f"2023_{episode_starts(2023)[1]}", f"2023_{episode_starts(2023)[4]}"}
PAIRS = [("ara_dddas", "ara_static"), ("ara_dddas", "static_rule"), ("ara_dddas", "no_defense"), ("benchmark", "ara_dddas")]
tot = T.episode_totals(T.load("B", "base")); tb = tot[tot.setting == "base"]
ho = tb[~tb.episode_id.isin(TUNE)]
sig = lambda r: not (r["lo"] <= 0 <= r["hi"])
rows = []
for a, b in PAIRS:
    for m in ("total_cost", "fraud", "dLOLE"):
        f = {r["true_type"]: r for r in T.paired_bootstrap(tb, a, b, m)}
        h = {r["true_type"]: r for r in T.paired_bootstrap(ho, a, b, m)}
        s2 = {r["true_type"]: r for r in T.paired_bootstrap_summers(tb, a, b, m)}
        for tt in ("honest", "cautious", "moderate", "aggressive"):
            sc = 1e3 if m == "dLOLE" else 1
            rows.append(dict(cmp=f"{a} - {b}", metric=m, type=tt,
                             full=f"{f[tt]['mean']*sc:.2f} [{f[tt]['lo']*sc:.2f}, {f[tt]['hi']*sc:.2f}]{'*' if sig(f[tt]) else ''}",
                             holdout=f"{h[tt]['mean']*sc:.2f} [{h[tt]['lo']*sc:.2f}, {h[tt]['hi']*sc:.2f}]{'*' if sig(h[tt]) else ''}",
                             two_level=f"[{s2[tt]['lo']*sc:.2f}, {s2[tt]['hi']*sc:.2f}]{'*' if sig(s2[tt]) else ''}",
                             holdout_same=(sig(f[tt]) == sig(h[tt])) and np.sign(f[tt]['mean']) == np.sign(h[tt]['mean']),
                             two_level_same=sig(f[tt]) == sig(s2[tt])))
d = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
txt = [d.to_string(index=False),
       f"\nholdout: same sign and significance in {d.holdout_same.sum()} of {len(d)}",
       f"two-level bootstrap: same significance in {d.two_level_same.sum()} of {len(d)}",
       "\nchanged under holdout:", d[~d.holdout_same][["cmp", "metric", "type", "full", "holdout"]].to_string(index=False),
       "\nchanged under two-level bootstrap:", d[~d.two_level_same][["cmp", "metric", "type", "full", "two_level"]].to_string(index=False)]
open(os.path.join(HERE, "logs", "robustness_checks.txt"), "w").write("\n".join(txt) + "\n")
print("\n".join(txt[1:]))
