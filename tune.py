"""Cost tuning on a few check episodes (2017, 2023; seed 1).

Criteria, fixed before looking at strategy rankings:
 * attacker types: with no defense, the cautious type rarely attacks and
   the aggressive type attacks in most non-event hours (spec Section 5);
 * the audit lever is active: under the aggressive attacker the ARA
   defender audits some flagged readings of participant 1, while under the
   honest participant audits stay rare;
 * the defender does not flag at the strictest level throughout.
"""
from __future__ import annotations

import itertools
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import episode  # noqa: E402
from config import BASE, episode_starts  # noqa: E402

EPS = [(2017, episode_starts(2017)[1]), (2017, episode_starts(2017)[4]),
       (2023, episode_starts(2023)[1]), (2023, episode_starts(2023)[4])]


def job(args):
    tag, kw, E, start, tt, strategies = args
    df = episode.run_episode(E, start, 1, tt, BASE.copy(**kw), strategies=strategies)
    df["tag"] = tag
    for k, v in kw.items():
        df[k] = v
    return df


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "costs"
    if mode == "types":
        js = [("types", {}, E, s, tt, ["no_defense"]) for (E, s) in EPS
              for tt in ("cautious", "moderate", "aggressive")]
    else:
        grid = list(itertools.product([10.0, 25.0, 50.0, 100.0], [1.0, 5.0]))
        js = [(f"CD{cd:g}_CF{cf:g}", dict(C_D=cd, C_F=cf), E, s, tt,
               ["static_rule", "ara_static", "ara_dddas", "benchmark"])
              for cd, cf in grid for (E, s) in EPS for tt in ("aggressive", "honest")]
    with Pool(8) as p:
        df = pd.concat(p.map(job, js), ignore_index=True)
    df.to_pickle(f"cache/tune_{mode}.pkl")
    ne = df[df.event == 0]
    if mode == "types":
        print(ne.groupby("true_type").apply(lambda d: pd.Series({
            "attack_share": (d.a_attempt > 0).mean(), "mean_attempt": d.a_attempt.mean(),
            "fraud_week": d.fraud.sum() / d.episode_id.nunique()})).round(3))
        return
    g = df.groupby(["tag", "true_type", "strategy"])
    out = pd.DataFrame({
        "audits": g.audited.sum() / 4, "flags1": g.flagged.sum() / 4,
        "fraud": g.fraud.sum() / 4, "dLOLE": g.dlolp.sum() / 4,
        "audit_cost": g.audit_cost.sum() / 4, "corr_cost": g.correction_cost.sum() / 4,
        "total": -g.utility.sum() / 4,
        "share_d020": ne.groupby(["tag", "true_type", "strategy"]).delta.apply(lambda x: (x == 0.2).mean()),
        "beta1_share": ne.groupby(["tag", "true_type", "strategy"]).beta.mean(),
    }).round(4)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(out)


if __name__ == "__main__":
    main()
