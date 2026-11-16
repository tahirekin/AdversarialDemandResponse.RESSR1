"""Batch runs of the nested experiment versions (design spec Section 7).

  python3 run_all.py B            # version B, all jobs
  python3 run_all.py B --only base
  python3 run_all.py B --procs 8

Each finished (summer, episode, seed, true type, setting) is written to
results/<version>/<setting>/<summer>_<start>_s<seed>_<type>.csv.gz, so an
interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

import calibrate  # noqa: E402
import data  # noqa: E402
from config import BASE, episode_starts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

SENS_B = {
    "L10": dict(L_frac=0.10),
    "L30": dict(L_frac=0.30),
    "CD25": dict(C_D=25.0),
    "CD400": dict(C_D=400.0),
    "sig025": dict(sigma_perc=0.25),
    "sig1": dict(sigma_perc=1.0),
    "sig_inf": dict(sigma_perc=float("inf")),        # feedback only
    "sg_half": dict(sigma_classes=(0.02, 0.04, 0.075)),
    "sg_double": dict(sigma_classes=(0.08, 0.16, 0.30)),
    "s1_10": dict(s1=0.10),
    "s1_40": dict(s1=0.40),
}
SENS_C = {**SENS_B,
          "w1_50k": dict(w1=50_000.0), "w1_500k": dict(w1=500_000.0),
          "q025": dict(q=0.25), "q1": dict(q=1.0), "kappa0": dict(kappa=0), "kappa72": dict(kappa=72)}


def jobs(version, only=None):
    out = []
    if version == "A":
        summers, seeds = [2017], [1, 2, 3]
        types = {2017: ["aggressive", "honest"]}
        sens = {"L10": SENS_B["L10"], "L30": SENS_B["L30"]}
        sens_summers = [2017]
    else:
        summers = data.EVAL_SUMMERS
        seeds = [1, 2, 3] if version == "B" else [1, 2, 3, 4, 5]
        if version == "B":
            # all four types in every summer (cautious and moderate extended
            # from 2017/2023 to all five summers on 16 Nov 2026)
            types = {E: ["aggressive", "honest", "cautious", "moderate"] for E in summers}
        else:
            types = {E: ["honest", "cautious", "moderate", "aggressive"] for E in summers}
        sens = SENS_B if version == "B" else SENS_C
        sens_summers = [2017, 2023]
    if only in (None, "base"):
        for E in summers:
            for start in episode_starts(E):
                for seed in seeds:
                    for tt in types[E]:
                        out.append(("base", {}, E, start, seed, tt))
    if only in (None, "sens"):
        for name, kw in sens.items():
            for E in sens_summers:
                for start in episode_starts(E):
                    for seed in seeds:
                        out.append((name, kw, E, start, seed, "aggressive"))
    return out


def _path(version, setting, E, start, seed, tt):
    return os.path.join(RESULTS, version, setting, f"{E}_{start}_s{seed}_{tt}.csv.gz")


def work(args):
    version, (setting, kw, E, start, seed, tt) = args
    import episode
    path = _path(version, setting, E, start, seed, tt)
    if os.path.exists(path):
        return path, 0.0, None
    t0 = time.time()
    try:
        cfg = BASE.copy(**kw)
        df = episode.run_episode(E, start, seed, tt, cfg, episode_id=f"{E}_{start}")
        df["setting"] = setting
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, path)
        return path, time.time() - t0, None
    except Exception:
        return path, time.time() - t0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("version", choices=["A", "B", "C"])
    ap.add_argument("--only", choices=["base", "sens"])
    ap.add_argument("--procs", type=int, default=max(1, os.cpu_count() - 2))
    a = ap.parse_args()
    js = jobs(a.version, a.only)
    todo = [j for j in js if not os.path.exists(_path(a.version, *[j[0]] + list(j[2:])))]
    print(f"version {a.version}: {len(js)} jobs, {len(todo)} to run, {a.procs} processes", flush=True)
    # calibrations first (one per distinct setting x summer), in parallel
    need = {}
    for setting, kw, E, *_ in todo:
        cfg = BASE.copy(**kw)
        need[calibrate.calib_key(E, cfg)] = (E, kw)
    with Pool(a.procs) as pool:
        pool.starmap(_calib, list(need.values()))
        t0 = time.time()
        done = 0
        for path, dt, err in pool.imap_unordered(work, [(a.version, j) for j in todo]):
            done += 1
            if err:
                print(f"ERROR {path}\n{err}", flush=True)
            elif done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"{done}/{len(todo)} done, {el/60:.1f} min elapsed, "
                      f"~{el/done*(len(todo)-done)/60:.1f} min left", flush=True)


def _calib(E, kw):
    calibrate.calibrate(E, BASE.copy(**kw), verbose=False)


if __name__ == "__main__":
    main()
