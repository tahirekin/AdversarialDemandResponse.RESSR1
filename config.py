"""All settings of the illustration (manuscript Section 7.3, design spec 3-5).

Values marked [P] are provisional and fixed after the check episode.
A run is configured by copying BASE and overriding fields.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict

import numpy as np

DELTA_GRID = np.array([0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
TYPES = ["honest", "cautious", "moderate", "aggressive"]  # index 0..3
STRATEGIES = ["no_defense", "static_rule", "ara_static", "ara_dddas", "benchmark"]


@dataclass
class Config:
    # system and portfolio
    peak_mw: float = 3000.0
    n_part: int = 50
    portfolio_share: float = 0.05            # [P]
    rho_g: float = 0.7                       # [P]
    sigma_classes: tuple = (0.04, 0.08, 0.15)  # [P]
    s1: float = 0.25                         # [P]
    attacker_class: int = 1                  # middle class
    structure_seed: int = 20261115           # shares and classes, fixed
    calib_seed: int = 7                      # consumption in calibration summer
    # program
    W: int = 24
    firm_frac: float = 0.8                   # [P]
    peak_hours: tuple = tuple(range(1, 25))  # hours entering the firm level: all hours,
                                             # consistent with the 24-hour baseline
    h: int = 1
    lolp_target: float = 0.01                # [P]
    ev_price_pct: float = 95.0               # [P] tuned: events in ~4% of hours
    ev_load_pct: float = 75.0                # [P]
    ev_lookback: int = 720                   # hours
    J: int = 48                              # [P]
    # costs ($)
    w1: float = 150_000.0                    # [P]
    w2: float = 1.0
    C_D: float = 100.0                       # tuned 15 Nov (logs/tuning.txt)
    C_F: float = 5.0                         # tuned 15 Nov (logs/tuning.txt)
    # attacker
    L_frac: float = 0.20
    g_frac: float = 0.025
    B_mult: tuple = (0.0, 2.0, 6.0, 12.0)    # [P] per type, times L
    C_A: tuple = (0.0, 6.0, 3.0, 1.0)        # [P] $/MWh
    Pi_pen: tuple = (0.0, 5000.0, 2000.0, 500.0)  # [P]
    sigma_perc: float = 0.5                  # [P] log scale; np.inf = feedback only
    lam: float = 0.9                         # [P]
    kappa: int = 24                          # [P]
    eps_A: float = 0.1                       # [P]
    q: float = 0.5                           # [P]
    # defender
    eps: float = 0.01
    n_traj: int = 200                        # [P]
    n_quad: int = 15                         # [P]
    # statistical models
    n_chains: int = 4
    n_burn: int = 2000                       # [P]
    n_keep: int = 2000                       # [P]
    n_draws: int = 200                       # [P] retained for the episodes
    sys_burn_mult: int = 2                   # system-load chains: 2x burn-in
    sys_thin: int = 3                        # and thinning 3 (2024 mixed slowly)
    K: int = 3
    # episodes
    episode_len: int = 168
    warmup: int = 168

    def copy(self, **kw) -> "Config":
        c = copy.deepcopy(self)
        for k, v in kw.items():
            if not hasattr(c, k):
                raise AttributeError(k)
            setattr(c, k, v)
        return c

    def asdict(self):
        return asdict(self)


BASE = Config()

# Episode start dates (delivery date of the first hour), fixed in advance:
# every 14 days from 8 June. 2024: the 6 July week overlaps the storm and is
# replaced by the next free week (27 July).
EPISODE_STARTS = ["06-08", "06-22", "07-06", "07-20", "08-03", "08-17"]
EPISODE_STARTS_2024 = ["06-08", "06-22", "07-20", "07-27", "08-03", "08-17"]


def episode_starts(year: int) -> list[str]:
    md = EPISODE_STARTS_2024 if year == 2024 else EPISODE_STARTS
    return [f"{year}-{d}" for d in md]
