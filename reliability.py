"""Event rule, forecast-error distribution, LOLP (Section 4.2) and the
decision-side quantities eq:exposure and eq:rel_attrib (Section 6.1)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def events(price, xsys, msys, cfg, start):
    """Event indicator for hours t >= start (earlier hours: 0).
    Event if price_t > pct(price[t-L:t], ev_price_pct) and
    msys_t > pct(xsys[t-L:t], ev_load_pct). System-level only."""
    T = len(price)
    L = cfg.ev_lookback
    ev = np.zeros(T, bool)
    pw = np.lib.stride_tricks.sliding_window_view(price, L)[:-1]   # windows ending t-1
    lw = np.lib.stride_tricks.sliding_window_view(xsys, L)[:-1]
    tp = np.percentile(pw, cfg.ev_price_pct, axis=1)
    tl = np.percentile(lw, cfg.ev_load_pct, axis=1)
    t = np.arange(L, T)
    ok = t >= start
    ev[t[ok]] = (price[t[ok]] > tp[ok]) & (msys[t[ok]] > tl[ok])
    return ev


class ErrorDist:
    """Distribution of e in eq:lolp_bias: normal or Student-t by AIC."""

    def __init__(self, e):
        e = np.asarray(e)
        mu, sd = stats.norm.fit(e)
        ll_n = stats.norm.logpdf(e, mu, sd).sum()
        df, loc, sc = stats.t.fit(e)
        ll_t = stats.t.logpdf(e, df, loc, sc).sum()
        self.aic = {"normal": 4 - 2 * ll_n, "t": 6 - 2 * ll_t}
        if self.aic["t"] < self.aic["normal"]:
            self.family, self.dist = "t", stats.t(df, loc, sc)
            self.params = dict(df=df, loc=loc, scale=sc)
        else:
            self.family, self.dist = "normal", stats.norm(mu, sd)
            self.params = dict(loc=mu, scale=sd)

    def sf(self, x):
        return self.dist.sf(x)

    def reserve(self, target):
        return float(self.dist.isf(target))


def hod_profile(values, hod, mask):
    """Mean of values by hour of day over the hours in mask (24,)."""
    out = np.zeros(24)
    for k in range(24):
        m = mask & (hod == k)
        out[k] = values[m].mean() if m.any() else 0.0
    return out


def exposure_and_event_prob(price, ev, hod, t, cfg):
    """E[p ev] and E[ev] by hour of day over the previous 30 days (decision
    #2 of the plan). Returns gbar_t (eq:exposure, h = 1..W ahead) and the
    event probabilities for hours t+h+j, j = 0..J."""
    L = cfg.ev_lookback
    lo = max(0, t - L)
    sl = slice(lo, t)
    h_ = hod[sl]
    pe = np.bincount(h_, weights=(price[sl] * ev[sl]), minlength=24) / np.maximum(np.bincount(h_, minlength=24), 1)
    pr = np.bincount(h_, weights=ev[sl].astype(float), minlength=24) / np.maximum(np.bincount(h_, minlength=24), 1)
    ahead = (hod[t] + np.arange(1, cfg.W + 1)) % 24
    gbar = pe[ahead].mean()
    ahead2 = (hod[t] + cfg.h + np.arange(cfg.J + 1)) % 24
    return gbar, pr[ahead2]


def rho(D, bhat, pev, c, R, edist):
    """eq:rel_attrib. D array (...,), bhat (J+1,), pev (J+1,), c (J+1,)."""
    D = np.asarray(D, float)
    base = edist.sf(R - bhat)
    shifted = edist.sf(R - bhat - c * D[..., None])
    return ((shifted - base) * pev).sum(-1)
