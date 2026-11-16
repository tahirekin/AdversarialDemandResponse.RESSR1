"""Bayesian seasonal ARIMA load model in state-space form (Section 4.2).

Model on levels (so that the predictive mean is linear in the
observations, eq:bias_weights):

    (1 - phi1 B - phi2 B^2)(1 - Phi B^24)(1 - B^24) y_t = w_t,  w_t ~ N(0, s2w)
    obs_t = y_t + v_t,  v_t ~ N(0, V),  obs_t missing in event hours.

y_t = sum_l c_l y_{t-l} + w_t with l = 1..50. State: the last 50 values of
y (companion form), so W = s2w e1 e1' has rank one.

Posterior sampling (offline): Gibbs sampler. The latent path y | rest is
Gaussian with a banded precision matrix and is drawn jointly by a banded
Cholesky factorisation; this is the same conditional distribution that
forward-filtering backward-sampling draws from. phi | Phi, Phi | phi,
s2w and V have conjugate (normal / inverse-gamma) full conditionals.

Inside an episode: exact Kalman filter for each retained draw,
vectorized over draws, using the companion structure (O(n^2) per step).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded, solve_banded

S = 24
NLAG = 2 + 2 * S  # 50


def ar_coefs(phi1, phi2, Phi):
    """c_1..c_50 of y_t = sum c_l y_{t-l} + w_t."""
    p = np.zeros(3); p[0] = 1; p[1] = -phi1; p[2] = -phi2
    sp = np.zeros(S + 1); sp[0] = 1; sp[S] = -Phi
    sd = np.zeros(S + 1); sd[0] = 1; sd[S] = -1
    a = np.convolve(np.convolve(p, sp), sd)  # a_0 .. a_50, a_0 = 1
    return -a[1:]


# ----------------------------------------------------------------------
#  Gibbs sampler
# ----------------------------------------------------------------------
def _ata_band(a, T):
    """Upper band (b+1, T) of A'A, A[s, s-m] = a_m for s = b..T-1."""
    b = len(a) - 1
    band = np.zeros((b + 1, T))
    j = np.arange(T)
    for l in range(b + 1):
        prod = a[l:] * a[:b + 1 - l]          # prod[m'] = a_{m'+l} a_{m'}
        cs = np.concatenate([[0.0], np.cumsum(prod)])
        lo = np.clip(b - j, 0, b - l)          # m' >= b - j
        hi = np.clip(T - 1 - j, -1, b - l)     # m' <= T-1-j
        val = np.where(hi >= lo, cs[np.maximum(hi, -1) + 1] - cs[lo], 0.0)
        # entry Q[j-l, j]; ab[b - l, j]
        band[b - l, l:] = val[l:]
    return band


def _draw_latent(obs, mask, c, s2w, V, rng):
    T = len(obs)
    a = np.concatenate([[1.0], -c])
    b = len(a) - 1
    band = _ata_band(a, T) / s2w
    band[b] += mask / V
    band[b, :b] += 1e-6 / V                    # weak prior on initial values
    U = cholesky_banded(band, lower=False)
    rhs = np.where(mask, obs, 0.0) / V
    mu = cho_solve_banded((U, False), rhs)
    z = rng.standard_normal(T)
    return mu + solve_banded((0, b), U, z)


def _reg_draw(Xr, yr, s2, prior_var, rng):
    P = Xr.T @ Xr / s2 + np.eye(Xr.shape[1]) / prior_var
    Cov = np.linalg.inv(P)
    m = Cov @ (Xr.T @ yr) / s2
    return rng.multivariate_normal(m, Cov)


def _stationary_ar2(p1, p2):
    return abs(p2) < 1 and p2 + p1 < 1 and p2 - p1 < 1


def fit(obs, mask, n_chains=4, n_burn=2000, n_keep=2000, seed=0, init=None, thin=1):
    """Return dict with arrays phi1, phi2, Phi, s2w, V of shape
    (n_chains, n_keep)."""
    out = {k: np.empty((n_chains, n_keep)) for k in ("phi1", "phi2", "Phi", "s2w", "V")}
    T = len(obs)
    o = np.where(mask, obs, np.interp(np.arange(T), np.flatnonzero(mask), obs[mask]))
    # empirical-Bayes scale: variance of the seasonal-differenced series
    d0 = o[S:] - o[:-S]
    s2_0 = np.var(np.diff(d0))
    a0 = 2.0
    b0w, b0v = s2_0, 0.05 * s2_0
    for ch in range(n_chains):
        rng = np.random.default_rng([seed, ch])
        phi1, phi2, Phi = rng.uniform(0.3, 0.9), 0.0, rng.uniform(-0.5, 0.0)
        s2w, V = s2_0 * rng.uniform(0.5, 1.5), 0.05 * s2_0 * rng.uniform(0.5, 2)
        y = o.copy()
        for it in range(n_burn + n_keep * thin):
            c = ar_coefs(phi1, phi2, Phi)
            y = _draw_latent(obs, mask, c, s2w, V, rng)
            d = y[S:] - y[:-S]
            # phi | Phi
            w = d[S:] - Phi * d[:-S]
            Xr = np.column_stack([w[1:-1], w[:-2]]); yr = w[2:]
            for _ in range(20):
                p1, p2 = _reg_draw(Xr, yr, s2w, 1.0, rng)
                if _stationary_ar2(p1, p2):
                    phi1, phi2 = p1, p2
                    break
            # Phi | phi
            v = d[2:] - phi1 * d[1:-1] - phi2 * d[:-2]
            for _ in range(20):
                P1 = _reg_draw(v[:-S, None], v[S:], s2w, 1.0, rng)[0]
                if abs(P1) < 1:
                    Phi = P1
                    break
            # s2w, V
            c = ar_coefs(phi1, phi2, Phi)
            res = y[NLAG:] - np.lib.stride_tricks.sliding_window_view(y[:-1], NLAG)[:, ::-1] @ c
            s2w = 1 / rng.gamma(a0 + len(res) / 2, 1 / (b0w + res @ res / 2))
            r2 = (obs - y)[mask]
            V = 1 / rng.gamma(a0 + len(r2) / 2, 1 / (b0v + r2 @ r2 / 2))
            if it >= n_burn and (it - n_burn) % thin == 0:
                k = (it - n_burn) // thin
                out["phi1"][ch, k], out["phi2"][ch, k], out["Phi"][ch, k] = phi1, phi2, Phi
                out["s2w"][ch, k], out["V"][ch, k] = s2w, V
    return out


def thin(draws, n, seed=0):
    flat = {k: v.reshape(-1) for k, v in draws.items()}
    m = len(flat["V"])
    idx = np.linspace(0, m - 1, n).astype(int)
    p = {k: v[idx] for k, v in flat.items()}
    p["c"] = np.stack([ar_coefs(a, b, c) for a, b, c in zip(p["phi1"], p["phi2"], p["Phi"])])
    return p


# ----------------------------------------------------------------------
#  Kalman filter, vectorized over draws (companion structure)
# ----------------------------------------------------------------------
class KF:
    """Covariance recursion for D parameter draws. The recursion depends on
    the parameters and the missing-data pattern only, so gains are computed
    once and reused for every observation series."""

    def __init__(self, params):
        self.c = params["c"]                       # (D, n)
        self.s2w = params["s2w"]
        self.V = params["V"]
        self.D, self.n = self.c.shape

    def predict_cov(self, P):
        c = self.c
        M = np.empty_like(P)
        M[:, 0, :] = np.einsum("dn,dnm->dm", c, P)
        M[:, 1:, :] = P[:, :-1, :]
        Pn = np.empty_like(P)
        Pn[:, :, 0] = np.einsum("dnm,dm->dn", M, c)
        Pn[:, :, 1:] = M[:, :, :-1]
        Pn[:, 0, 0] += self.s2w
        return Pn

    def predict_mean(self, a):
        """a: (D, n) or (D, n, k) -> one-step state prediction."""
        out = np.empty_like(a)
        out[:, 0] = np.einsum("dn,dn...->d...", self.c, a)
        out[:, 1:] = a[:, :-1]
        return out

    def gains(self, mask, P0):
        """Run the covariance recursion over len(mask) steps starting from
        the predicted covariance P0 at the first step. Returns K (T, D, n)
        (zero at missing steps), the predictive variance of the observation
        Sv (T, D), and the final predicted covariance."""
        T = len(mask)
        K = np.zeros((T, self.D, self.n))
        Sv = np.empty((T, self.D))
        P = P0.copy()
        for t in range(T):
            Sv[t] = P[:, 0, 0] + self.V
            if mask[t]:
                k = P[:, :, 0] / Sv[t][:, None]
                K[t] = k
                P = P - k[:, :, None] * P[:, 0, None, :]
            P = self.predict_cov(P)
        return K, Sv, P

    def steady_state(self, n_iter=1500):
        P = np.tile(np.eye(self.n), (self.D, 1, 1)) * self.s2w[:, None, None] * 10
        _, _, P = self.gains(np.ones(n_iter, bool), P)
        return P

    def forecast_weights(self, nlag=400):
        """c_{1,j}: response of the one-step predictive mean at origin t+j to
        a unit change in the observation at t, steady state, mean over draws."""
        P = self.steady_state()
        S_ = P[:, 0, 0] + self.V
        k = P[:, :, 0] / S_[:, None]               # (D, n) steady gain
        delta = k.copy()                           # filtered-state change at t
        w = np.empty((nlag, self.D))
        for j in range(nlag):
            pred = self.predict_mean(delta)        # change of prediction for t+j+1
            w[j] = pred[:, 0]
            # observation at t+j+1 is unchanged: filtered = pred - k * pred[0]
            delta = pred - k * pred[:, :1]
        return w.mean(1)


def event_weights(params, mask, ev_next, nlag=400, n_draws=10, fallback=None):
    """Forecast weights for event hours, from the exact (time-varying) gains.

    For every observed hour s, a unit change in the observation is
    propagated through the filter with the actual missing-data pattern
    (event hours have no reading). The effect on the one-step prediction of
    hour s+j+1 is recorded whenever s+j+1 is an event hour; c_ev[j] is the
    mean over sources and draws. Event hours come in runs, so the relevant
    prediction is often several steps past the last reading and the
    steady-state weights understate the bias there.
    ev_next[t] = event indicator of hour t+1. Lags with fewer than 5
    occurrences fall back to the steady-state weights."""
    sub = {k: (v[:n_draws] if np.ndim(v) else v) for k, v in params.items()}
    kf = KF(sub)
    K, _, _ = kf.gains(mask, kf.steady_state())
    T = len(mask)
    src = np.flatnonzero(mask)
    S_ = len(src)
    delta = np.zeros((kf.D, kf.n, S_))
    active = np.zeros(S_, bool)
    start = np.full(T, -1); start[src] = np.arange(S_)
    sums = np.zeros(nlag); cnt = np.zeros(nlag)
    for t in range(T):
        j = start[t]
        if j >= 0:                                   # unit change in obs at t
            delta[:, :, j] = K[t]
            active[j] = True
        pred = kf.predict_mean(delta)                # effect on prediction of t+1
        if t + 1 < T:
            if ev_next[t]:
                lags = t - src
                ok = active & (lags < nlag) & (lags >= 0)
                np.add.at(sums, lags[ok], pred[:, 0, ok].mean(0))
                np.add.at(cnt, lags[ok], 1)
            delta = pred - K[t + 1][:, :, None] * pred[:, :1, :]
            old = (t + 1 - src) >= nlag
            delta[:, :, old] = 0.0
            active &= ~old
    c = np.where(cnt >= 5, sums / np.maximum(cnt, 1), np.nan)
    if fallback is not None:
        c = np.where(np.isnan(c), fallback[:nlag], c)
    return c, cnt


def filter_means(kf, K, a0, obs, mask):
    """One-step predictive means for several observation series.

    a0: (D, n, k) predicted state at step 0 for k series; obs: (T, k).
    Returns m (T, D, k) predictive means of y_t given data through t-1 and
    the predicted state after the last step."""
    T = obs.shape[0]
    a = a0.copy()
    m = np.empty((T, kf.D, obs.shape[1]))
    for t in range(T):
        m[t] = a[:, 0, :]
        if mask[t]:
            a = a + K[t][:, :, None] * (obs[t][None, :] - a[:, 0, :])[:, None, :]
        a = kf.predict_mean(a)
    return m, a


def init_state(history, D, k=1):
    """Predicted state built from the last 50 values (most recent first)."""
    h = np.asarray(history)[::-1][:NLAG]
    a = np.repeat(h[None, :, None], D, 0)
    return np.repeat(a, k, 2)
