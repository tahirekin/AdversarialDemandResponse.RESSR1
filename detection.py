"""Bayesian Gaussian mixture outlier detection (Section 4.1).

Readings are profile-adjusted (reading / participant's hour-of-day mean
in the calibration summer). Each participant belongs to one of K
load-behaviour clusters (z_i is per participant: pi_k is the population
share of participants in cluster k). Conjugate Normal-Inverse-Gamma /
Dirichlet priors; Gibbs sampling offline. Within an episode the
membership probabilities r_ik are updated in closed form and the
predictive density eq:pred_density uses the posterior means.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm


def fit_gmm(Y, K=3, n_chains=4, n_burn=2000, n_keep=2000, seed=0):
    """Y: (T, N) profile-adjusted readings (NaN = not used).
    Returns draws of mu, s2 (n_chains, n_keep, K) and pi, sorted by s2."""
    T, N = Y.shape
    ok = ~np.isnan(Y)
    n_i = ok.sum(0)
    s1 = np.where(ok, Y, 0).sum(0)
    s2 = np.where(ok, Y ** 2, 0).sum(0)
    # empirical Bayes hyperparameters from the pooled clean readings
    mu0 = s1.sum() / n_i.sum()
    var0 = s2.sum() / n_i.sum() - mu0 ** 2
    kappa0, a0, b0, alpha0 = 0.01, 2.0, var0, 1.0
    out = {k: np.empty((n_chains, n_keep, K)) for k in ("mu", "s2", "pi")}
    for ch in range(n_chains):
        rng = np.random.default_rng([seed, ch])
        # initialise by sample variance terciles of participants
        v_i = s2 / n_i - (s1 / n_i) ** 2
        z = np.searchsorted(np.quantile(v_i, np.linspace(0, 1, K + 1)[1:-1]), v_i)
        z = np.clip(z, 0, K - 1)
        mu = np.full(K, mu0); sig2 = np.quantile(v_i, (np.arange(K) + 0.5) / K)
        for it in range(n_burn + n_keep):
            # cluster parameters | z
            for k in range(K):
                m = z == k
                n = n_i[m].sum(); S1 = s1[m].sum(); S2 = s2[m].sum()
                if n == 0:
                    sig2[k] = 1 / rng.gamma(a0, 1 / b0)
                    mu[k] = rng.normal(mu0, np.sqrt(sig2[k] / kappa0))
                    continue
                ybar = S1 / n
                kn = kappa0 + n
                mn = (kappa0 * mu0 + S1) / kn
                an = a0 + n / 2
                bn = b0 + 0.5 * (S2 - n * ybar ** 2) + kappa0 * n * (ybar - mu0) ** 2 / (2 * kn)
                sig2[k] = 1 / rng.gamma(an, 1 / bn)
                mu[k] = rng.normal(mn, np.sqrt(sig2[k] / kn))
            pi = rng.dirichlet(alpha0 + np.bincount(z, minlength=K))
            # z | parameters (participant level)
            ll = (-0.5 * n_i[:, None] * np.log(2 * np.pi * sig2)[None]
                  - (s2[:, None] - 2 * mu[None] * s1[:, None] + n_i[:, None] * mu[None] ** 2)
                  / (2 * sig2[None]) + np.log(pi)[None])
            p = np.exp(ll - logsumexp(ll, 1, keepdims=True))
            z = (rng.uniform(size=(N, 1)) > np.cumsum(p, 1)).sum(1)
            z = np.minimum(z, K - 1)
            order = np.argsort(sig2)                  # identify by variance
            if it >= n_burn:
                j = it - n_burn
                out["mu"][ch, j] = mu[order]; out["s2"][ch, j] = sig2[order]
                out["pi"][ch, j] = pi[order]
    return out


def gmm_summary(draws):
    """Posterior means used in eq:pred_density, V_k = posterior var of mu_k."""
    mu = draws["mu"].reshape(-1, draws["mu"].shape[-1])
    s2 = draws["s2"].reshape(mu.shape)
    pi = draws["pi"].reshape(mu.shape)
    return dict(mu=mu.mean(0), s2=s2.mean(0), V=mu.var(0), pi=pi.mean(0))


def log_member_lik(y, g):
    """(..., K) log N(y; mu_k, s2_k + V_k)."""
    sd = np.sqrt(g["s2"] + g["V"])
    return norm.logpdf(np.asarray(y)[..., None], g["mu"], sd)


def membership_from_history(Y, g):
    """r_ik = P(z_i = k | readings), Y (T, N) with NaN for unused."""
    ll = np.where(~np.isnan(Y)[..., None], log_member_lik(np.nan_to_num(Y), g), 0).sum(0)
    ll = ll + np.log(g["pi"])
    return np.exp(ll - logsumexp(ll, 1, keepdims=True)), ll


class TailProb:
    """u_i(y) of eq:tail_prob for a mixture with weights r (N, K).

    u(y) = P(f(Ytilde) <= f(y)). For a participant whose membership is
    essentially certain the mixture is a single normal and
    u(y) = 2 Phi(-|z|). Otherwise u is tabulated on a fine grid."""

    def __init__(self, g, r, ngrid=4001):
        self.g = g
        self.sd = np.sqrt(g["s2"] + g["V"])
        self.r = r
        N, K = r.shape
        self.single = r.max(1) > 1 - 1e-10
        self.k1 = r.argmax(1)
        self.tables = {}
        lo = (g["mu"] - 10 * self.sd).min(); hi = (g["mu"] + 10 * self.sd).max()
        grid = np.linspace(lo, hi, ngrid)
        dx = grid[1] - grid[0]
        for i in np.flatnonzero(~self.single):
            f = (r[i] * norm.pdf(grid[:, None], g["mu"], self.sd)).sum(1)
            o = np.argsort(f)
            cdf = np.cumsum(f[o]) * dx            # mass of {f <= f_(j)}
            self.tables[i] = (f[o], np.minimum(cdf, 1.0))

    def density(self, i, y):
        y = np.asarray(y, float)
        return (self.r[i] * norm.pdf(y[..., None], self.g["mu"], self.sd)).sum(-1)

    def u(self, i, y):
        y = np.asarray(y, float)
        if self.single[i]:
            k = self.k1[i]
            return 2 * norm.sf(np.abs(y - self.g["mu"][k]) / self.sd[k])
        fs, cdf = self.tables[i]
        return np.interp(self.density(i, y), fs, cdf)

    def u_all(self, y):
        """u for every participant at its own reading y (N,)."""
        out = np.empty(len(y))
        s = self.single
        k = self.k1[s]
        out[s] = 2 * norm.sf(np.abs(y[s] - self.g["mu"][k]) / self.sd[k])
        for i in np.flatnonzero(~s):
            out[i] = self.u(i, y[i])
        return out

    def imputed(self):
        """m_i on the adjusted scale: sum_k r_ik mu_k."""
        return self.r @ self.g["mu"]

    def quadrature(self, i, n):
        """Nodes and weights for E_f[h(Y)] under participant i's mixture
        (Gauss-Hermite per component with weight r_ik > 1e-8)."""
        x, w = np.polynomial.hermite_e.hermegauss(n)
        w = w / w.sum()
        nodes, wts = [], []
        for k in np.flatnonzero(self.r[i] > 1e-8):
            nodes.append(self.g["mu"][k] + self.sd[k] * x)
            wts.append(self.r[i, k] * w)
        return np.concatenate(nodes), np.concatenate(wts)
