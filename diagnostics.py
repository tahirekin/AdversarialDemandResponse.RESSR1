"""Split R-hat and effective sample size (Gelman et al., BDA3, 11.4-11.5)."""
from __future__ import annotations

import numpy as np


def _split(x):
    n = x.shape[1] // 2
    return np.concatenate([x[:, :n], x[:, n:2 * n]], 0)


def rhat(x):
    x = _split(np.asarray(x))
    m, n = x.shape
    B = n * x.mean(1).var(ddof=1)
    Wv = x.var(1, ddof=1).mean()
    var = (n - 1) / n * Wv + B / n
    return float(np.sqrt(var / Wv))


def ess(x):
    x = _split(np.asarray(x))
    m, n = x.shape
    Wv = x.var(1, ddof=1).mean()
    B = n * x.mean(1).var(ddof=1)
    var = (n - 1) / n * Wv + B / n
    xc = x - x.mean(1, keepdims=True)
    f = np.fft.rfft(xc, 2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
    rho = 1 - (Wv - acov.mean(0)) / var
    s = 0.0
    for t in range(1, n - 1, 2):              # Geyer initial positive sequence
        pair = rho[t] + rho[t + 1]
        if pair < 0:
            break
        s += pair
    return float(m * n / (1 + 2 * s))


def rhat_ess(draws: dict) -> dict:
    return {k: (rhat(v), ess(v)) for k, v in draws.items()}
