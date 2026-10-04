"""Reference implementations: one date at a time, library calls where the
vectorised code uses hand-rolled indexing (np.quantile, np.cov). They exist
to be compared against risk.py in the tests and the benchmark."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .risk import tail_index


def _es(window: np.ndarray, alpha: float) -> float:
    s, n = np.sort(window), len(window)
    k = tail_index(n, alpha)
    return max((s[k:].sum() + (k - n * alpha) * s[k - 1]) / (n * (1.0 - alpha)), s[k - 1])


def naive_hs(loss, window, alphas) -> dict:
    T = len(loss)
    out = {a: (np.empty(T - window), np.empty(T - window)) for a in alphas}
    for t in range(window, T):
        win = loss[t - window:t]
        for a in alphas:
            out[a][0][t - window] = np.quantile(win, a, method="inverted_cdf")
            out[a][1][t - window] = _es(win, a)
    return out


def naive_parametric(R, w, window, alphas, zero_mean=True) -> dict:
    T = len(R)
    out = {a: (np.empty(T - window), np.empty(T - window)) for a in alphas}
    for t in range(window, T):
        win = R[t - window:t]
        sigma = float(np.sqrt(w @ np.cov(win, rowvar=False) @ w))
        loc = 0.0 if zero_mean else -float(win.mean(axis=0) @ w)
        for a in alphas:
            z = norm.ppf(a)
            out[a][0][t - window] = loc + sigma * z
            out[a][1][t - window] = loc + sigma * norm.pdf(z) / (1.0 - a)
    return out


def naive_mc_normal(R, w, window, alphas, zero_mean=True, n_draws=100_000, seed=0) -> dict:
    T, N = R.shape
    Z = np.random.default_rng(seed).standard_normal((n_draws, N))
    out = {a: (np.empty(T - window), np.empty(T - window)) for a in alphas}
    for t in range(window, T):
        win = R[t - window:t]
        L = np.linalg.cholesky(np.cov(win, rowvar=False))
        loss = -((Z @ L.T) @ w)
        loc = 0.0 if zero_mean else -float(win.mean(axis=0) @ w)
        for a in alphas:
            out[a][0][t - window] = loc + np.quantile(loss, a, method="inverted_cdf")
            out[a][1][t - window] = loc + _es(loss, a)
    return out
