"""M3: rolling 1-day VaR / ES by historical simulation, parametric normal and
Monte Carlo multivariate normal. No Python loop over dates.

Alignment (the no-look-ahead rule): with T returns and window W, forecast i
applies to day W+i and is built from returns [i, W+i-1]. sliding_window_view
yields T-W+1 windows; the last one has no realised day to be compared with
and is discarded ([:-1]).

Quantile convention: VaR_alpha = inf{x : F_n(x) >= alpha}, i.e. order
statistic k = ceil(n*alpha) of the losses (numpy's method="inverted_cdf").
W=250, alpha=0.99 -> the 3rd-worst loss. ES is the exact average of the upper
(1-alpha) tail, fractional weight on the k-th loss included, so ES >= VaR at
the same alpha by construction.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

METHODS = ("hs", "parametric", "mc_normal")


def tail_index(n: int, alpha: float) -> int:
    """1-based k = ceil(n*alpha). Rounded first: 1000*0.99 is 990.0000000000001 in floats."""
    return int(math.ceil(round(n * alpha, 9)))


def empirical_var_es(sorted_losses: np.ndarray, alpha: float):
    """VaR and ES along the last axis. Input must be ascending, or at least
    partitioned so that position k-1 is in place and everything after it is >= it."""
    n = sorted_losses.shape[-1]
    k = tail_index(n, alpha)
    var = sorted_losses[..., k - 1]
    tail = sorted_losses[..., k:].sum(axis=-1) + (k - n * alpha) * var
    es = tail / (n * (1.0 - alpha))
    return var, np.maximum(es, var)  # maximum() only absorbs last-bit rounding


def rolling_hs(loss: np.ndarray, window: int, alphas) -> dict:
    win = np.sort(sliding_window_view(loss, window)[:-1], axis=1)  # (T-W, W)
    return {a: empirical_var_es(win, a) for a in alphas}


def rolling_moments(R: np.ndarray, window: int):
    """Rolling sample mean (T-W, N) and covariance (T-W, N, N) of asset returns."""
    X = sliding_window_view(R, window, axis=0)[:-1]  # (T-W, N, W), a view
    mu = X.mean(axis=2)
    Xc = X - mu[:, :, None]
    return mu, Xc @ Xc.transpose(0, 2, 1) / (window - 1)


def rolling_parametric(mu, cov, w, alphas, zero_mean: bool = True) -> dict:
    """Normal VaR/ES from the rolling covariance. zero_mean drops the location
    term (a 250-day mean return is mostly noise); the covariance stays centred."""
    sigma = np.sqrt(np.einsum("i,tij,j->t", w, cov, w))
    loc = np.zeros_like(sigma) if zero_mean else -(mu @ w)
    out = {}
    for a in alphas:
        z = norm.ppf(a)
        out[a] = (loc + sigma * z, loc + sigma * norm.pdf(z) / (1.0 - a))
    return out


def rolling_mc_normal(mu, cov, w, alphas, zero_mean: bool = True, n_draws: int = 100_000,
                      seed: int = 0, chunk: int = 64) -> dict:
    """Monte Carlo under N(mu_t, cov_t), same seeded draws for every date
    (common random numbers, so day-to-day changes are not simulation noise).

    Asset returns r = L_t z with L_t the Cholesky factor, so portfolio P&L is
    (L_t' w) . z: one matmul per block of dates. The block loop bounds memory
    (chunk x n_draws floats); it is not a per-date loop.

    For a linear portfolio this converges to rolling_parametric as n_draws
    grows. It is here as a check on the plumbing and as the hook for
    non-linear positions or non-normal draws, not as an independent model.
    """
    n_dates, n_assets = mu.shape
    Z = np.random.default_rng(seed).standard_normal((n_draws, n_assets))
    A = np.einsum("tji,j->ti", np.linalg.cholesky(cov), w)  # rows are L_t' w
    loc = np.zeros(n_dates) if zero_mean else -(mu @ w)
    kth = sorted({tail_index(n_draws, a) - 1 for a in alphas})
    out = {a: (np.empty(n_dates), np.empty(n_dates)) for a in alphas}
    for s in range(0, n_dates, chunk):
        sl = slice(s, s + chunk)
        loss = -(A[sl] @ Z.T)  # (c, n_draws)
        loss.partition(kth, axis=1)
        for a in alphas:
            var, es = empirical_var_es(loss, a)
            out[a][0][sl], out[a][1][sl] = loc[sl] + var, loc[sl] + es
    return out


def compute_risk(R: np.ndarray, w: np.ndarray, windows, alphas, zero_mean: bool = True,
                 mc_draws: int = 100_000, mc_seed: int = 0, methods=METHODS) -> dict:
    """{(method, window): {alpha: (var, es)}}, each array of length T-window."""
    R, w = np.asarray(R, float), np.asarray(w, float)
    loss = -(R @ w)
    out = {}
    for W in windows:
        if W >= len(R):
            raise ValueError(f"window {W} needs more than {len(R)} returns")
        if "hs" in methods:
            out[("hs", W)] = rolling_hs(loss, W, alphas)
        if "parametric" in methods or "mc_normal" in methods:
            mu, cov = rolling_moments(R, W)
            if "parametric" in methods:
                out[("parametric", W)] = rolling_parametric(mu, cov, w, alphas, zero_mean)
            if "mc_normal" in methods:
                out[("mc_normal", W)] = rolling_mc_normal(mu, cov, w, alphas, zero_mean, mc_draws, mc_seed)
    return out


def risk_table(returns: pd.DataFrame, weights: pd.Series, portfolio: str, windows, alphas, **kw) -> pd.DataFrame:
    """Long results table in the risk_results layout."""
    R = returns[weights.index].to_numpy()
    w = weights.to_numpy()
    port = R @ w
    frames = []
    for (method, W), by_alpha in compute_risk(R, w, windows, alphas, **kw).items():
        for a, (var, es) in by_alpha.items():
            pnl = port[W:]
            frames.append(pd.DataFrame(dict(
                date=returns.index[W:], portfolio=portfolio, method=method, window_days=W,
                alpha=a, var=var, es=es, realised_pnl=pnl, exception=-pnl > var)))
    return pd.concat(frames, ignore_index=True)
