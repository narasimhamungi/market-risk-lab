"""Vectorised vs one-date-at-a-time timing on the data actually loaded.
Timings are machine-dependent; the summary records what this run measured."""
from __future__ import annotations

import time

import numpy as np

from . import naive, risk


def _best(fn, repeats: int) -> tuple[float, object]:
    best, res = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t0)
    return best, res


def run_benchmark(R: np.ndarray, w: np.ndarray, windows, alphas, zero_mean=True, repeats: int = 3) -> list[dict]:
    loss = -(R @ w)
    out = []
    for W in windows:
        cases = {
            "hs": (lambda: risk.rolling_hs(loss, W, alphas), lambda: naive.naive_hs(loss, W, alphas)),
            "parametric": (lambda: risk.rolling_parametric(*risk.rolling_moments(R, W), w, alphas, zero_mean),
                           lambda: naive.naive_parametric(R, w, W, alphas, zero_mean)),
        }
        for method, (vec, slow) in cases.items():
            t_vec, a = _best(vec, repeats)
            t_naive, b = _best(slow, 1)
            diff = max(float(np.max(np.abs(a[al][i] - b[al][i]))) for al in alphas for i in (0, 1))
            out.append(dict(method=method, window_days=W, n_forecasts=len(R) - W, vectorised_s=t_vec,
                            naive_s=t_naive, speedup=t_naive / t_vec, max_abs_diff=diff))
    return out
