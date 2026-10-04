"""M4: exception counting, Kupiec POF, Basel traffic light, ES sanity check."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlog1py, xlogy
from scipy.stats import chi2

BASEL_WINDOW = 250
GREEN_MAX, YELLOW_MAX = 4, 9  # 99% VaR, 250 observations


def kupiec_pof(x: int, n: int, p: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures LR test of H0: exception probability = p.

    LR = -2 [ln L(p) - ln L(x/n)] ~ chi2(1) under H0. xlogy/xlog1py define
    0*log(0) = 0, which is exactly the x = 0 (and x = n) limit of the
    likelihood, so no special-casing is needed. Unconditional coverage only:
    it says nothing about whether exceptions cluster.
    """
    if not 0 <= x <= n or n <= 0:
        raise ValueError("need 0 <= x <= n, n > 0")
    phat = x / n
    ll0 = xlogy(x, p) + xlog1py(n - x, -p)
    ll1 = xlogy(x, phat) + xlog1py(n - x, -phat)
    lr = max(-2.0 * float(ll0 - ll1), 0.0)
    return lr, float(chi2.sf(lr, 1))


def rolling_exception_count(exc: np.ndarray, window: int = BASEL_WINDOW) -> np.ndarray:
    """Trailing count including the current day; NaN until a full window exists."""
    c = np.concatenate(([0], np.cumsum(np.asarray(exc, dtype=np.int64))))
    out = np.full(len(exc), np.nan)
    if len(exc) >= window:
        out[window - 1:] = c[window:] - c[:-window]
    return out


def traffic_light(count) -> np.ndarray:
    """Basel zones for exceptions in 250 days at 99%: green 0-4, yellow 5-9, red 10+."""
    count = np.asarray(count, dtype=float)
    zone = np.select([np.isnan(count), count <= GREEN_MAX, count <= YELLOW_MAX],
                     [None, "green", "yellow"], default="red")
    return zone.astype(object)


def backtest_summary(tbl: pd.DataFrame) -> pd.DataFrame:
    """One row per (portfolio, method, window_days, alpha).

    The ES columns compare the mean realised loss on exception days with the
    mean ES predicted for those same days. A ratio near 1 is consistent with
    the tail being sized correctly; it is a sanity check on a handful of
    observations, not a formal ES backtest.
    """
    rows = []
    for key, g in tbl.groupby(["portfolio", "method", "window_days", "alpha"], sort=True):
        n, x, p = len(g), int(g["exception"].sum()), 1.0 - key[3]
        lr, pval = kupiec_pof(x, n, p)
        hit = g[g["exception"]]
        mean_loss = float(-hit["realised_pnl"].mean()) if x else np.nan
        mean_es = float(hit["es"].mean()) if x else np.nan
        rows.append(dict(zip(["portfolio", "method", "window_days", "alpha"], key),
                         n_obs=n, exceptions=x, expected_exceptions=n * p,
                         exception_rate=x / n, expected_rate=p,
                         kupiec_lr=lr, kupiec_p=pval, reject_5pct=pval < 0.05,
                         mean_loss_on_exceptions=mean_loss, mean_es_on_exceptions=mean_es,
                         es_ratio=mean_loss / mean_es if x else np.nan))
    return pd.DataFrame(rows)


def traffic_light_table(tbl: pd.DataFrame, alpha: float = 0.99) -> pd.DataFrame:
    """Rolling 250-day exception count and zone per (portfolio, method, window_days) at 99%."""
    out = []
    sel = tbl[np.isclose(tbl["alpha"], alpha)].sort_values("date")
    for key, g in sel.groupby(["portfolio", "method", "window_days"], sort=True):
        cnt = rolling_exception_count(g["exception"].to_numpy())
        ok = ~np.isnan(cnt)
        out.append(pd.DataFrame(dict(date=g["date"].to_numpy()[ok], portfolio=key[0], method=key[1],
                                     window_days=key[2], exceptions_250d=cnt[ok].astype(int),
                                     zone=traffic_light(cnt[ok]))))
    return pd.concat(out, ignore_index=True)


def traffic_light_summary(tl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in tl.groupby(["portfolio", "method", "window_days"], sort=True):
        share = g["zone"].value_counts(normalize=True)
        rows.append(dict(zip(["portfolio", "method", "window_days"], key), n_windows=len(g),
                         green_share=float(share.get("green", 0.0)), yellow_share=float(share.get("yellow", 0.0)),
                         red_share=float(share.get("red", 0.0)), max_exceptions_250d=int(g["exceptions_250d"].max()),
                         latest_zone=g.sort_values("date")["zone"].iloc[-1]))
    return pd.DataFrame(rows)
