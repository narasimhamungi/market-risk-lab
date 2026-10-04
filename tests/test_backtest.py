import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, chi2, norm

from market_risk_lab import backtest as bt


@pytest.mark.parametrize("x,n,p", [(5, 250, 0.01), (1, 250, 0.01), (30, 1000, 0.025), (12, 500, 0.01)])
def test_kupiec_matches_binomial_likelihood_ratio(x, n, p):
    lr, pval = bt.kupiec_pof(x, n, p)
    ref = -2 * (binom.logpmf(x, n, p) - binom.logpmf(x, n, x / n))
    assert np.isclose(lr, ref) and np.isclose(pval, chi2.sf(ref, 1))


def test_kupiec_zero_and_all_exceptions():
    lr, pval = bt.kupiec_pof(0, 250, 0.01)
    assert np.isclose(lr, -2 * 250 * np.log(0.99)) and 0 < pval < 1
    lr_all, p_all = bt.kupiec_pof(250, 250, 0.01)
    assert np.isfinite(lr_all) and p_all < 1e-6
    with pytest.raises(ValueError):
        bt.kupiec_pof(5, 4, 0.01)


def test_kupiec_accepts_correct_model_rejects_miscalibrated():
    """Losses ~ N(0,1). A VaR built on the true sigma passes; one built on 0.7*sigma is rejected."""
    loss = np.random.default_rng(42).standard_normal(2000)
    z = norm.ppf(0.99)
    _, p_good = bt.kupiec_pof(int((loss > z).sum()), 2000, 0.01)
    _, p_bad = bt.kupiec_pof(int((loss > 0.7 * z).sum()), 2000, 0.01)
    assert p_good > 0.05 and p_bad < 0.01


def test_traffic_light_boundaries():
    zones = bt.traffic_light([0, 4, 5, 9, 10, 40, np.nan])
    assert list(zones) == ["green", "green", "yellow", "yellow", "red", "red", None]


def test_rolling_exception_count_matches_pandas():
    exc = np.random.default_rng(0).random(900) < 0.02
    got = bt.rolling_exception_count(exc, 250)
    ref = pd.Series(exc.astype(int)).rolling(250).sum().to_numpy()
    assert np.isnan(got[:249]).all() and np.array_equal(got[249:], ref[249:])
    assert np.isnan(bt.rolling_exception_count(exc[:100], 250)).all()


def _table(n=600, seed=1):
    rng = np.random.default_rng(seed)
    pnl = rng.standard_normal(n) * 0.01
    var = np.full(n, 0.01 * norm.ppf(0.99))
    return pd.DataFrame(dict(date=pd.bdate_range("2020-01-01", periods=n), portfolio="p", method="hs",
                             window_days=250, alpha=0.99, var=var, es=var * 1.15, realised_pnl=pnl,
                             exception=-pnl > var))


def test_backtest_summary_counts_and_es_check():
    tbl = _table()
    row = bt.backtest_summary(tbl).iloc[0]
    hits = tbl[tbl["exception"]]
    assert row["n_obs"] == 600 and row["exceptions"] == len(hits)
    assert np.isclose(row["exception_rate"], len(hits) / 600) and np.isclose(row["expected_exceptions"], 6.0)
    assert np.isclose(row["mean_loss_on_exceptions"], -hits["realised_pnl"].mean())
    assert np.isclose(row["es_ratio"], -hits["realised_pnl"].mean() / hits["es"].mean())


def test_backtest_summary_with_no_exceptions():
    tbl = _table().assign(var=1.0, es=1.1, exception=False)
    row = bt.backtest_summary(tbl).iloc[0]
    assert row["exceptions"] == 0 and np.isnan(row["es_ratio"]) and np.isfinite(row["kupiec_lr"])


def test_traffic_light_table_and_summary():
    tbl = _table()
    tl = bt.traffic_light_table(tbl)
    assert len(tl) == 600 - 249 and set(tl["zone"]) <= {"green", "yellow", "red"}
    s = bt.traffic_light_summary(tl).iloc[0]
    assert np.isclose(s["green_share"] + s["yellow_share"] + s["red_share"], 1.0)
    assert s["max_exceptions_250d"] == tl["exceptions_250d"].max()
