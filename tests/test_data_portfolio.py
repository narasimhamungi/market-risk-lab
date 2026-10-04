import numpy as np
import pandas as pd
import pytest

from market_risk_lab.data import DataValidationError, build_returns
from market_risk_lab.portfolio import portfolio_returns, resolve_weights

from .conftest import long_prices

T3 = ["AAA", "BBB", "CCC"]
KW = dict(min_history_days=10)


def test_clean_panel_gives_simple_returns(wide_prices):
    rets, rep = build_returns(long_prices(wide_prices), T3, **KW)
    assert rets.shape == (59, 3) and rep.issues == []
    assert np.isclose(rets.iloc[0, 0], wide_prices.iloc[1, 0] / wide_prices.iloc[0, 0] - 1)


def test_missing_price_drops_both_affected_returns_and_logs(wide_prices):
    d = wide_prices.index[20]
    px = long_prices(wide_prices)
    px = px[~((px["date"] == d) & (px["ticker"] == "BBB"))]
    rets, rep = build_returns(px, T3, max_missing_frac=0.05, **KW)
    assert len(rets) == 57 and d not in rets.index and wide_prices.index[21] not in rets.index
    assert rep.counts() == {"missing_price": 2}
    assert not rets.isna().any().any()


def test_excess_missing_fails(wide_prices):
    px = long_prices(wide_prices)
    px = px[~((px["ticker"] == "BBB") & px["date"].isin(wide_prices.index[10:20]))]
    with pytest.raises(DataValidationError, match="missing_excess"):
        build_returns(px, T3, **KW)


@pytest.mark.parametrize("mutate,match", [
    (lambda px: pd.concat([px, px.iloc[[5]]]), "structural"),                     # duplicate key
    (lambda px: px.assign(adj_close=px["adj_close"].where(px.index != 7, -1.0)), "structural"),
    (lambda px: px[px["ticker"] != "CCC"], "structural"),                         # ticker absent
])
def test_structural_problems_fail(wide_prices, mutate, match):
    with pytest.raises(DataValidationError, match=match):
        build_returns(mutate(long_prices(wide_prices)), T3, **KW)


def test_duplicates_are_reported_once_per_ticker(wide_prices):
    px = long_prices(wide_prices)
    px = pd.concat([px, px[px["ticker"].isin(["AAA", "CCC"])]])  # every date doubled for two tickers
    with pytest.raises(DataValidationError, match=r"2 structural.*duplicate_key\[AAA\].*60 dates"):
        build_returns(px, T3, **KW)


def test_stale_price_fails(wide_prices):
    wide_prices.iloc[30:34, 1] = wide_prices.iloc[30, 1]  # 3 consecutive zero returns
    with pytest.raises(DataValidationError, match="stale_price"):
        build_returns(long_prices(wide_prices), T3, **KW)


def test_outlier_is_flagged_and_kept(wide_prices):
    wide_prices.iloc[40:, 0] *= 0.6  # one -40% day
    rets, rep = build_returns(long_prices(wide_prices, flag="single_source"), T3, **KW)
    assert len(rets) == 59 and rep.counts() == {"outlier": 1}
    assert rep.issues[0]["action"] == "kept" and "single_source" in rep.issues[0]["detail"]
    assert np.isclose(rets.iloc[39, 0], wide_prices.iloc[40, 0] / wide_prices.iloc[39, 0] - 1)


def test_calendar_gap_drops_the_multi_day_return(wide_prices):
    px = long_prices(wide_prices.drop(wide_prices.index[25:31]))  # whole universe missing for 6 sessions
    rets, rep = build_returns(px, T3, **KW)
    assert rep.counts() == {"calendar_gap": 1} and len(rets) == 52
    assert wide_prices.index[31] not in rets.index


def test_trading_day_without_any_price_is_missing_not_ignored(wide_prices):
    """With the gold calendar supplied, a date absent for every ticker is still accounted for."""
    px = long_prices(wide_prices.drop(wide_prices.index[25]))
    rets, rep = build_returns(px, T3, calendar=wide_prices.index, max_missing_frac=0.05, **KW)
    assert rep.counts() == {"missing_price": 2} and len(rets) == 57


def test_short_history_fails(wide_prices):
    with pytest.raises(DataValidationError, match="history_short"):
        build_returns(long_prices(wide_prices), T3, min_history_days=500)


def test_weights():
    w = resolve_weights(dict(weights="equal", tickers=list("ABCD")))
    assert list(w.index) == list("ABCD") and np.allclose(w, 0.25)
    assert resolve_weights(dict(weights={"A": 0.6, "B": 0.4})).to_dict() == {"A": 0.6, "B": 0.4}
    for bad in ({"A": 0.6, "B": 0.5}, {"A": 1.2, "B": -0.2}):
        with pytest.raises(ValueError):
            resolve_weights(dict(weights=bad))
    with pytest.raises(ValueError):
        resolve_weights(dict(weights="risk_parity", tickers=["A"]))


def test_portfolio_return_is_weighted_sum(wide_prices):
    rets, _ = build_returns(long_prices(wide_prices), T3, **KW)
    w = resolve_weights(dict(weights={"CCC": 0.5, "AAA": 0.5}))
    assert np.allclose(portfolio_returns(rets, w), 0.5 * rets["CCC"] + 0.5 * rets["AAA"])
    with pytest.raises(ValueError):
        portfolio_returns(rets, resolve_weights(dict(weights={"ZZZ": 1.0})))
