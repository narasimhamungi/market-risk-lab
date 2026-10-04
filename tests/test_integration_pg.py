"""End to end against a real Postgres holding the lakehouse gold schema.
The prices are seeded synthetic test input; nothing here is a result."""
import json

import pandas as pd
import psycopg2
import pytest
import yaml

from market_risk_lab import backtest, run as runner, store
from market_risk_lab.data import fetch_df

from .conftest import TICKERS

pytestmark = pytest.mark.integration
WINDOWS, ALPHAS = [250, 500], [0.99, 0.975]


@pytest.fixture
def cfg_path(tmp_path):
    cfg = dict(data=dict(start_date="2019-01-01", end_date=None, min_history_days=900),
               validation=dict(max_missing_frac=0.01, max_calendar_gap_days=5, stale_run_days=3, outlier_sigma=5.0),
               risk=dict(windows=WINDOWS, alphas=ALPHAS, zero_mean=True, mc_draws=20_000, mc_seed=1),
               portfolios=dict(test_ew=dict(description="fixture", weights="equal", tickers=TICKERS)))
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_pipeline_end_to_end(gold, cfg_path, tmp_path):
    conn, out = gold["conn"], tmp_path / "out"
    summary = runner.run(cfg_path, out, conn=conn, benchmark=True)["portfolios"]["test_ew"]

    n = gold["n_dates"] - 1 - 2  # first day has no return; the removed price costs two returns
    assert summary["n_returns"] == n and summary["validation_counts"]["missing_price"] == 2
    expected_rows = sum(n - w for w in WINDOWS) * 3 * len(ALPHAS)
    assert summary["results_rows_written"] == expected_rows

    db = fetch_df(conn, "SELECT count(*) AS n, min(date) AS d0, bool_and(es >= var) AS ok FROM risk.risk_results", {})
    assert db["n"][0] == expected_rows and db["ok"][0]

    # SQL window-function output equals the numpy rolling count, read back from the store
    stored = fetch_df(conn, "SELECT date, portfolio, method, window_days, alpha::float8 AS alpha, exception "
                            "FROM risk.risk_results", {})
    stored["date"] = pd.to_datetime(stored["date"])
    ref = backtest.traffic_light_table(stored).sort_values(["method", "window_days", "date"]).reset_index(drop=True)
    sql = store.rolling_exceptions(conn, "test_ew")
    assert len(sql) == len(ref) == sum(n - w - 249 for w in WINDOWS) * 3
    assert (sql["exceptions_250d"].to_numpy() == ref["exceptions_250d"].to_numpy()).all()
    assert (sql["zone"].to_numpy() == ref["zone"].to_numpy()).all()

    assert all((out / c).stat().st_size > 10_000 for c in summary["charts"]) and len(summary["charts"]) == 3
    saved = json.loads((out / "summary.json").read_text())["portfolios"]["test_ew"]
    assert len(saved["backtest"]) == 3 * len(WINDOWS) * len(ALPHAS)
    assert all(b["speedup"] > 1 and b["max_abs_diff"] < 1e-12 for b in saved["benchmark"])

    # idempotent: a second run replaces, never appends
    runner.run(cfg_path, out, conn=conn, benchmark=False)
    assert fetch_df(conn, "SELECT count(*) AS n FROM risk.risk_results", {})["n"][0] == expected_rows


def test_store_rejects_rows_that_break_the_sign_convention(gold):
    conn = gold["conn"]
    store.ensure_schema(conn)
    bad = [("2020-01-02", "p", "hs", 250, 0.99, 0.02, 0.01, -0.01, False),   # es < var
           ("2020-01-02", "p", "hs", 250, 0.99, 0.02, 0.03, -0.05, False)]   # loss 5% > VaR 2% but not flagged
    for row in bad:
        with pytest.raises(psycopg2.errors.CheckViolation):
            with conn.cursor() as cur:
                cur.execute("INSERT INTO risk.risk_results VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)", row)
        conn.rollback()


def test_data_layer_reads_only_requested_tickers(gold):
    from market_risk_lab.data import pull_prices
    prices, cal = pull_prices(gold["conn"], TICKERS[:3], "2020-01-01", "2020-12-31")
    assert set(prices["ticker"]) == set(TICKERS[:3]) and prices["adj_close"].dtype == float
    assert prices["date"].min() >= pd.Timestamp("2020-01-01") and len(cal) == prices["date"].nunique()
