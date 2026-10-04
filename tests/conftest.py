import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from psycopg2.extras import execute_values

from market_risk_lab import store

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def correlated_returns(T, N, seed, vol=0.012, rho=0.3, df=None):
    """Seeded synthetic returns. Test input only: never written to outputs/."""
    rng = np.random.default_rng(seed)
    cov = vol ** 2 * (rho * np.ones((N, N)) + (1 - rho) * np.eye(N))
    z = rng.standard_normal((T, N)) if df is None else rng.standard_t(df, (T, N)) * np.sqrt((df - 2) / df)
    return z @ np.linalg.cholesky(cov).T, cov


def long_prices(wide: pd.DataFrame, flag="agreed") -> pd.DataFrame:
    df = wide.rename_axis(index="date", columns="ticker").stack().rename("adj_close").reset_index()
    return df.assign(primary_source="yfinance", reconciliation_flag=flag)


@pytest.fixture
def wide_prices():
    dates = pd.bdate_range("2022-01-03", periods=60)
    R, _ = correlated_returns(60, 3, seed=1)
    return pd.DataFrame(100 * np.cumprod(1 + R, axis=0), index=dates, columns=["AAA", "BBB", "CCC"])


@pytest.fixture(scope="session")
def pg_conn():
    """Real Postgres, not a mock: SQL correctness is the thing under test. Refuses any
    database whose name does not end in _test because the fixture truncates gold tables."""
    required = os.environ.get("MRL_REQUIRE_DB") == "1"
    try:
        conn = store.connect()
    except Exception as e:  # noqa: BLE001
        if required:
            raise
        pytest.skip(f"no Postgres available: {e}")
    with conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        name = cur.fetchone()[0]
    if not name.endswith("_test"):
        conn.close()
        if required:
            raise RuntimeError(f"refusing to run integration tests against {name!r}")
        pytest.skip(f"database {name!r} is not a *_test database")
    yield conn
    conn.close()


@pytest.fixture
def gold(pg_conn):
    """Lakehouse gold tables filled with seeded synthetic prices, one price removed on purpose."""
    dates = pd.bdate_range("2019-01-02", "2022-12-30")
    R, _ = correlated_returns(len(dates), len(TICKERS), seed=7, df=5)
    px = 100 * np.cumprod(1 + R, axis=0)
    with pg_conn.cursor() as cur:
        cur.execute((Path(__file__).parent / "gold_schema_fixture.sql").read_text())
        cur.execute("TRUNCATE dim_security, dim_date RESTART IDENTITY CASCADE")
        cur.execute("DROP SCHEMA IF EXISTS risk CASCADE")
        execute_values(cur, "INSERT INTO dim_date VALUES %s", [
            (int(d.strftime("%Y%m%d")), d.date(), d.year, d.quarter, d.month, d.month_name(), d.day,
             d.dayofweek, d.day_name(), True, True) for d in dates])
        execute_values(cur, "INSERT INTO dim_security (ticker, effective_from) VALUES %s",
                       [(t, dates[0].date()) for t in TICKERS])
        cur.execute("SELECT ticker, security_key FROM dim_security")
        key = dict(cur.fetchall())
        missing = (TICKERS[2], dates[700])
        execute_values(cur, "INSERT INTO fact_price_daily_consensus VALUES %s", [
            (key[t], int(d.strftime("%Y%m%d")), round(float(px[i, j]), 6), "yfinance", ["yfinance"], None, "single_source")
            for i, d in enumerate(dates) for j, t in enumerate(TICKERS) if (t, d) != missing], page_size=5000)
    pg_conn.commit()
    return dict(conn=pg_conn, n_dates=len(dates))
