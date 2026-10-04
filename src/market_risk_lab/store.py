"""M5: results store. Same Postgres as the lakehouse gold layer, separate schema."""
from __future__ import annotations

import os

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from .data import SQL_DIR, fetch_df

COLUMNS = ["date", "portfolio", "method", "window_days", "alpha", "var", "es", "realised_pnl", "exception"]


def connect():
    """MRL_DSN if set, otherwise the lakehouse's MDL_DB_* variables. The defaults are
    the local docker-compose development values published in marketdata-lakehouse."""
    if os.environ.get("MRL_DSN"):
        return psycopg2.connect(os.environ["MRL_DSN"])
    return psycopg2.connect(
        host=os.environ.get("MDL_DB_HOST", "localhost"), port=int(os.environ.get("MDL_DB_PORT", "5432")),
        dbname=os.environ.get("MDL_DB_NAME", "marketdata_lakehouse"),
        user=os.environ.get("MDL_DB_USER", "mdl"), password=os.environ.get("MDL_DB_PASSWORD", "mdl_local_dev"))


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute((SQL_DIR / "risk_results_ddl.sql").read_text())
    conn.commit()


def write_results(conn, tbl: pd.DataFrame) -> int:
    """Replace each portfolio's rows in one transaction, so a re-run is idempotent
    and a failed run leaves the previous results intact."""
    rows = [(d.date(), p, m, int(w), float(a), float(v), float(e), float(r), bool(x))
            for d, p, m, w, a, v, e, r, x in tbl[COLUMNS].itertuples(index=False, name=None)]
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM risk.risk_results WHERE portfolio = ANY(%s)",
                        (sorted(tbl["portfolio"].unique()),))
            execute_values(cur, f"INSERT INTO risk.risk_results ({', '.join(COLUMNS)}) VALUES %s", rows,
                           page_size=5000)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rows)


def rolling_exceptions(conn, portfolio: str) -> pd.DataFrame:
    df = fetch_df(conn, (SQL_DIR / "rolling_exceptions.sql").read_text(), dict(portfolio=portfolio))
    df["date"] = pd.to_datetime(df["date"])
    df["exceptions_250d"] = df["exceptions_250d"].astype(int)
    return df
