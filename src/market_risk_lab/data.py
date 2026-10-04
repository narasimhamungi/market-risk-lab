"""M1: pull adjusted close from the gold layer, validate, build simple returns.

Nothing is repaired silently. Hard problems raise DataValidationError after
every issue has been logged; soft problems drop a return row and say so.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("market_risk_lab.data")
SQL_DIR = Path(__file__).parent / "sql"

DEFAULTS = dict(max_missing_frac=0.01, max_calendar_gap_days=5, stale_run_days=3,
                outlier_sigma=5.0, min_history_days=1250)


class DataValidationError(RuntimeError):
    pass


@dataclass
class ValidationReport:
    issues: list[dict] = field(default_factory=list)

    def add(self, check, severity, action, detail, ticker=None, date=None):
        rec = dict(check=check, severity=severity, action=action, ticker=ticker,
                   date=None if date is None else str(pd.Timestamp(date).date()), detail=detail)
        self.issues.append(rec)
        (log.error if severity == "fail" else log.warning)("%s", rec)

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for i in self.issues:
            out[i["check"]] = out.get(i["check"], 0) + 1
        return out

    def failures(self) -> list[dict]:
        return [i for i in self.issues if i["severity"] == "fail"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.issues, columns=["check", "severity", "action", "ticker", "date", "detail"])


def fetch_df(conn, sql: str, params: dict) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])


def pull_prices(conn, tickers, start, end=None) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Long prices (date, ticker, adj_close, primary_source, reconciliation_flag) + trading calendar."""
    p = dict(tickers=list(tickers), start=start, end=end)
    prices = fetch_df(conn, (SQL_DIR / "adj_close.sql").read_text(), p)
    cal = fetch_df(conn, (SQL_DIR / "trading_calendar.sql").read_text(), p)
    prices["date"] = pd.to_datetime(prices["date"])
    return prices, pd.DatetimeIndex(pd.to_datetime(cal["date"]))


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """(length, end_position) of the longest run of True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if starts.size == 0:
        return 0, -1
    i = int(np.argmax(ends - starts))
    return int(ends[i] - starts[i]), int(ends[i] - 1)


def build_returns(prices: pd.DataFrame, tickers, calendar: pd.DatetimeIndex | None = None,
                  **cfg) -> tuple[pd.DataFrame, ValidationReport]:
    """Validated wide simple-return panel (dates x tickers), columns in `tickers` order."""
    c = {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
    rep = ValidationReport()
    tickers = list(tickers)

    # --- structural checks: nothing downstream is meaningful if these fail
    absent = sorted(set(tickers) - set(prices["ticker"]))
    for t in absent:
        rep.add("ticker_absent", "fail", "abort", "no rows in fact_price_daily_consensus", ticker=t)
    def _span(g):
        return f"{g['date'].nunique()} dates, {g['date'].min().date()}..{g['date'].max().date()}"

    dup = prices[prices.duplicated(["date", "ticker"], keep=False)]
    for t, g in dup.groupby("ticker"):
        rep.add("duplicate_key", "fail", "abort", f"more than one row per date on {_span(g)}", ticker=t)
    bad = prices[~(prices["adj_close"] > 0)]  # also catches NaN
    for t, g in bad.groupby("ticker"):
        rep.add("non_positive_price", "fail", "abort", f"adj_close <= 0 or null on {_span(g)}", ticker=t)
    if rep.failures():
        raise DataValidationError(f"{len(rep.failures())} structural data failure(s): "
                                  + "; ".join(f"{i['check']}[{i['ticker']}] {i['detail']}" for i in rep.failures()))

    wide = prices.pivot(index="date", columns="ticker", values="adj_close")[tickers].sort_index()

    # --- common history: a portfolio return needs every asset
    first, last = wide.apply(pd.Series.first_valid_index), wide.apply(pd.Series.last_valid_index)
    start, end = first.max(), last.min()
    for t in tickers:
        if first[t] > wide.index[0] or last[t] < wide.index[-1]:
            rep.add("history_truncated", "warn", "panel limited to common range",
                    f"{t} covers {first[t].date()}..{last[t].date()}; panel {start.date()}..{end.date()}", t)
    cal = wide.index if calendar is None else calendar
    cal = cal[(cal >= start) & (cal <= end)]
    for d in wide.loc[start:end].index.difference(cal):
        rep.add("off_calendar", "fail", "abort", "price on a date dim_date does not mark as trading day", date=d)
    wide = wide.reindex(cal)

    # --- missing prices: never forward-filled (a filled price is a fake zero return)
    miss = wide.isna()
    for t in tickers:
        frac = float(miss[t].mean())
        if frac > c["max_missing_frac"]:
            rep.add("missing_excess", "fail", "abort",
                    f"{frac:.2%} of trading days missing > {c['max_missing_frac']:.2%}", t)

    rets = (wide / wide.shift(1) - 1.0).iloc[1:]

    # --- calendar gaps: dim_date.is_trading_day is derived from the data itself, so a
    # universe-wide ingestion hole is invisible to it; a long gap is the only symptom.
    gap_days = pd.Series(cal[1:] - cal[:-1], index=cal[1:]).dt.days
    drop = pd.Series(False, index=rets.index)
    for d, g in gap_days[gap_days > c["max_calendar_gap_days"]].items():
        rep.add("calendar_gap", "warn", "return dropped", f"{g} calendar days since previous trading day", date=d)
        drop[d] = True
    for d, row in rets[rets.isna().any(axis=1)].iterrows():
        rep.add("missing_price", "warn", "return dropped",
                "no price on this or the previous trading day", ",".join(row.index[row.isna()]), d)
        drop[d] = True
    rets = rets[~drop]

    # --- stale prices
    for t in tickers:
        n, end_pos = _longest_true_run((rets[t] == 0.0).to_numpy())
        if n >= c["stale_run_days"]:
            rep.add("stale_price", "fail", "abort",
                    f"adj_close unchanged for {n} consecutive returns", t, rets.index[end_pos])

    # --- outliers: QA diagnostic on full-sample moments (not a model input, so no
    # look-ahead issue). Flagged, never removed: deleting tail days from a risk model's
    # input is how VaR gets understated. The reconciliation flag says whether a second
    # vendor confirms the price.
    flags = prices.set_index(["date", "ticker"])["reconciliation_flag"]
    zs = (rets - rets.mean()) / rets.std(ddof=1)
    for i, j in np.argwhere(zs.abs().to_numpy() > c["outlier_sigma"]):  # not .stack(): its NaN handling differs across pandas versions
        d, t = rets.index[i], tickers[j]
        fl = flags.get((d, t), "unknown")
        rep.add("outlier", "warn", "kept",
                f"return {rets.at[d, t]:+.4f} = {zs.at[d, t]:+.1f} sigma; reconciliation_flag={fl}", t, d)
    used = prices[prices["date"].isin(wide.index)]
    for t, n in used[used["reconciliation_flag"] == "disagreed"].groupby("ticker").size().items():
        rep.add("vendor_disagreement", "warn", "kept",
                f"{n} prices where yfinance and Tiingo differ beyond tolerance", t)

    if len(rets) < c["min_history_days"]:
        rep.add("history_short", "fail", "abort", f"{len(rets)} returns < {c['min_history_days']} required")
    if rep.failures():
        raise DataValidationError(f"{len(rep.failures())} data failure(s): "
                                  + "; ".join(f"{i['check']}[{i['ticker']}] {i['detail']}" for i in rep.failures()))
    log.info("returns panel: %d days x %d assets, %s..%s", len(rets), len(tickers),
             rets.index[0].date(), rets.index[-1].date())
    return rets, rep
