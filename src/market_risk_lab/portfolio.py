"""M2: fixed-weight portfolios from YAML.

Fixed weights applied to every day's simple returns means the portfolio is
rebalanced back to target at each close. That is an assumption, not a
description of a real book: it ignores drift and transaction costs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not cfg.get("portfolios"):
        raise ValueError("config has no portfolios")
    return cfg


def resolve_weights(pcfg: dict) -> pd.Series:
    """Weights indexed by ticker. `weights: equal` or a {ticker: weight} mapping. Long-only, sum to 1."""
    w = pcfg.get("weights", "equal")
    if isinstance(w, str):
        if w != "equal":
            raise ValueError(f"unknown weights scheme {w!r}")
        tickers = list(pcfg["tickers"])
        s = pd.Series(1.0 / len(tickers), index=tickers)
    else:
        s = pd.Series(w, dtype=float)
    if s.index.duplicated().any():
        raise ValueError("duplicate tickers in portfolio")
    if (s < 0).any():
        raise ValueError("negative weight: long-only portfolios only")
    if not np.isclose(s.sum(), 1.0, atol=1e-9):
        raise ValueError(f"weights sum to {s.sum():.6f}, not 1")
    return s


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    missing = set(weights.index) - set(returns.columns)
    if missing:
        raise ValueError(f"no returns for {sorted(missing)}")
    return pd.Series(returns[weights.index].to_numpy() @ weights.to_numpy(), index=returns.index, name="ret")
