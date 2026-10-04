"""M6: the three charts. Inputs are the results table and the SQL rolling-exception output."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABEL = {"hs": "Historical simulation", "parametric": "Parametric normal", "mc_normal": "Monte Carlo normal"}
COLOR = {"hs": "#1f4e79", "parametric": "#c55a11", "mc_normal": "#548235"}


def _sel(tbl, method, window, alpha):
    m = (tbl["method"] == method) & (tbl["window_days"] == window) & np.isclose(tbl["alpha"], alpha)
    return tbl[m].sort_values("date")


def returns_vs_var(tbl: pd.DataFrame, path: Path, window: int, alpha: float = 0.99) -> None:
    hs, pa = _sel(tbl, "hs", window, alpha), _sel(tbl, "parametric", window, alpha)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hs["date"], hs["realised_pnl"], lw=0.5, color="0.55", label="Portfolio daily return")
    ax.plot(hs["date"], -hs["var"], lw=1.2, color=COLOR["hs"], label=f"-VaR {alpha:.0%}, {LABEL['hs']}")
    ax.plot(pa["date"], -pa["var"], lw=1.2, color=COLOR["parametric"], label=f"-VaR {alpha:.0%}, {LABEL['parametric']}")
    ex = hs[hs["exception"]]
    ax.scatter(ex["date"], ex["realised_pnl"], s=22, color="#c00000", zorder=3, label=f"HS exceptions (n={len(ex)})")
    ax.axhline(0, lw=0.5, color="k")
    ax.set(title=f"{hs['portfolio'].iloc[0]}: daily return vs 1-day {alpha:.0%} VaR, {window}-day window", ylabel="Return")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def rolling_exceptions_chart(rolling: pd.DataFrame, path: Path) -> None:
    windows = sorted(rolling["window_days"].unique())
    fig, axes = plt.subplots(len(windows), 1, figsize=(11, 3.6 * len(windows)), sharex=True, squeeze=False)
    top = max(12, int(rolling["exceptions_250d"].max()) + 2)
    for ax, W in zip(axes[:, 0], windows):
        ax.axhspan(-0.5, 4.5, color="#70ad47", alpha=0.18)
        ax.axhspan(4.5, 9.5, color="#ffc000", alpha=0.22)
        ax.axhspan(9.5, top, color="#c00000", alpha=0.15)
        for method in LABEL:  # MC drawn last and dashed: it tracks parametric almost exactly
            g = rolling[(rolling["window_days"] == W) & (rolling["method"] == method)]
            ax.step(g["date"], g["exceptions_250d"], where="post", lw=1.3, color=COLOR[method], label=LABEL[method],
                    ls="--" if method == "mc_normal" else "-")
        ax.set(ylim=(-0.5, top), ylabel="Exceptions, trailing 250 days",
               title=f"99% VaR exceptions vs Basel traffic-light zones, {W}-day estimation window")
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def method_comparison(tbl: pd.DataFrame, path: Path, window: int, alpha: float = 0.99) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for method in LABEL:
        g = _sel(tbl, method, window, alpha)
        if len(g):
            ax.plot(g["date"], g["var"], lw=1.2, color=COLOR[method], label=LABEL[method],
                    ls="--" if method == "mc_normal" else "-")
    ax.set(title=f"1-day {alpha:.0%} VaR by method, {window}-day window", ylabel="VaR (fraction of portfolio value)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_all(tbl: pd.DataFrame, rolling: pd.DataFrame, out_dir: Path, portfolio: str) -> list[str]:
    W = int(min(tbl["window_days"]))
    paths = [out_dir / f"{portfolio}_{n}.png" for n in ("returns_vs_var", "rolling_exceptions", "var_method_comparison")]
    returns_vs_var(tbl, paths[0], W)
    rolling_exceptions_chart(rolling, paths[1])
    method_comparison(tbl, paths[2], W)
    return [p.name for p in paths]
