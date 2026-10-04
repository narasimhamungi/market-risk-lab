"""M6: one command regenerates everything.

    python -m market_risk_lab.run

gold layer -> validated returns -> VaR/ES -> backtests -> risk.risk_results ->
SQL rolling exceptions -> charts + outputs/summary.json. Every number quoted
anywhere about this project must come out of summary.json or the CSVs.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import backtest, charts, store
from .benchmark import run_benchmark
from .data import build_returns, pull_prices
from .portfolio import load_config, portfolio_returns, resolve_weights
from .risk import risk_table

log = logging.getLogger("market_risk_lab")


def _mc_vs_parametric(tbl) -> list[dict]:
    """Max relative gap between Monte Carlo and parametric: should be simulation error only."""
    p = tbl.pivot_table(index=["date", "window_days", "alpha"], columns="method", values=["var", "es"])
    out = []
    for (W, a), g in p.groupby(level=["window_days", "alpha"]):
        out.append(dict(window_days=int(W), alpha=float(a),
                        max_rel_diff_var=float((g["var"]["mc_normal"] / g["var"]["parametric"] - 1).abs().max()),
                        max_rel_diff_es=float((g["es"]["mc_normal"] / g["es"]["parametric"] - 1).abs().max())))
    return out


def run(config_path="config/portfolios.yaml", out_dir="outputs", conn=None, benchmark: bool = True) -> dict:
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rk, dc = cfg["risk"], cfg["data"]
    own = conn is None
    conn = conn or store.connect()
    summary = dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   python=platform.python_version(), risk_config=rk, portfolios={})
    try:
        store.ensure_schema(conn)
        for name, pcfg in cfg["portfolios"].items():
            w = resolve_weights(pcfg)
            prices, cal = pull_prices(conn, w.index, dc["start_date"], dc.get("end_date"))
            rets, rep = build_returns(prices, w.index, cal, min_history_days=dc.get("min_history_days"),
                                      **cfg.get("validation", {}))
            rep.to_frame().to_csv(out / f"{name}_data_validation.csv", index=False)
            tbl = risk_table(rets, w, name, rk["windows"], rk["alphas"], zero_mean=rk["zero_mean"],
                             mc_draws=rk["mc_draws"], mc_seed=rk["mc_seed"])
            n_rows = store.write_results(conn, tbl)
            rolling = store.rolling_exceptions(conn, name)

            # the SQL window query and the numpy implementation must agree row for row
            tl = backtest.traffic_light_table(tbl)
            key = ["method", "window_days", "date"]
            a = rolling.sort_values(key).reset_index(drop=True)
            b = tl.sort_values(key).reset_index(drop=True)
            if len(a) != len(b) or not (a["exceptions_250d"].to_numpy() == b["exceptions_250d"].to_numpy()).all() \
                    or not (a["zone"].to_numpy() == b["zone"].to_numpy()).all():
                raise RuntimeError("SQL rolling exception count disagrees with the numpy implementation")

            bt = backtest.backtest_summary(tbl)
            tls = backtest.traffic_light_summary(tl)
            bt.to_csv(out / f"{name}_backtest_summary.csv", index=False)
            tls.to_csv(out / f"{name}_traffic_light_summary.csv", index=False)
            rolling.to_csv(out / f"{name}_rolling_exceptions.csv", index=False)
            port = portfolio_returns(rets, w)
            summary["portfolios"][name] = dict(
                description=pcfg.get("description"), weights=w.round(6).to_dict(),
                first_return_date=str(rets.index[0].date()), last_return_date=str(rets.index[-1].date()),
                n_returns=len(rets), mean_daily_return=float(port.mean()), daily_vol=float(port.std(ddof=1)),
                validation_counts=rep.counts(), results_rows_written=n_rows,
                backtest=json.loads(bt.to_json(orient="records")),
                traffic_light=json.loads(tls.to_json(orient="records")),
                mc_vs_parametric=_mc_vs_parametric(tbl),
                benchmark=run_benchmark(rets[w.index].to_numpy(), w.to_numpy(), rk["windows"], rk["alphas"],
                                        rk["zero_mean"]) if benchmark else None,
                charts=charts.make_all(tbl, rolling, out, name))
    finally:
        if own:
            conn.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    log.info("wrote %s", out / "summary.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/portfolios.yaml")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-benchmark", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(args.config, args.out, benchmark=not args.no_benchmark)


if __name__ == "__main__":
    main()
