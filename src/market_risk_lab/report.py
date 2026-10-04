"""Render the results section of README.md from outputs/.

    python -m market_risk_lab.report            # rewrite the block between the markers
    python -m market_risk_lab.report --check    # exit 1 if README and outputs/ disagree (CI)

Every number in that section comes from outputs/summary.json or
outputs/<portfolio>_data_validation.csv. Nothing between the markers is typed by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
LABEL = {"hs": "Historical simulation", "parametric": "Parametric normal", "mc_normal": "Monte Carlo normal"}
LOWER = {"hs": "historical simulation", "parametric": "parametric normal", "mc_normal": "Monte Carlo normal"}
ORDER = {m: i for i, m in enumerate(LABEL)}
CAPTION = {"returns_vs_var": "Daily return against 1-day VaR, exceptions marked",
           "rolling_exceptions": "Trailing 250-day exception count against the Basel zones",
           "var_method_comparison": "VaR by method"}


def _pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}%"


def _level(a: float) -> str:
    return f"{100 * a:g}%"


def _table(head: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _name(r: dict) -> str:
    return f"{LOWER[r['method']]} ({r['window_days']}-day)"


def _key(r: dict):
    return (-r.get("alpha", 0), r["window_days"], ORDER[r["method"]])


def _validation_line(counts: dict, csv_path: Path) -> str:
    dropped = counts.get("missing_price", 0) + counts.get("calendar_gap", 0)
    parts = [f"{dropped} returns dropped" if dropped else "no returns dropped"]
    n_out = counts.get("outlier", 0)
    if n_out and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            flags = Counter(row["detail"].rsplit("reconciliation_flag=", 1)[-1]
                            for row in csv.DictReader(fh) if row["check"] == "outlier")
        by_flag = ", ".join(f"{k} {v}" for k, v in sorted(flags.items()))
        parts.append(f"{n_out} returns beyond the outlier threshold flagged and kept "
                     f"(vendor reconciliation flag on those prices: {by_flag})")
    elif n_out:
        parts.append(f"{n_out} returns beyond the outlier threshold flagged and kept")
    other = {k: v for k, v in counts.items() if k not in ("missing_price", "calendar_gap", "outlier")}
    if other:
        parts.append("other warnings: " + ", ".join(f"{k} {v}" for k, v in sorted(other.items())))
    return "; ".join(parts)


def _portfolio(name: str, p: dict, cfg: dict, out: Path) -> str:
    var_a, es_a = max(cfg["alphas"]), min(cfg["alphas"])
    w = p["weights"]
    weights = (f"{', '.join(w)} ({_pct(next(iter(w.values())), 1)} each)" if len(set(w.values())) == 1
               else ", ".join(f"{k} {_pct(v, 1)}" for k, v in w.items()))
    bt = sorted(p["backtest"], key=_key)
    tl = sorted(p["traffic_light"], key=_key)

    lines = [f"### Portfolio `{name}`", "", p.get("description") or "", "",
             f"- Assets: {weights}",
             f"- Returns: {p['n_returns']:,} daily, {p['first_return_date']} to {p['last_return_date']}; "
             f"mean {_pct(p['mean_daily_return'], 3)}, volatility {_pct(p['daily_vol'])} per day",
             f"- Data checks: {_validation_line(p['validation_counts'], out / f'{name}_data_validation.csv')}",
             f"- Rows written to `risk.risk_results`: {p['results_rows_written']:,}", ""]

    lines += ["#### Backtest over the full sample", "", _table(
        ["Method", "Window", "Level", "Forecasts", "Exceptions", "Expected", "Rate", "Kupiec p", "Kupiec at 5%",
         "Realised loss / predicted ES"],
        [[LABEL[r["method"]], r["window_days"], _level(r["alpha"]), f"{r['n_obs']:,}", r["exceptions"],
          f"{r['expected_exceptions']:.1f}", _pct(r["exception_rate"]), f"{r['kupiec_p']:.4f}",
          "rejected" if r["reject_5pct"] else "not rejected",
          "n/a" if r["es_ratio"] is None else f"{r['es_ratio']:.2f}"] for r in bt]), "",
        "The last column is the mean realised loss on exception days divided by the mean ES predicted for those "
        "days at the same level. Above 1 means the tail was worse than the model said.", ""]

    lines += [f"#### Basel traffic light ({_level(var_a)} VaR, exceptions in the trailing 250 forecasts)", "", _table(
        ["Method", "Estimation window", "250-day windows", "Green", "Yellow", "Red", "Worst count", "Latest zone"],
        [[LABEL[r["method"]], r["window_days"], f"{r['n_windows']:,}", _pct(r["green_share"], 1),
          _pct(r["yellow_share"], 1), _pct(r["red_share"], 1), r["max_exceptions_250d"], r["latest_zone"]]
         for r in tl]), ""]

    head = [r for r in bt if r["alpha"] == var_a]
    rej, ok = [_name(r) for r in head if r["reject_5pct"]], [_name(r) for r in head if not r["reject_5pct"]]
    red, never = [r for r in tl if r["red_share"] > 0], [_name(r) for r in tl if r["red_share"] == 0]
    read = [f"- Kupiec at 5% on {_level(var_a)} VaR: "
            + "; ".join(x for x in (f"rejected for {_join(rej)}" if rej else "",
                                    f"not rejected for {_join(ok)}" if ok else "") if x) + "."]
    if red:
        read.append("- Red zone (10 or more exceptions in 250 days), share of days and worst count: "
                    + "; ".join(f"{_name(r)} {_pct(r['red_share'], 1)}, {r['max_exceptions_250d']}" for r in red)
                    + (f". Never reached by {_join(never)}." if never else "."))
    else:
        read.append("- No method reached the red zone (10 or more exceptions in 250 days).")
    es_rows = [r for r in bt if r["alpha"] == es_a and r["es_ratio"] is not None]
    by_method = []
    for m in LABEL:
        v = sorted(r["es_ratio"] for r in es_rows if r["method"] == m)
        if v:
            by_method.append(f"{LOWER[m]} {v[0]:.2f}" + (f" to {v[-1]:.2f}" if len(v) > 1 else ""))
    if by_method:
        read.append(f"- Realised loss on exception days relative to predicted ES at {_level(es_a)}: "
                    + "; ".join(by_method) + ".")
    mc = p.get("mc_vs_parametric") or []
    if mc:
        read.append(f"- Monte Carlo against parametric: largest relative gap {_pct(max(r['max_rel_diff_var'] for r in mc))} "
                    f"in VaR and {_pct(max(r['max_rel_diff_es'] for r in mc))} in ES across all dates, "
                    f"with {cfg['mc_draws']:,} draws.")
    lines += ["#### Reading the tables", "", *read, ""]

    bench = p.get("benchmark")
    if bench:
        sp = [b["speedup"] for b in bench]
        lines += ["#### Vectorised against one-date-at-a-time", "", _table(
            ["Method", "Window", "Forecasts", "Vectorised (ms)", "Loop (ms)", "Speedup", "Largest difference"],
            [[LABEL[b["method"]], b["window_days"], f"{b['n_forecasts']:,}", f"{1000 * b['vectorised_s']:.1f}",
              f"{1000 * b['naive_s']:.1f}", f"{b['speedup']:.0f}x", f"{b['max_abs_diff']:.1e}"]
             for b in sorted(bench, key=_key)]), "",
            f"{min(sp):.0f}x to {max(sp):.0f}x on the machine that produced this run. Timings are machine-dependent; "
            "the equality of the two outputs is not.", ""]

    for c in p.get("charts", []):
        stem = c.rsplit(".", 1)[0].removeprefix(f"{name}_")
        lines += [f"![{CAPTION.get(stem, stem)}]({out.as_posix()}/{c})", ""]
    return "\n".join(lines)


def render(out_dir="outputs") -> str:
    out = Path(out_dir)
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    cfg = s["risk_config"]
    head = [f"_Rendered by `python -m market_risk_lab.report` from `{out.as_posix()}/summary.json`: run of "
            f"{s['generated_at']} on Python {s['python']}. Do not edit by hand._", "",
            f"Settings: estimation windows {_join([str(w) for w in cfg['windows']])} days; levels "
            f"{_join([_level(a) for a in sorted(cfg['alphas'], reverse=True)])}; "
            f"{'zero-mean' if cfg['zero_mean'] else 'sample-mean'} normal methods; "
            f"{cfg['mc_draws']:,} Monte Carlo draws, seed {cfg['mc_seed']}.", ""]
    body = [_portfolio(n, p, cfg, out) for n, p in s["portfolios"].items()]
    return "\n".join(head + body).rstrip() + "\n"


def _split(text: str) -> tuple[str, str, str]:
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise SystemExit(f"README needs exactly one {START} ... {END} pair")
    a, rest = text.split(START)
    b, c = rest.split(END)
    return a, b, c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    path = Path(args.readme)
    a, current, c = _split(path.read_text(encoding="utf-8"))
    block = "\n" + render(args.out)
    if args.check:
        if current != block:
            sys.exit(f"{path} results section is out of date with {args.out}/: run python -m market_risk_lab.report")
        print(f"{path} matches {args.out}/")
        return
    path.write_text(a + START + block + END + c, encoding="utf-8")
    print(f"updated {path} from {args.out}/")


if __name__ == "__main__":
    main()
