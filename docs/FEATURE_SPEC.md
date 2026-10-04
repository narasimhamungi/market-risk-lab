# Feature spec: rolling VaR/ES with automated backtesting and traffic-light reporting

**Status: self-initiated exercise.** Written as a product requirement for a hypothetical client feature. No client asked for it, and it describes no real vendor's product or internals. The prototype in this repository implements the scope below.

## Problem and target user

A market risk analyst at a small buy-side firm reports daily VaR and ES on a few model portfolios and has to show the numbers can be trusted. Today that is one spreadsheet for risk, another for exception counting, and no record of which data was used. The user needs one run that produces the measures, tests them against realised P&L and says plainly when a model is failing.

## Functional requirements

1. Compute 1-day VaR and ES per portfolio on rolling windows by historical simulation, parametric normal and Monte Carlo normal. Windows and levels are configuration.
2. Define portfolios as fixed weights in a config file; reject negative weights and weights that do not sum to one.
3. Backtest every method, window and level: exception count and rate, Kupiec test with p-value.
4. Report the Basel traffic-light zone from the trailing 250-forecast exception count.
5. Compare realised losses on exception days with predicted ES.
6. Validate prices first; stop on structural faults; record every dropped or flagged observation with its reason.
7. Persist results to a SQL table and regenerate all tables and charts with one command.

## Non-functional requirements

- No forecast may use data from its own day or later.
- Same data and config give identical results; simulation is seeded.
- Every reported number traces to a generated file.
- A run over 8 assets and about 2,000 days finishes in under one minute on a laptop.
- A re-run replaces results and never appends.

## Inputs and outputs

- **In:** daily adjusted closes and a trading calendar from a SQL price store; a YAML file of portfolios, windows, levels and validation thresholds.
- **Out:** `risk_results(date, portfolio, method, window_days, alpha, var, es, realised_pnl, exception)`; backtest and traffic-light summaries; a validation log; three charts; a machine-readable run summary.

## Methodology

Simple returns; loss is minus return; VaR and ES are positive. Historical simulation uses the empirical quantile and exact tail average of the window's losses; parametric uses the rolling sample covariance and normal quantiles; Monte Carlo draws from the same covariance. The forecast for day t uses the window ending on day t − 1. Details are in the README.

## Acceptance criteria

Each is a test in `tests/`.

1. Perturbing the return on day t leaves the day-t forecast unchanged and changes the next day's.
2. ES is at least VaR at the same level in every row, also enforced as a database constraint.
3. Vectorised output equals a one-date-at-a-time reference to a relative tolerance of 1e-9.
4. On a large independent normal sample every method recovers the analytical VaR and ES within 2%.
5. The Kupiec statistic matches an independently computed binomial likelihood ratio and is finite at zero exceptions.
6. Zone boundaries fall between 4 and 5 and between 9 and 10 exceptions.
7. The SQL rolling exception count equals the in-memory count row for row.
8. Duplicate keys, non-positive prices, stale prices and excess missing data abort the run with the check named.
9. A second run leaves the same number of rows in the results table.

## Out of scope

Horizons beyond one day, non-linear instruments, volatility-weighted or filtered historical simulation, stressed VaR, formal ES backtests, exception-independence tests, a user interface, scheduling and alerting.

## Open questions and risks

- Which P&L does the client backtest against? The prototype uses hypothetical fixed-weight P&L.
- A frequency test can pass a model whose exceptions cluster; an independence test is the first candidate for a second release.
- Vendors restate adjusted prices, so results are tied to a data snapshot. Snapshot retention is undecided.
- Small exception counts make every test low-powered; a pass must not be reported as proof.

## Rollout

1. Prototype on one equity portfolio with committed outputs (this repository).
2. Add an independence test and one volatility-weighted method; re-run the same backtests.
3. Read-only pilot on a client's portfolios and price source.
4. Daily schedule, with the traffic-light zone as the alert condition.
