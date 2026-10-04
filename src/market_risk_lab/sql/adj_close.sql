-- Trusted adjusted close per (ticker, date) from the lakehouse gold layer.
-- reconciliation_flag travels with every price so the validation layer can
-- tell a two-vendor-confirmed move from a single-source one.
SELECT d.date,
       s.ticker,
       c.adj_close::float8 AS adj_close,
       c.primary_source,
       c.reconciliation_flag
FROM fact_price_daily_consensus c
JOIN dim_security s USING (security_key)
JOIN dim_date d USING (date_key)
WHERE s.ticker = ANY(%(tickers)s)
  -- dim_security is SCD-2: a ticker gets a new security_key whenever a tracked
  -- attribute changes, and the lakehouse loads facts under whichever key is
  -- current at load time. Rows left under a superseded key are an older load
  -- of the same dates, so only the current key is read. A ticker with no
  -- current key (left the index) returns nothing and fails validation.
  AND s.is_current
  AND d.date >= %(start)s
  AND (%(end)s::date IS NULL OR d.date <= %(end)s::date)
ORDER BY d.date, s.ticker;
