-- Rolling 250-observation exception count per method at the 0.99 level, with the Basel
-- traffic-light zone. ROWS (not RANGE): the Basel count is over trading
-- observations, not calendar days. Incomplete leading windows are excluded
-- rather than reported as a misleadingly low count.
WITH counted AS (
    SELECT date, portfolio, method, window_days,
           SUM(exception::int) OVER w AS exceptions_250d,
           COUNT(*)            OVER w AS obs_in_window
    FROM risk.risk_results
    WHERE portfolio = %(portfolio)s
      AND alpha = 0.99
    WINDOW w AS (
        PARTITION BY portfolio, method, window_days
        ORDER BY date
        ROWS BETWEEN 249 PRECEDING AND CURRENT ROW
    )
)
SELECT date, portfolio, method, window_days, exceptions_250d,
       CASE WHEN exceptions_250d <= 4 THEN 'green'
            WHEN exceptions_250d <= 9 THEN 'yellow'
            ELSE 'red'
       END AS zone
FROM counted
WHERE obs_in_window = 250
ORDER BY method, window_days, date;
