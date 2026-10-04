SELECT date
FROM dim_date
WHERE is_trading_day
  AND date >= %(start)s
  AND (%(end)s::date IS NULL OR date <= %(end)s::date)
ORDER BY date;
