-- The three gold tables this project reads, copied from
-- marketdata-lakehouse src/model/schema.sql at commit 7a2394c.
-- If the lakehouse schema changes, this fixture and sql/adj_close.sql change with it.
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INTEGER PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    year SMALLINT NOT NULL,
    quarter SMALLINT NOT NULL,
    month SMALLINT NOT NULL,
    month_name TEXT NOT NULL,
    day SMALLINT NOT NULL,
    day_of_week SMALLINT NOT NULL,
    day_name TEXT NOT NULL,
    is_weekday BOOLEAN NOT NULL,
    is_trading_day BOOLEAN NOT NULL
);
CREATE TABLE IF NOT EXISTS dim_security (
    security_key SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    company_name TEXT,
    gics_sector TEXT,
    gics_sub_industry TEXT,
    figi TEXT,
    effective_from DATE NOT NULL,
    effective_to DATE,
    is_current BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE TABLE IF NOT EXISTS fact_price_daily_consensus (
    security_key INTEGER NOT NULL REFERENCES dim_security(security_key),
    date_key INTEGER NOT NULL REFERENCES dim_date(date_key),
    adj_close NUMERIC(18, 6) NOT NULL,
    primary_source TEXT NOT NULL,
    sources_available TEXT[] NOT NULL,
    pct_diff NUMERIC(9, 6),
    reconciliation_flag TEXT NOT NULL,
    PRIMARY KEY (security_key, date_key)
);
