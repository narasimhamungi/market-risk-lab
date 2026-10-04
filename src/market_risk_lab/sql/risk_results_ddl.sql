-- Results live in their own schema: the lakehouse gold tables are read-only here.
-- "window" is a reserved word in Postgres, hence window_days.
CREATE SCHEMA IF NOT EXISTS risk;

CREATE TABLE IF NOT EXISTS risk.risk_results (
    date         DATE             NOT NULL,   -- day the forecast applies to
    portfolio    TEXT             NOT NULL,
    method       TEXT             NOT NULL,   -- hs | parametric | mc_normal
    window_days  SMALLINT         NOT NULL,
    alpha        NUMERIC(5, 4)    NOT NULL,
    var          DOUBLE PRECISION NOT NULL,   -- positive = loss
    es           DOUBLE PRECISION NOT NULL,
    realised_pnl DOUBLE PRECISION NOT NULL,   -- portfolio simple return on date
    exception    BOOLEAN          NOT NULL,
    PRIMARY KEY (date, portfolio, method, window_days, alpha),
    CHECK (es >= var),
    CHECK (exception = (-realised_pnl > var))  -- sign convention enforced in the store
);
