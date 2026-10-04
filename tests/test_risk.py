import numpy as np
import pytest
from scipy.stats import norm

from market_risk_lab import naive, risk

from .conftest import correlated_returns

ALPHAS = (0.99, 0.975)


def test_converges_to_analytical_normal_value():
    """Large iid normal sample: every method must recover z*sigma and sigma*pdf(z)/(1-alpha)."""
    W = 100_000
    R, cov = correlated_returns(W + 3, 3, seed=11)
    w = np.array([0.5, 0.3, 0.2])
    sigma = np.sqrt(w @ cov @ w)
    res = risk.compute_risk(R, w, [W], ALPHAS, mc_draws=100_000, mc_seed=5)
    for a in ALPHAS:
        var_true, es_true = sigma * norm.ppf(a), sigma * norm.pdf(norm.ppf(a)) / (1 - a)
        for method, tol in (("hs", 0.02), ("parametric", 0.01), ("mc_normal", 0.02)):
            var, es = res[(method, W)][a]
            assert np.allclose(var, var_true, rtol=tol), (method, a)
            assert np.allclose(es, es_true, rtol=tol), (method, a)


@pytest.mark.parametrize("zero_mean", [True, False])
def test_es_never_below_var(zero_mean):
    R, _ = correlated_returns(900, 6, seed=3, df=4)  # fat tails
    res = risk.compute_risk(R, np.full(6, 1 / 6), [250, 500], ALPHAS, zero_mean=zero_mean, mc_draws=20_000)
    for by_alpha in res.values():
        for var, es in by_alpha.values():
            assert (es >= var).all() and (var > 0).all()


def test_vectorised_equals_naive_loop():
    R, _ = correlated_returns(330, 5, seed=2, df=5)
    w = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
    W, loss = 250, -(R @ w)
    for zero_mean in (True, False):
        fast = risk.compute_risk(R, w, [W], ALPHAS, zero_mean=zero_mean, mc_draws=20_000, mc_seed=9)
        slow = {"hs": naive.naive_hs(loss, W, ALPHAS),
                "parametric": naive.naive_parametric(R, w, W, ALPHAS, zero_mean),
                "mc_normal": naive.naive_mc_normal(R, w, W, ALPHAS, zero_mean, 20_000, 9)}
        for method, ref in slow.items():
            for a in ALPHAS:
                for i in (0, 1):
                    np.testing.assert_allclose(fast[(method, W)][a][i], ref[a][i], rtol=1e-9, atol=0)


def test_no_look_ahead():
    """A shock on day t must leave the forecast for day t untouched and move the one for t+1."""
    R, _ = correlated_returns(400, 4, seed=4)
    w, W, t = np.full(4, 0.25), 250, 320
    shocked = R.copy()
    shocked[t] -= 0.30
    base = risk.compute_risk(R, w, [W], ALPHAS, mc_draws=5_000)
    pert = risk.compute_risk(shocked, w, [W], ALPHAS, mc_draws=5_000)
    for key in base:
        for a in ALPHAS:
            for i in (0, 1):
                b, p = base[key][a][i], pert[key][a][i]
                assert np.array_equal(b[: t - W + 1], p[: t - W + 1]), key   # forecasts up to and including day t
                assert p[t - W + 1] > b[t - W + 1], key                      # day t+1 sees the shock


def test_alignment_on_a_known_sequence():
    """Losses 0,1,2,...: window i is {i..i+W-1}, so VaR_i = i + ceil(W*alpha) - 1 exactly."""
    T, W, a = 300, 250, 0.99
    loss = np.arange(T, dtype=float)
    var, es = risk.rolling_hs(loss, W, [a])[a]
    assert len(var) == T - W
    assert np.array_equal(var, np.arange(T - W) + 248 - 1)
    assert np.allclose(es, np.arange(T - W) + (248 + 249 + 0.5 * 247) / 2.5)


def test_tail_index_and_fractional_es_weights():
    assert risk.tail_index(250, 0.99) == 248      # 3rd-worst of 250
    assert risk.tail_index(250, 0.975) == 244
    assert risk.tail_index(1000, 0.99) == 990     # float product is 990.0000000000001
    var, es = risk.empirical_var_es(np.arange(1.0, 251.0), 0.975)
    assert var == 244 and np.isclose(es, (245 + 246 + 247 + 248 + 249 + 250 + 0.25 * 244) / 6.25)


def test_parametric_sigma_is_the_portfolio_series_volatility():
    """w' S w with the sample covariance equals the sample variance of the weighted return series."""
    R, _ = correlated_returns(300, 4, seed=6)
    w, W = np.array([0.1, 0.2, 0.3, 0.4]), 250
    mu, cov = risk.rolling_moments(R, W)
    var = risk.rolling_parametric(mu, cov, w, [0.99])[0.99][0]
    port = R @ w
    expected = np.array([port[i:i + W].std(ddof=1) for i in range(300 - W)]) * norm.ppf(0.99)
    np.testing.assert_allclose(var, expected, rtol=1e-10)


def test_monte_carlo_is_parametric_plus_simulation_error():
    R, _ = correlated_returns(300, 8, seed=8, df=5)
    res = risk.compute_risk(R, np.full(8, 0.125), [250], ALPHAS, mc_draws=100_000, mc_seed=1)
    for a in ALPHAS:
        for i in (0, 1):
            assert np.allclose(res[("mc_normal", 250)][a][i], res[("parametric", 250)][a][i], rtol=0.02)


def test_window_longer_than_history_raises():
    with pytest.raises(ValueError):
        risk.compute_risk(np.zeros((100, 2)), np.array([0.5, 0.5]), [250], ALPHAS)
