"""market-risk-lab: rolling VaR/ES and backtesting on the marketdata-lakehouse gold layer.

Conventions used everywhere: simple returns; loss = -return; VaR and ES are
positive numbers in return units (fraction of portfolio value); the forecast
for day t uses returns in [t-W, t-1] only.
"""
__version__ = "0.1.0"
