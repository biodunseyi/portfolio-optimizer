import numpy as np
import pandas as pd

def calculate_daily_returns(price_data):
    """
    Calculates daily returns from price data.
    """
    return price_data.pct_change().dropna()

def calculate_expected_returns(daily_returns):
    """
    Calculates mean daily returns and annualizes them.
    """
    return daily_returns.mean() * 252  # 252 trading days in a year

def calculate_covariance_matrix(daily_returns):
    """
    Calculates the annualized covariance matrix of daily returns.
    """
    return daily_returns.cov() * 252

def simulate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate=0.01):
    """
    Generates random portfolios and calculates their performance metrics.
    Returns DataFrame with weights, returns, risk, and Sharpe Ratio.
    """
    results = []
    num_assets = len(mean_returns)
    tickers = mean_returns.index.tolist()

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results.append({
            'Return': portfolio_return,
            'Risk': portfolio_volatility,
            'Sharpe': sharpe_ratio,
            'Weights': dict(zip(tickers, weights))
        })

    return pd.DataFrame(results)
