from data_loader import fetch_data
from optimizer import calculate_daily_returns, calculate_expected_returns, calculate_covariance_matrix, simulate_random_portfolios

tickers = ['AAPL', 'MSFT', 'GOOGL']
data = fetch_data(tickers)

daily_returns = calculate_daily_returns(data)
mean_returns = calculate_expected_returns(daily_returns)
cov_matrix = calculate_covariance_matrix(daily_returns)

portfolio_df = simulate_random_portfolios(5000, mean_returns, cov_matrix)

# Show top 5 portfolios by Sharpe ratio
print(portfolio_df.sort_values(by='Sharpe', ascending=False).head())
