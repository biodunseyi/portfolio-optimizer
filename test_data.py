from data_loader import fetch_data

tickers = ['AAPL', 'MSFT', 'GOOGL']
data = fetch_data(tickers)

print(data.tail())
