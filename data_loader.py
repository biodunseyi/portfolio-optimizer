import yfinance as yf
import pandas as pd

def fetch_data(tickers, start_date='2023-01-01', end_date='2024-01-01'):
    """
    Fetch closing prices for a list of tickers.
    If 'Adj Close' is missing, it uses 'Close' instead.
    """
    all_data = pd.DataFrame()

    for ticker in tickers:
        try:
            print(f"Fetching {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if 'Adj Close' in data.columns:
                all_data[ticker] = data['Adj Close']
            elif 'Close' in data.columns:
                all_data[ticker] = data['Close']
            else:
                print(f"{ticker} returned no 'Adj Close' or 'Close'")
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")

    return all_data.dropna()
