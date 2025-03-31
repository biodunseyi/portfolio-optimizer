import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_loader import fetch_data
from optimizer import (
    calculate_daily_returns,
    calculate_expected_returns,
    calculate_covariance_matrix,
    simulate_random_portfolios
)

# App config
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("📊 Portfolio Optimizer using Modern Portfolio Theory")
st.markdown("Enter your stock preferences and let the model do the rest.")

# Sidebar inputs
st.sidebar.header("Portfolio Settings")

tickers_input = st.sidebar.text_input("Enter stock tickers (comma separated)", "AAPL,MSFT,GOOGL")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

num_portfolios = st.sidebar.slider("Number of Portfolios to Simulate", 1000, 10000, 5000, step=500)
budget = st.sidebar.number_input("Enter total investment budget ($)", min_value=100.0, value=5000.0, step=100.0)

# Fetch data and run optimization
try:
    price_data = fetch_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    daily_returns = calculate_daily_returns(price_data)
    mean_returns = calculate_expected_returns(daily_returns)
    cov_matrix = calculate_covariance_matrix(daily_returns)

    portfolio_df = simulate_random_portfolios(num_portfolios, mean_returns, cov_matrix)
    best_port = portfolio_df.sort_values(by='Sharpe', ascending=False).iloc[0]

    st.success("Optimization Complete!")
    st.subheader("💼 Optimal Portfolio Allocation")
    weights_df = pd.DataFrame(best_port['Weights'], index=['Weight']).T
    st.dataframe(weights_df.style.format({'Weight': '{:.2%}'}))

    st.metric("📈 Expected Annual Return", f"{best_port['Return']:.2%}")
    st.metric("⚖️ Risk (Volatility)", f"{best_port['Risk']:.2%}")
    st.metric("🔍 Sharpe Ratio", f"{best_port['Sharpe']:.2f}")

    # Budget-based allocation
    weights_df['Dollar Amount ($)'] = weights_df['Weight'] * budget
    st.subheader("💰 Investment Allocation")
    st.dataframe(weights_df.style.format({
        'Weight': '{:.2%}',
        'Dollar Amount ($)': '${:,.2f}'
    }))

    # Pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(best_port['Weights'].values(), labels=best_port['Weights'].keys(), autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Efficient Frontier
    st.subheader("📉 Efficient Frontier")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    scatter = ax2.scatter(
        portfolio_df['Risk'],
        portfolio_df['Return'],
        c=portfolio_df['Sharpe'],
        cmap='viridis',
        alpha=0.7
    )
    ax2.set_xlabel('Volatility (Risk)')
    ax2.set_ylabel('Expected Return')
    ax2.set_title('Efficient Frontier')
    ax2.scatter(best_port['Risk'], best_port['Return'], color='red', marker='*', s=200, label='Optimal Portfolio')
    ax2.legend()
    fig2.colorbar(scatter, label='Sharpe Ratio')
    st.pyplot(fig2)

except Exception as e:
    st.error(f"⚠️ Error: {e}")
