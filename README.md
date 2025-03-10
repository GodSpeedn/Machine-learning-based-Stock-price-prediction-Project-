# Machine-learning-based-Stock-price-prediction-Project-
# Stock Price Trend Prediction using Machine Learning

## Overview
This project uses machine learning to predict the 5-day stock price trend for Apple Inc. (AAPL) based on historical data. It utilizes features such as moving averages, RSI, Bollinger Bands, and MACD, and employs a Random Forest classifier to forecast whether the stock price will go up or down in the next five trading days.

## Features Used
- **Simple Moving Averages (SMA)**: 5-day and 20-day moving averages.
- **Relative Strength Index (RSI)**: Measures momentum and overbought/oversold conditions.
- **Bollinger Bands**: Upper and lower bands to indicate volatility.
- **MACD (Moving Average Convergence Divergence)**: Captures trend strength.
- **Volatility & Momentum**: Helps in identifying stock movement strength.
- **Lagged Returns**: Previous day's percentage change in price.

## Requirements
Before running the script, ensure you have Python installed along with the following dependencies:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib
```

## How to Run
1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/stock-prediction.git
   cd stock-prediction
   ```

2. Run the script:

   ```bash
   python stock_prediction.py
   ```

## How It Works
1. **Fetches historical stock data** for AAPL from Yahoo Finance.
2. **Calculates technical indicators** (RSI, Bollinger Bands, MACD, etc.).
3. **Prepares training and testing datasets** with an 80-20 split.
4. **Tunes a Random Forest classifier** using TimeSeriesSplit and GridSearchCV.
5. **Trains the best model** and makes predictions on test data.
6. **Evaluates model performance** using accuracy, feature importance, and a confusion matrix.
7. **Displays graphs** for actual vs. predicted trends and feature importance.
8. **Calculates Value at Risk (VaR)** to estimate potential losses.

## Output
- Model accuracy on test data.
- Feature importance rankings.
- Confusion matrix visualization.
- Actual vs. predicted trend graph.
- Value at Risk (VaR) estimation.



