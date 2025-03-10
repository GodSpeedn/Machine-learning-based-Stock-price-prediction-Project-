import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt

# RSI calculation
def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Bollinger Bands calculation
def calculate_bollinger_bands(data, window=20):
    """Calculate Bollinger Bands."""
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    return sma + 2 * std, sma - 2 * std

# MACD calculation
def calculate_macd(data, short_window=12, long_window=26):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    ema_short = data['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = data['Close'].ewm(span=long_window, adjust=False).mean()
    return ema_short - ema_long

# Fetch 10 years of data
print("Fetching 10 years of historical data for AAPL...")
try:
    stock = yf.download('AAPL', start='2015-01-01', end='2025-01-01', auto_adjust=False)
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()
data = stock[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Define target: 5-day trend
data['Target'] = (data['Close'].shift(-5) > data['Close']).astype(int)

# Feature engineering
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['RSI'] = calculate_rsi(data)
data['Lag_Return'] = data['Close'].pct_change().shift(1)
data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
data['Momentum'] = data['Close'].diff(5)
data['Upper_BB'], data['Lower_BB'] = calculate_bollinger_bands(data)
data['MACD'] = calculate_macd(data)

# Drop NaN
data = data.dropna()

# Features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI',
            'Lag_Return', 'Volatility', 'Momentum', 'Upper_BB', 'Lower_BB', 'MACD']
X = data[features]
y = data['Target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Chronological split (80% train, 20% test)
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
test_dates = data.index[train_size:]

# Hyperparameter tuning with Random Forest
print("Tuning the Random Forest model...")
model = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [5, 10, 15]
}
tscv = TimeSeriesSplit(n_splits=3)
grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Train and predict with best model
print("Training the optimized Random Forest model...")
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Prediction Accuracy (5-day trend): {accuracy:.2%}")

# Feature importance
importances = best_model.feature_importances_
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Calculate VaR
portfolio_value = 100 * data['Close']
portfolio_returns = portfolio_value.pct_change().dropna()
VaR_95 = np.percentile(portfolio_returns, 5) * portfolio_value.iloc[-1].item()
print(f"\n95% Daily VaR: ${-VaR_95:.2f} (Loss at 5% probability)")

# Visualize trend
plt.figure(figsize=(10, 6))
plt.plot(test_dates, y_test, label='Actual 5-day Trend', alpha=0.7)
plt.plot(test_dates, y_pred, label='Predicted 5-day Trend', alpha=0.7)
plt.title('Actual vs Predicted 5-day Price Trends (AAPL)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Trend (1 = Up, 0 = Down)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Visualize feature importance
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance (Random Forest)', fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()