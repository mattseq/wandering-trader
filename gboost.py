import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(pd.__version__)

ticker = "SPY"

data = yf.download(ticker, start="2015-01-01", end="2025-01-01")
print(data.head())

close = data['Close'].squeeze()

print(type(data))
print(type(data['Close']))

# features

# -------------------------
# 1. Returns / Momentum
# -------------------------
data['log_ret_1'] = np.log(close / close.shift(1))
data['momentum_5'] = close / close.shift(5) - 1
data['momentum_10'] = close / close.shift(10) - 1
data['momentum_20'] = close / close.shift(20) - 1
data['ret_1_sign'] = np.sign(data['log_ret_1'])

# -------------------------
# 2. Moving Averages / Trend
# -------------------------
data['ma_10'] = close.rolling(10).mean()
data['ma_20'] = close.rolling(20).mean()
data['ma_30'] = close.rolling(30).mean()
data['ma_ratio_10'] = close / data['ma_10'] - 1
data['ma_ratio_20'] = close / data['ma_20'] - 1
data['ma_ratio_30'] = close / data['ma_30'] - 1

data['ema_10'] = close.ewm(span=10, adjust=False).mean()
data['ema_20'] = close.ewm(span=20, adjust=False).mean()
data['ema_ratio_10'] = close / data['ema_10'] - 1
data['ema_ratio_20'] = close / data['ema_20'] - 1

data['ma_cross_10_20'] = (data['ma_10'] > data['ma_20']).astype(int)

# -------------------------
# 3. Volatility / Risk
# -------------------------
data['vol_10'] = data['log_ret_1'].rolling(10).std()
data['vol_20'] = data['log_ret_1'].rolling(20).std()
data['vol_30'] = data['log_ret_1'].rolling(30).std()

data['ret_skew_10'] = data['log_ret_1'].rolling(10).skew()
data['ret_skew_20'] = data['log_ret_1'].rolling(20).skew()
data['ret_kurt_10'] = data['log_ret_1'].rolling(10).kurt()
data['ret_kurt_20'] = data['log_ret_1'].rolling(20).kurt()

# Average True Range (ATR)
high_low = data['High'] - data['Low']
high_close = (data['High'] - data['Close'].shift()).abs()
low_close = (data['Low'] - data['Close'].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
data['ATR_14'] = tr.rolling(14).mean()

# -------------------------
# 4. Technical Indicators
# -------------------------
# RSI
delta = close.diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
roll_up = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
RS = roll_up / roll_down
data['RSI_14'] = 100 - (100 / (1 + RS))

# MACD
ema_12 = close.ewm(span=12, adjust=False).mean()
ema_26 = close.ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26
data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Bollinger Bands
ma_20 = close.rolling(20).mean()
std_20 = close.rolling(20).std()
data['BB_upper'] = ma_20 + 2 * std_20
data['BB_lower'] = ma_20 - 2 * std_20
data['BB_width'] = data['BB_upper'] - data['BB_lower']

data['weekly_return'] = close.pct_change(5).shift(-5)
data['daily_return'] = close.pct_change(1).shift(-1)

# target var is whether the return the next day is positive (1) or negative (0)
data['target'] = (data['weekly_return'] > 0).astype(int)

print(data.head())

# inputs and labels
features = [
    # -------------------------
    # 1. Returns / Momentum
    # -------------------------
    'log_ret_1',
    'momentum_5',
    'momentum_10',
    'momentum_20',
    'ret_1_sign',
    
    # -------------------------
    # 2. Moving Averages / Trend
    # -------------------------
    'ma_ratio_10',
    'ma_ratio_20',
    'ma_ratio_30',
    'ema_ratio_10',
    'ema_ratio_20',
    'ma_cross_10_20',
    
    # -------------------------
    # 3. Volatility / Risk
    # -------------------------
    'vol_10',
    'vol_20',
    'vol_30',
    'ret_skew_10',
    'ret_skew_20',
    'ret_kurt_10',
    'ret_kurt_20',
    'ATR_14',
    
    # -------------------------
    # 4. Technical Indicators
    # -------------------------
    'RSI_14',
    'MACD',
    'MACD_signal',
    'BB_upper',
    'BB_lower',
    'BB_width'
]

X = data[features].values
y = data['target'].values.reshape(-1, 1)

# train window, model always trains on the last year of data and tests/evals on the next week of data, then rolls forward by a week and repeats the process
train_window = 252
# test window is 5 days (1 week) bc using weekly returns as target
test_window = 5

predictions = []
actuals = []
model_returns = []
buy_and_hold_returns = []
test_indices = []

for end in range(train_window, len(X) - train_window - test_window, test_window):
    # end = start + train_window

    # create training windows and test windows
    X_train_window = X[0:end-test_window]  # drop the last 5 in the training set to avoid future leakage
    y_train_window = y[0:end-test_window]
    X_test_window = X[end:end+test_window]
    y_test_window = y[end:end+test_window]

    # drop NaNs inside the training window only
    mask_train = ~np.isnan(X_train_window).any(axis=1) & ~np.isnan(y_train_window).any(axis=1)
    X_train = X_train_window[mask_train]
    y_train = y_train_window[mask_train]

    mask_test = ~np.isnan(X_test_window).any(axis=1) & ~np.isnan(y_test_window).any(axis=1)
    X_test = X_test_window[mask_test]
    y_test = y_test_window[mask_test]

    if len(X_train) == 0 or len(X_test) == 0:
        print("No valid test data for window ending at index", end)
        continue
    
    # retrain the model on the new training data
    model = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )

    model.fit(X_train, y_train.ravel())

    pred = model.predict(X_test)
    predictions.append(pred[0])
    actuals.append(y_test[0][0])

    model_returns.append(data['weekly_return'].values[end] if pred[0] == 1 else 0)
    buy_and_hold_returns.append(data['weekly_return'].values[end])

    test_indices.append(end)

predictions = np.array(predictions)
actuals = np.array(actuals)

# literally just 5, 10, 15, 20, etc because rolling forward by a week (5 days) each time
model_returns = np.array(model_returns)
buy_and_hold_returns = np.array(buy_and_hold_returns)
test_indices = np.array(test_indices)

print(test_indices)

print("Accuracy:", accuracy_score(actuals, predictions))
print(classification_report(actuals, predictions))

cumulative_model_returns = np.cumprod(1 + model_returns) - 1
cumulative_buy_and_hold_returns = np.cumprod(1 + buy_and_hold_returns) - 1

periods = len(model_returns)
annualized_model_return = (1 + cumulative_model_returns[-1]) ** (52 / periods) - 1
annualized_buy_and_hold_return = (1 + cumulative_buy_and_hold_returns[-1]) ** (52 / periods) - 1

print("Cumulative Model Returns:", cumulative_model_returns[-1])
print("Cumulative Buy and Hold Returns:", cumulative_buy_and_hold_returns[-1])

print("Annualized Model Return:", annualized_model_return)
print("Annualized Buy and Hold Return:", annualized_buy_and_hold_return)

print("Pct weeks invested:", np.mean(predictions))
print("Weekly mean return (all):", np.mean(buy_and_hold_returns))
print("Weekly mean return (model):", np.mean(model_returns))
# print("Model Sharpe:", np.mean(model_returns) / np.std(model_returns))
# print("B&H Sharpe:", np.mean(buy_and_hold_returns) / np.std(buy_and_hold_returns))

model_sharpe = (np.mean(model_returns) / np.std(model_returns)) * np.sqrt(52)
print("Model Sharpe:", model_sharpe)
buy_and_hold_sharpe = (np.mean(buy_and_hold_returns) / np.std(buy_and_hold_returns)) * np.sqrt(52)
print("B&H Sharpe:", buy_and_hold_sharpe)


plt.figure(figsize=(12, 6))
plt.plot(test_indices, cumulative_model_returns, label='Model Returns')
plt.plot(test_indices, cumulative_buy_and_hold_returns, label='Buy and Hold Returns')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.title('Model vs Buy and Hold Returns on ' + ticker)
plt.legend()
plt.show()

### Look into random forest, gradient boosting, lstm, transformer models.
### more features like technical indicators, sentiment analysis, macroeconomic data, etc.

### Random Forest
# forest has subtrees which are decision trees (glorified piecewise functions)
# each tree is trained on a random subset of the data and features
# the final prediction is made by averaging the predictions of all the trees (for regression) or taking a majority vote (for classification)

### Improvements:
# 1. Simulate capital: right now, model returns are just compounded percentages, but simulating actual capital would be more realistic
#    start with $10k, invest all of it when the model predicts a positive return, and keep all of it when the model predicts a negative return
# 2. Add transaction costs: frequent trading would incur transaction costs, so add a realistic cost per trade to see how it affects performance
#    this also means its a good idea to trade as infrequently as possible, so maybe try a longer test window
# 3. Hyperparameter tuning: experiment with different numbers of trees, max depth, etc using grid search
# 4. Switch to a regression random forest that predicts either probability of a positive return or the expected return itself and allocate capital based off of that
# 5. Include cross-asset signals like VIX for volatility, etc.


### THE PROBLEM:
# the first 30 wont be trained on bc features make them nan
# however, the last 5 in the training set point to the future bc they're weekly return correlates to the next 5 days returns
# solution: drop the last 5 in the training set