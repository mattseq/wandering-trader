import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(pd.__version__)

ticker = "SPY"

data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
print(data.head())

# data = data[['Close']]

close = data['Close'].squeeze()

print(type(data))
print(type(data['Close']))

# features
data['ret_1'] = close.pct_change(1)
data['ret_5'] = close.pct_change(5)
data['ret_10'] = close.pct_change(10)
data['ret_20'] = close.pct_change(20)
data['ma_10'] = close.rolling(10).mean()
data['ma_20'] = close.rolling(20).mean()
data['ma_diff_10'] = close - data['ma_10']
data['ma_diff_20'] = close - data['ma_20']
data['vol_10'] = data['ret_5'].rolling(10).std()
data['vol_20'] = data['ret_10'].rolling(20).std()
data['weekly_return'] = close.pct_change(5).shift(-5)
data['daily_return'] = close.pct_change(1).shift(-1)

# drop rows with NaN values, like the first few rows where rolling calculations arent possible
data.dropna(inplace=True)

# target var is whether the return the next day is positive (1) or negative (0)
data['target'] = (data['weekly_return'] > 0).astype(int)
data.dropna(inplace=True)

print(data.head())

# inputs and labels
features = ['ret_1', 'ret_5', 'ret_10', 'ret_20', 'ma_10', 'ma_20', 'ma_diff_10', 'ma_diff_20', 'vol_10', 'vol_20']
X = data[features].values
y = data['target'].values.reshape(-1, 1)

# standardize features
scalar = StandardScaler()
X = scalar.fit_transform(X)

# train window, model always trains on the last year of data and tests/evals on the next week of data, then rolls forward by a week and repeats the process
train_window = 252
# test window is 5 days (1 week) bc using weekly returns as target
test_window = 5

predictions = []
actuals = []
model_returns = []
test_indices = []

for start in range(0, len(X) - train_window - test_window, test_window):
    end = start + train_window
    X_train, y_train = X[start:end], y[start:end]
    X_test, y_test = X[end:end + test_window], y[end:end + test_window]
    
    # retrain the model on the new training data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.ravel())

    pred = model.predict(X_test)
    predictions.append(pred[0])
    actuals.append(y_test[0][0])
    model_returns.append(data['weekly_return'].values[end] if pred[0] == 1 else 0)
    test_indices.append(end)

predictions = np.array(predictions)
actuals = np.array(actuals)
model_returns = np.array(model_returns)

# literally just 5, 10, 15, 20, etc because rolling forward by a week (5 days) each time
test_indices = np.array(test_indices)

print("Accuracy:", accuracy_score(actuals, predictions))
print(classification_report(actuals, predictions))

cumulative_model_returns = np.cumprod(1 + model_returns)
returns = data['weekly_return'].values[test_indices]
buy_and_hold_returns = np.cumprod(1 + returns) - 1

print("Cumulative Model Returns:", cumulative_model_returns[-1])
print("Cumulative Buy and Hold Returns:", buy_and_hold_returns[-1])

plt.figure(figsize=(12, 6))
plt.plot(test_indices, cumulative_model_returns, label='Model Returns')
plt.plot(test_indices, buy_and_hold_returns, label='Buy and Hold Returns')
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