import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(pd.__version__)

ticker = input("Enter ticker symbol (data must be available for 2015-2025): ")

START = "2015-01-01"
END = "2025-01-01"

print(f"Fetching data for {ticker} from {START} to {END}...")

data = yf.download(ticker, start=START, end=END)
vix = yf.download("^VIX", start=START, end=END)
vix = vix.reindex(data.index).fillna(method='ffill')

print(data.head())

close = data['Close'].squeeze()

print(type(data))
print(type(data['Close']))

# features
data['ret_1'] = close.pct_change(1)
data['ret_5'] = close.pct_change(5)
data['ret_10'] = close.pct_change(10)
data['ret_20'] = close.pct_change(20)
data['ret_30'] = close.pct_change(30)

data['ma_10'] = close.rolling(10).mean()
data['ma_20'] = close.rolling(20).mean()
data['ma_30'] = close.rolling(30).mean()
data['ma_diff_10'] = close - data['ma_10']
data['ma_diff_20'] = close - data['ma_20']
data['ma_diff_30'] = close - data['ma_30']

data['vol_10'] = data['ret_5'].rolling(10).std()
data['vol_20'] = data['ret_10'].rolling(20).std()
data['vol_30'] = data['ret_20'].rolling(30).std()

data['volume_10'] = data['Volume'].rolling(10).mean()
data['volume_20'] = data['Volume'].rolling(20).mean()
data['volume_30'] = data['Volume'].rolling(30).mean()

data['vix_close'] = vix['Close'].values
data['vix_ret_1'] = data['vix_close'].pct_change(1)
data['vix_ret_5'] = data['vix_close'].pct_change(5)
data['vix_ret_10'] = data['vix_close'].pct_change(10)

data['weekly_return'] = close.pct_change(5).shift(-5)
data['daily_return'] = close.pct_change(1).shift(-1)

# target var is whether the return the next day is positive (1) or negative (0)
data['target'] = (data['weekly_return'] > 0).astype(int)

print(data.head())

# inputs and labels
features = ['ret_1', 'ret_5', 'ret_10', 'ret_20', 'ret_30', 'ma_10', 'ma_20', 'ma_30', 'ma_diff_10', 'ma_diff_20', 'ma_diff_30', 'vol_10', 'vol_20', 'vol_30', 'volume_10', 'volume_20', 'volume_30', 'vix_close', 'vix_ret_1', 'vix_ret_5', 'vix_ret_10']
X = data[features].values
y = data['target'].values.reshape(-1, 1)

# train window, model always trains on the last year of data and tests/evals on the next week of data, then rolls forward by a week and repeats the process
train_window = 500
# test window is 5 days (1 week) bc using weekly returns as target
test_window = 5

predictions = []
actuals = []
model_returns = []
buy_and_hold_returns = []
test_indices = []

for end in range(train_window, len(X) - test_window, test_window):
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
    if (~mask_train).any():
        dropped_indices = np.where(~mask_train)[0]
        # expect first 49 rows to be dropped due to 30-day rolling features, but if there are any NaNs after that, print a warning
        if np.any(dropped_indices > 49):
            print(f"NaN in training set after index 49 for window ending at {end}: {dropped_indices[dropped_indices > 49]}")


    mask_test = ~np.isnan(X_test_window).any(axis=1) & ~np.isnan(y_test_window).any(axis=1)
    X_test = X_test_window[mask_test]
    y_test = y_test_window[mask_test]
    if (~mask_test).any():
        print(f"Dropping {len(X_test_window) - np.sum(mask_test)} rows with NaNs from test data for window ending at index {end}")

    if len(X_train) == 0 or len(X_test) == 0:
        print("No valid test data for window ending at index", end)
        continue

    n_train = len(X_train)

    decay = 0.05
    weights = np.exp(np.linspace(-decay * n_train, 0, n_train))

    weights = weights / np.sum(weights)
    
    # retrain the model on the new training data
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42
    )

    model.fit(X_train, y_train.ravel(), sample_weight=weights)

    pred = model.predict(X_test)
    predictions.append(pred[0])
    actuals.append(y_test[0][0])

    # print(f"Difference: {pred[0] - y_test[0][0]}, Confidence: {model.predict_proba(X_test)[0][1]:.4f}, Predicted: {pred[0]}, Actual: {y_test[0][0]}")

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

accuracies = predictions == actuals
correct_model_returns = model_returns[accuracies]
correct_buy_and_hold_returns = buy_and_hold_returns[accuracies]
incorrect_model_returns = model_returns[~accuracies]
incorrect_buy_and_hold_returns = buy_and_hold_returns[~accuracies]

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(test_indices, cumulative_model_returns, label='Model Returns')
plt.plot(test_indices, cumulative_buy_and_hold_returns, label='Buy and Hold Returns')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.title('Model vs Buy and Hold Returns on ' + ticker)
plt.legend()

# plot actual stock prices with green dots for correct predictions and red dots for incorrect predictions
plt.subplot(2, 1, 2)
plt.plot(data.index, data['Close'], label='Stock Price')
correct_indices = test_indices[accuracies]
incorrect_indices = test_indices[~accuracies]
plt.scatter(data.index[correct_indices], data['Close'].values[correct_indices], color='green', label='Correct Predictions', marker='o')
plt.scatter(data.index[incorrect_indices], data['Close'].values[incorrect_indices], color='red', label='Incorrect Predictions', marker='x')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price with Correct/Incorrect Predictions')
plt.legend()

plt.tight_layout()

plt.figure()
plt.subplot(2, 1, 1)
plt.hist(correct_model_returns, bins=20, alpha=0.5, label='Model Returns (Correct Predictions)')
plt.hist(incorrect_model_returns, bins=20, alpha=0.5, label='Model Returns (Incorrect Predictions)')
plt.xlabel('Weekly Return')
plt.ylabel('Frequency')
plt.title('Returns When Model is Correct/Incorrect')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(test_indices, model_returns, label='Model Weekly Returns', linestyle='--', alpha=0.7)
plt.plot(test_indices, buy_and_hold_returns, label='Buy and Hold Weekly Returns', linestyle='--', alpha=0.7)
plt.xlabel('Time')
plt.ylabel('Weekly Returns')
plt.title('Model vs Buy and Hold Weekly Returns on ' + ticker)
plt.legend()

plt.tight_layout()
plt.savefig(f"RF_{ticker}_model_vs_buy_and_hold.png")
plt.show()

window = 252
model_returns_series = pd.Series(model_returns)
buy_and_hold_returns_series = pd.Series(buy_and_hold_returns)

rolling_model = model_returns_series.rolling(window).mean()
rolling_bh = buy_and_hold_returns_series.rolling(window).mean()

plt.figure()
plt.plot(rolling_model, label='Model Rolling Mean Return')
plt.plot(rolling_bh, label='Buy & Hold Rolling Mean Return')
plt.xlabel('Weeks')
plt.ylabel(f'Rolling Mean Return ({window} weeks)')
plt.title('Rolling Mean of Weekly Returns')
plt.legend()
plt.savefig(f"RF_{ticker}_rolling_mean_returns.png")
plt.show()

print("Graphs saved to working directory.")
input("Press Enter to exit...")

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

### investigate weekly return differences between model and buy and hold when the model is correct vs wrong to see if the model is adding value in terms of risk-adjusted returns, not just accuracy.
# weekly returns look more stable for model? cant really tell, might not be. maybe record missed opportunities (weeks where model predicted 0 but buy and hold had a positive return) and bad investments (weeks where model predicted 1 but buy and hold had a negative return) and compare