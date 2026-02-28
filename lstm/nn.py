import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ticker = "SPY"

data = yf.download(ticker, start="2015-01-01", end="2025-01-01")
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
data['weekly_return'] = close.pct_change(5).shift(-5)
data['daily_return'] = close.pct_change(1).shift(-1)

# target var is whether the return the next day is positive (1) or negative (0)
data['target'] = (data['weekly_return'] > 0).astype(int)

data.dropna(inplace=True)

print(data.head())


# inputs and labels
features = ['ret_1', 'ret_5', 'ret_10', 'ret_20', 'ret_30', 'ma_10', 'ma_20', 'ma_30', 'ma_diff_10', 'ma_diff_20', 'ma_diff_30', 'vol_10', 'vol_20', 'vol_30']
X = data[features].values
y = data['target'].values.reshape(-1, 1)

# standardize features
scalar = StandardScaler()
X = scalar.fit_transform(X)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# convert to tensors and move to gpu
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
model = Net(len(features)).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5000

for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

        with torch.no_grad():
            preds = model(X_test)
            predicted = (preds > 0.5).float()
            accuracy = (predicted == y_test).float().mean()
            print(f'Accuracy: {accuracy.item():.4f}')

with torch.no_grad():
    preds = model(X_test)
    predicted = (preds > 0.5).float()
    accuracy = (predicted == y_test).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')
    
    # std
    print(f'Std of predictions: {preds.std().item():.4f}')
    print(f'Std of actuals: {y_test.std().item():.4f}')

    weekly_returns = torch.tensor(data['weekly_return'].values[train_size:], dtype=torch.float32).to(device)

    strategy_returns = weekly_returns * predicted.squeeze()

    strategy_cum_returns = torch.cumprod(1 + strategy_returns, dim=0) - 1
    buy_hold_cum_returns = torch.cumprod(1 + weekly_returns, dim=0) - 1

    strategy_cum_returns = strategy_cum_returns.cpu().numpy()
    buy_hold_cum_returns = buy_hold_cum_returns.cpu().numpy()

    plt.plot(buy_hold_cum_returns, label='Buy and Hold')
    plt.plot(strategy_cum_returns, label='Model Strategy')
    plt.xlabel('Weeks')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.title('Cumulative Return: Model vs Buy and Hold')
    plt.show()