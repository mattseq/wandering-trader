import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import torch

print(pd.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data = yf.download("AAPL", start="2010-01-01", end="2015-01-01")
print(data.head())

# data = data[['Close']]

close = data['Close'].squeeze()

print(type(data))
print(type(data['Close']))

# features
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

# drop rows with NaN values, like the first few rows where rolling calculations arent possible
data.dropna(inplace=True)

# target var is whether the return the next day is positive (1) or negative (0)
data['target'] = (data['weekly_return'] > 0).astype(int)
data.dropna(inplace=True)

print(data.head())

# inputs and labels
features = ['ret_5', 'ret_10', 'ret_20', 'ma_10', 'ma_20', 'ma_diff_10', 'ma_diff_20', 'vol_10', 'vol_20']
X = data[features].values
y = data['target'].values.reshape(-1, 1)

# standardize features
scalar = StandardScaler()
X = scalar.fit_transform(X)

# split into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# convert to tensors and move to gpu
# X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
# y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
# X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)

rf.fit(X_train, y_train.ravel())

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Random Forest Accuracy: {accuracy:.4f}')

# class Net(nn.Module):
#     def __init__(self, input_size):
#         super(Net, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x)
    
# model = Net(len(features)).to(device)

# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# epochs = 500

# for epoch in range(epochs):
#     optimizer.zero_grad()

#     outputs = model(X_train)

#     loss = criterion(outputs, y_train)

#     loss.backward()
#     optimizer.step()
    
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# with torch.no_grad():
#     preds = model(X_test)
#     predicted = (preds > 0.5).float()
#     accuracy = (predicted == y_test).float().mean()
#     print(f'Accuracy: {accuracy.item():.4f}')

### Look into random forest, gradient boosting, lstm, transformer models.
### more features like technical indicators, sentiment analysis, macroeconomic data, etc.

### Random Forest
# forest has subtrees which are decision trees (glorified piecewise functions)
# each tree is trained on a random subset of the data and features
# the final prediction is made by averaging the predictions of all the trees (for regression) or taking a majority vote (for classification)

