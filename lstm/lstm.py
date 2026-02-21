import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ticker = "IBM"

data = yf.download(ticker, start="2010-01-01", end="2025-01-01")
vix = yf.download("^VIX", start="2010-01-01", end="2025-01-01")

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

data['vix_close'] = vix['Close'].values
data['vix_ret_1'] = data['vix_close'].pct_change(1)

# data['weekly_return'] = close.pct_change(5).shift(-5)
data['daily_return'] = close.pct_change(1).shift(-1)

# data['log_ret_1'] = np.log(close / close.shift(1))
# data['vix_log_ret_1'] = np.log(vix['Close'] / vix['Close'].shift(1))
# data['log_ret_5'] = np.log(close / close.shift(5))
# data['log_ret_10'] = np.log(close / close.shift(10))

data['ma_10'] = close.rolling(10).mean()
data['ma_20'] = close.rolling(20).mean()
data['ma_30'] = close.rolling(30).mean()
data['ma_diff_10'] = close - data['ma_10']
data['ma_diff_20'] = close - data['ma_20']
data['ma_diff_30'] = close - data['ma_30']
data['ma_10_ratio'] = close / data['ma_10']
data['ma_20_ratio'] = close / data['ma_20']
data['ma_30_ratio'] = close / data['ma_30']

data['vol_10'] = data['ret_5'].rolling(10).std()
data['vol_20'] = data['ret_10'].rolling(20).std()
data['vol_30'] = data['ret_20'].rolling(30).std()


# target var is whether the return the next day is positive (1) or negative (0)
data['target'] = (data['daily_return'] > 0).astype(int)
# data['target'] = data['log_ret_1'].shift(-1)

print(data.head())

data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

# use the last 10 days of features as sequence input
sequence_length = 10

# inputs and labels
features = ['ret_1', 'ret_5', 'ret_10', 'ret_20', 'ret_30', 'ma_10', 'ma_20', 'ma_30', 'ma_diff_10', 'ma_diff_20', 'ma_diff_30', 'ma_10_ratio', 'ma_20_ratio', 'ma_30_ratio', 'vol_10', 'vol_20', 'vol_30', 'vix_close', 'vix_ret_1']

scaler = StandardScaler()
scaler.fit(train_data[features])


# prep sequences for LSTM
def create_sequences(X, y, sequence_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)

X_train, y_train = create_sequences(train_data[features].values, train_data['target'].values, sequence_length)
X_test, y_test = create_sequences(test_data[features].values, test_data['target'].values, sequence_length)

X_train = scaler.transform(X_train.reshape(-1, len(features))).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, len(features))).reshape(X_test.shape)

# convert to tensors and move to gpu
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm1 = nn.LSTM(input_size=16, hidden_size=10, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=10, hidden_size=10, num_layers=1, batch_first=True)
        self.linear = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is (batch_size, seq_len, input_size), conv1 expects (batch_size, input_size/channels, seq_len)
        out = self.conv1(x.permute(0, 2, 1))
        out = self.relu(out)

        # switch back to (batch_size, seq_len, features) for LSTM again
        out = out.permute(0, 2, 1)

        out, _ = self.lstm1(out)
        out = self.dropout(out)

        out, _ = self.lstm2(out)
        out = self.dropout(out)

        # lstm outputs sequence of preds with length seq_len, so just get the last output
        out = out[:, -1, :]

        out = self.linear(out)
        out = self.sigmoid(out)
        return out
    
model = LSTMModel(input_size=len(features))

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

model.to(device)

epochs = 100

# early stopping
best_loss = float('inf')
patience = 20
counter = 0

batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# step = 10

# for start in range(0, len(X_train), step):
#     end = start + step
    
#     X_train_window = X_train[start:end]
#     y_train_window = y_train[start:end]
#     X_test_window = X_test[start:end]
#     y_test_window = y_test[start:end]

#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_window)
#     loss = criterion(outputs, y_train_window)

#     loss.backward()
#     optimizer.step()

#     model.eval()
#     with torch.no_grad():
#         test_outputs = model(X_test_window)
#         test_loss = criterion(test_outputs, y_test_window)

#     if test_loss.item() < best_loss:
#         best_loss = test_loss.item()
#         counter = 0
#     else:
#         counter += 1

#     if counter >= patience:
#         print("Early stopping triggered.")
#         break

for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

    if test_loss.item() < best_loss:
        best_loss = test_loss.item()
        counter = 0
    else:
        counter += 1

    if epoch % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    if counter >= patience:
        print("Early stopping triggered.")
        break

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

plt.plot(y_test.cpu().numpy(), label='Actual')
plt.plot(test_outputs.cpu().numpy(), label='Predicted')
plt.legend()
plt.show()

daily_returns = test_data['daily_return'].values[sequence_length:]

buy = test_outputs.cpu().numpy().flatten() > 0.5
model_returns = buy * daily_returns
model_cumulative_returns = np.cumprod(1 + model_returns) - 1
actual_cumulative_returns = np.cumprod(1 + daily_returns) - 1

plt.plot(actual_cumulative_returns, label='Actual Cumulative Returns')
plt.plot(model_cumulative_returns, label='Model Cumulative Returns')
plt.legend()
plt.show()