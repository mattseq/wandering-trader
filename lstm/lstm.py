import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ticker = "IBM"

data = yf.download(ticker, start="2005-01-01", end="2025-01-01")
vix = yf.download("^VIX", start="2005-01-01", end="2025-01-01")
vix = vix.reindex(data.index).fillna(method='ffill')

if (data.empty):
    print("No data found for ticker:", ticker)
    exit(0)

print(data.head())

data['MA5'] = data['Close'].rolling(5).mean()
data['MA10'] = data['Close'].rolling(10).mean()
data['Return_1'] = data['Close'].pct_change()
data['Return_5'] = data['Close'].pct_change(5)
data['Return_10'] = data['Close'].pct_change(10)
data['HL_range'] = data['High'] - data['Low']
data['OC_diff'] = data['Open'] - data['Close']
data['Volatility'] = data['Return_1'].rolling(10).std()
data['Volume_Change'] = data['Volume'].pct_change()

delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

ema12 = data['Close'].ewm(span=12, adjust=False).mean()
ema26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema12 - ema26
data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

ma20 = data['Close'].rolling(20).mean()
std20 = data['Close'].rolling(20).std()
data['BB_upper'] = ma20 + 2 * std20
data['BB_lower'] = ma20 - 2 * std20

high_low = data['High'] - data['Low']
high_close = np.abs(data['High'] - data['Close'].shift())
low_close = np.abs(data['Low'] - data['Close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
data['ATR'] = tr.rolling(14).mean()

data['VIX_Close'] = vix['Close'].values
data['VIX_Return_1'] = data['VIX_Close'].pct_change()
data['VIX_Return_5'] = data['VIX_Close'].pct_change(5)
data['VIX_Return_10'] = data['VIX_Close'].pct_change(10)


data['Target'] = data['Close'].shift(-1)

data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

split_idx = int(0.6 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

# use the last 60 days of features as sequence input
sequence_length = 60

# features to use for prediction, output is next day's close price
features = ['Close', 'High', 'Low', 'Volume', 'MA5', 'MA10', 'Return_1', 'Return_5', 'Return_10', 'HL_range', 'OC_diff', 'Volatility', 'Volume_Change', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'ATR', 'VIX_Close', 'VIX_Return_1', 'VIX_Return_5', 'VIX_Return_10']
# features = ['Close']

# standardize features, fit scaler only on training data to avoid data leakage
x_scaler = MinMaxScaler()
x_scaler.fit(train_data[features])
y_scaler = MinMaxScaler()
y_scaler.fit(train_data['Target'].values.reshape(-1, 1))

# prep sequences for LSTM
def create_sequences(X, y, sequence_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])
    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)

# create sequences for training and testing data
X_train, y_train = create_sequences(train_data[features].values, train_data['Target'].values, sequence_length)
X_test, y_test = create_sequences(test_data[features].values, test_data['Target'].values, sequence_length)

# standardize features using the scaler fit on training data
X_train = x_scaler.transform(X_train.reshape(-1, len(features))).reshape(X_train.shape)
X_test = x_scaler.transform(X_test.reshape(-1, len(features))).reshape(X_test.shape)
y_train = y_scaler.transform(y_train)
y_test = y_scaler.transform(y_test)


# convert to tensors and move to gpu
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

class LSTMModel(nn.Module):
    def __init__(self, input_size, num_targets=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        # self.dropout3 = nn.Dropout(0.2)

        self.linear = nn.Linear(50, num_targets)
    def forward(self, x):
        # _ is the hidden state, not needed for prediction so just ignore it
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out, _ = self.lstm3(out)
        # out = self.dropout3(out)

        # lstm outputs sequence of preds with length seq_len, so just get the last output
        out = out[:, -1, :]

        out = self.linear(out)

        return out
    
# I AM GRUT
class GRUModel(nn.Module):
    def __init__(self, input_size, num_targets=1):
        super(GRUModel, self).__init__()
        
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=50, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.gru3 = nn.GRU(input_size=50, hidden_size=50, num_layers=1, batch_first=True)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        # self.dropout3 = nn.Dropout(0.2)

        self.linear = nn.Linear(50, num_targets)
    def forward(self, x):
        # _ is the hidden state, not needed for prediction so just ignore it
        out, _ = self.gru1(x)
        out = self.dropout1(out)

        out, _ = self.gru2(out)
        out = self.dropout2(out)

        out, _ = self.gru3(out)
        # out = self.dropout3(out)

        # lstm outputs sequence of preds with length seq_len, so just get the last output
        out = out[:, -1, :]

        out = self.linear(out)

        return out
    
def quantile_loss(y_pred, y_true, quantile=0.7):
    error = y_true - y_pred
    return torch.mean(torch.max(quantile * error, (quantile - 1) * error))

model = LSTMModel(input_size=len(features))

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

model.to(device)

epochs = 300

# early stopping
best_loss = float('inf')
patience = 40
counter = 0

# create dataloader for training data
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

best_model_state = None

# training loop
for epoch in range(epochs):
    # main training loop
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        # loss = criterion(outputs, y_batch)
        loss = quantile_loss(outputs, y_batch)

        loss.backward()
        optimizer.step()

    # evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = quantile_loss(test_outputs, y_test)

    if test_loss.item() < best_loss:
        best_loss = test_loss.item()
        counter = 0
        best_model_state = model.state_dict()
    else:
        counter += 1

    # print training and test loss
    if epoch % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    # check for early stopping
    if counter >= patience:
        print("Early stopping triggered.")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

# final evaluation on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = quantile_loss(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# inverse transform predictions and actual values to get back to original scale
y_test_inv = y_scaler.inverse_transform(y_test.cpu().numpy())
test_outputs_inv = y_scaler.inverse_transform(test_outputs.cpu().numpy())

# plot actual vs predicted close prices
plt.plot(y_test_inv, label='Actual')
plt.plot(test_outputs_inv, label='Predicted')
plt.legend()
plt.show()

print(f"Std of predictions: {test_outputs.std().item():.4f}")
print(f"Std of actuals: {y_test.std().item():.4f}")

# Calculate actual returns (buy and hold)
actual_returns = np.diff(y_test_inv.squeeze()) / y_test_inv.squeeze()[:-1]
actual_cum_returns = np.cumprod(1 + actual_returns) - 1

# generate trading signals based on model predictions
predicted_next = test_outputs_inv.squeeze()[1:]
current = test_outputs_inv.squeeze()[:-1]
signal = predicted_next > current

# model returns: only take return if signal is true, else no return
model_returns = actual_returns * signal
model_cum_returns = np.cumprod(1 + model_returns) - 1

# smart model strategy: adjust signal based on how much the model is predicting the price will increase. Only buy if predicted next close is at least 0.1% higher than current close
threshold = 0.001
smart_signal = predicted_next > (current * (1 + threshold))
smart_model_returns = actual_returns * smart_signal
smart_model_cum_returns = np.cumprod(1 + smart_model_returns) - 1

# naive momentum strategy: buy if previous day's return > 0
naive_signal = np.roll(actual_returns > 0, 1)
naive_signal[0] = False
naive_returns = actual_returns * naive_signal
naive_cum_returns = np.cumprod(1 + naive_returns) - 1

plt.plot(actual_cum_returns, label='Buy and Hold (Actual)')
plt.plot(model_cum_returns, label='Model Strategy')
plt.plot(naive_cum_returns, label='Naive Momentum Strategy')
plt.plot(smart_model_cum_returns, label='Smart Model Strategy')
plt.legend()
plt.title("Cumulative Returns: Strategies vs Buy and Hold on " + ticker)
plt.show()

# Difference between predicted and previous day's actual close
pred_diff = predicted_next - current
actual_diff = y_test_inv.squeeze()[1:] - current
plt.plot(pred_diff, label='Prediction - Previous Close')
plt.plot(actual_diff, label='Actual - Previous Close')
plt.title("Difference Between Prediction and Previous Day's Close")
plt.legend()
plt.show()

# # Recursive forecast for the test set
# recursive_preds = []
# seq = X_test[0].cpu().numpy()

# for i in range(len(X_test)):
#     # Predict next close
#     seq_tensor = torch.tensor(seq.reshape(1, sequence_length, len(features)), dtype=torch.float32).to(device)
#     with torch.no_grad():
#         pred = model(seq_tensor).cpu().numpy()
#     pred_inv = y_scaler.inverse_transform(pred)[0, 0]
#     recursive_preds.append(pred_inv)
    
#     # Prepare next sequence: drop oldest, add new prediction as 'Close'
#     if i < len(X_test) - 1:
#         next_seq = seq[1:].copy()
#         next_row = np.zeros(len(features))
#         next_row[features.index('Close')] = x_scaler.transform([[pred_inv]])[0][0]
#         seq = np.vstack([next_seq, next_row])

# # Plot recursive predictions vs actual
# plt.plot(y_test_inv.squeeze(), label='Actual')
# plt.plot(recursive_preds, label='Recursive Predictions')
# plt.legend()
# plt.title("Recursive Forecast vs Actual")
# plt.show()