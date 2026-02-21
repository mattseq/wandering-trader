import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ticker = "BA"

data = yf.download(ticker, start="2015-01-01", end="2025-01-01")

if (data.empty):
    print("No data found for ticker:", ticker)
    exit(0)

print(data.head())

data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

split_idx = int(0.8 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

# use the last 60 days of features as sequence input
sequence_length = 60

# features to use for prediction, output is next day's close price
features = ['Close', 'Volume', 'High', 'Low']

# standardize features, fit scaler only on training data to avoid data leakage
x_scaler = MinMaxScaler()
x_scaler.fit(train_data[features])
y_scaler = MinMaxScaler()
y_scaler.fit(train_data['Close'].values.reshape(-1, 1))

# prep sequences for LSTM
def create_sequences(X, y, sequence_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)

# create sequences for training and testing data
X_train, y_train = create_sequences(train_data[features].values, train_data['Close'].values, sequence_length)
X_test, y_test = create_sequences(test_data[features].values, test_data['Close'].values, sequence_length)

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
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        # self.dropout3 = nn.Dropout(0.2)

        self.linear = nn.Linear(50, 1)
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
    
model = LSTMModel(input_size=len(features))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

model.to(device)

epochs = 100

# early stopping
best_loss = float('inf')
patience = 20
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
        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

    # evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

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
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# inverse transform predictions and actual values to get back to original scale
y_test_inv = y_scaler.inverse_transform(y_test.cpu().numpy())
test_outputs_inv = y_scaler.inverse_transform(test_outputs.cpu().numpy())

# plot actual vs predicted close prices
plt.plot(y_test_inv, label='Actual')
plt.plot(test_outputs_inv, label='Predicted')
plt.legend()
plt.show()

# Calculate actual returns (buy and hold)
actual_returns = np.diff(y_test_inv.squeeze()) / y_test_inv.squeeze()[:-1]
actual_cum_returns = np.cumprod(1 + actual_returns) - 1

# model strategy: buy if model predicts next close > current close
predicted_next = test_outputs_inv.squeeze()[1:]
current = test_outputs_inv.squeeze()[:-1]
signal = predicted_next > current

# model returns: only take return if signal is true, else no return
model_returns = actual_returns * signal
model_cum_returns = np.cumprod(1 + model_returns) - 1

plt.plot(actual_cum_returns, label='Buy and Hold (Actual)')
plt.plot(model_cum_returns, label='Model Strategy')
plt.legend()
plt.title("Cumulative Returns: Buy and Hold vs Model Strategy")
plt.show()