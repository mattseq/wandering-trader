import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# TODO: consider using a single config var for all these settings so its easier to pass into functions

TICKER = "IBM"
START = "2005-01-01"
END = "2025-01-01"

SEQUENCE_LENGTH = 10
TRAIN_SPLIT = 0.6

HIDDEN_SIZE = 32
NUM_LAYERS = 3
DROPOUT = 0.3
BATCH_SIZE = 64

EPOCHS = 300
LR = 0.0001
WEIGHT_DECAY = 1e-3

PATIENCE = 40

SCHEDULER_PATIENCE = 15
SCHEDULER_FACTOR = 0.5

QUANTILE = 0.5
DIRECTIONAL_PENALTY = 0.2
THRESHOLD = 0.001

FEATURES = [
    'Close',
    'High',
    'Low',
    'Volume',

    'MA5',
    'MA10',

    'Return_1',
    'Return_5',
    'Return_10',

    'HL_range',
    'OC_diff',

    'Volatility',

    'Volume_Change',

    'RSI',
    'MACD',
    'MACD_signal',
    'BB_upper',
    'BB_lower',
    'ATR',

    'VIX_Close',
    'VIX_Return_1',
    'VIX_Return_5',
    'VIX_Return_10',
]

def download_data():
    data = yf.download(TICKER, start=START, end=END)
    vix = yf.download("^VIX", start=START, end=END)
    vix = vix.reindex(data.index).ffill()
    if data.empty:
        print("No data found for ticker:", TICKER)
        exit(0)
    return data, vix

def add_features(data, vix):
    data = data.copy()

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

    # TODO: use log returns as target, or maybe just binary signal
    # dont forget the target var
    data['Target'] = data['Close'].shift(-1)
    # data['Target'] = np.log(data['Return_1'] + 1).shift(-1)
    
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)

    return data

def split_and_scale(data):
    split_idx = int(TRAIN_SPLIT * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    x_scaler = MinMaxScaler()
    x_scaler.fit(train_data[FEATURES])
    y_scaler = MinMaxScaler()
    y_scaler.fit(train_data['Target'].values.reshape(-1, 1))

    return train_data, test_data, x_scaler, y_scaler

# prep sequences for LSTM
def create_sequences(X, y):
    X_seq = []
    y_seq = []
    for i in range(len(X) - SEQUENCE_LENGTH):
        X_seq.append(X[i:i+SEQUENCE_LENGTH])
        y_seq.append(y[i+SEQUENCE_LENGTH-1])
    return np.array(X_seq), np.array(y_seq).reshape(-1, 1)

def prepare_tensors(train_data, test_data, x_scaler, y_scaler):

    # create sequences for training and testing data
    X_train, y_train = create_sequences(train_data[FEATURES].values, train_data['Target'].values)
    X_test, y_test = create_sequences(test_data[FEATURES].values, test_data['Target'].values)

    # standardize features using the scaler fit on training data
    X_train = x_scaler.transform(X_train.reshape(-1, len(FEATURES))).reshape(X_train.shape)
    X_test = x_scaler.transform(X_test.reshape(-1, len(FEATURES))).reshape(X_test.shape)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)

    # convert to tensors and move to gpu
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train, y_train, X_test, y_test

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_targets=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)

        self.attention = nn.Linear(hidden_size, 1)

        self.linear = nn.Linear(hidden_size, num_targets)
    def forward(self, x):
        # _ is the hidden state, not needed for prediction so just ignore it
        out, _ = self.lstm(x)

        # use self.attention layer to get weights for each time step, basically "attention" over the sequence output by the lstm
        attn_weights = torch.softmax(self.attention(out), dim=1)
        # weigth the initial lstm output by the attention weights and sum over the sequence dimension to get back to (batch_size, hidden_size)
        out = torch.sum(out * attn_weights, dim=1)

        out = self.linear(out)

        return out
    
def quantile_loss(y_pred, y_true):
    error = y_true - y_pred
    return torch.mean(torch.max(QUANTILE * error, (QUANTILE - 1) * error))

def directional_loss(y_pred, y_true):
    mse = nn.MSELoss()(y_pred, y_true)
    # Add penalty for wrong direction
    pred_sign = torch.sign(y_pred)
    true_sign = torch.sign(y_true)
    direction_penalty = torch.mean((pred_sign != true_sign).float())
    return mse + DIRECTIONAL_PENALTY * direction_penalty

def train(model, X_train, y_train, X_test, y_test):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=0.5
    )

    # early stopping
    best_loss = float('inf')
    patience = 40
    counter = 0

    # create dataloader for training data
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_model_state = None

    # training loop
    for epoch in range(EPOCHS):
        # main training loop
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = directional_loss(outputs, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = directional_loss(test_outputs, y_test)

        scheduler.step(test_loss)

        if test_loss.item() < best_loss:
            best_loss = test_loss.item()
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1

        # print training and test loss
        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Training Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

        # check for early stopping
        if counter >= patience:
            print("Early stopping triggered.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def evaluate(model, X_test, y_test, y_scaler):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = directional_loss(test_outputs, y_test)
        print(f"Test Loss: {test_loss.item():.4f}")

    # inverse transform predictions and actual values to get back to original scale
    y_test_inv = y_scaler.inverse_transform(y_test.cpu().numpy())
    test_outputs_inv = y_scaler.inverse_transform(test_outputs.cpu().numpy())

    return y_test_inv, test_outputs_inv

def get_model_signals(actual, preds_inv):
    pred_next = preds_inv.squeeze()[1:]
    actual_curr = actual.squeeze()[:-1]

    signal = pred_next > actual_curr

    # signal = (preds_inv.squeeze() > 0)[:-1]

    return signal

def get_attention_weights(model, X_sample):
    model.eval()
    with torch.no_grad():
        out, _ = model.lstm(X_sample)
        attn_weights = torch.softmax(model.attention(out), dim=1)
    return attn_weights.squeeze().cpu().numpy()

def backtest(actual_prices, preds):
    actual_returns = np.diff(actual_prices.squeeze()) / actual_prices.squeeze()[:-1]
    actual_cum = np.cumprod(1 + actual_returns) - 1

    signal = get_model_signals(actual_prices, preds)
    model_returns = actual_returns * signal
    model_cum = np.cumprod(1 + model_returns) - 1

    # smart_signal = pred_next > (pred_curr * (1 + THRESHOLD))
    # smart_returns = actual_returns * smart_signal
    # smart_cum = np.cumprod(1 + smart_returns) - 1

    naive_signal = np.roll(actual_returns > 0, 1)
    naive_signal[0] = False
    naive_returns = actual_returns * naive_signal
    naive_cum = np.cumprod(1 + naive_returns) - 1

    return {
        'actual': actual_cum,
        'model': model_cum,
        # 'smart': smart_cum,
        'naive': naive_cum,
        'actual_returns': actual_returns,
        'model_signals': signal,
    }

def analyze_predictions(y_test_inv, preds_inv):
    
    actual_prices = y_test_inv.squeeze()
    predicted_prices = preds_inv.squeeze()
    
    # calculate returns from prices
    actual_returns = np.diff(actual_prices) / actual_prices[:-1]
    pred_returns = np.diff(predicted_prices) / predicted_prices[:-1]
    
    # correlation between actual and predicted returns
    correlation = np.corrcoef(actual_returns, pred_returns)[0,1]
    print(f"Return Prediction Correlation: {correlation:.4f}")
    
    # directional accuracy of returns
    actual_direction = (actual_returns > 0).astype(int)
    pred_direction = (pred_returns > 0).astype(int)
    directional_accuracy = np.mean(actual_direction == pred_direction)
    print(f"Directional Accuracy: {directional_accuracy:.2%}")
    
    return correlation, directional_accuracy

def sharpe_ratio(returns, annualize=252):
    return (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(annualize)

def print_metrics(results):
    print(f"\nSharpe (Model):       {sharpe_ratio(results['actual_returns'] * (results['model_signals'])):.2f}")
    print(f"Sharpe (Buy & Hold):  {sharpe_ratio(results['actual_returns']):.2f}")
    print(f"Total Return (Model): {results['model'][-1]:.2%}")
    print(f"Total Return (B&H):   {results['actual'][-1]:.2%}")

def plot_predictions(y_test_inv, preds_inv):
    plt.figure()
    plt.plot(y_test_inv, label='Actual')
    plt.plot(preds_inv, label='Predicted')
    plt.legend()
    plt.title("Actual vs Predicted Close Price")
    plt.show()

def random_signal_returns(actual_returns, n_simulations=200, seed=42):
    # generate random buy/sell signals and calculate cumulative returns for each simulation
    rng = np.random.default_rng(seed)
    sim_cum_returns = []
    
    for _ in range(n_simulations):
        signal = rng.integers(0, 2, size=len(actual_returns)).astype(bool)
        returns = actual_returns * signal
        cum = np.cumprod(1 + returns) - 1
        sim_cum_returns.append(cum)
    
    return np.array(sim_cum_returns)  # shape: [n_simulations, n_days]


def plot_cumulative_returns(results, n_simulations=200):
    sims = random_signal_returns(results['actual_returns'], n_simulations)
    
    # percentiles of random simulations as bands to show distribution
    p10 = np.percentile(sims, 10, axis=0)
    p25 = np.percentile(sims, 25, axis=0)
    p50 = np.percentile(sims, 50, axis=0)
    p75 = np.percentile(sims, 75, axis=0)
    p90 = np.percentile(sims, 90, axis=0)

    # percentile bands
    plt.fill_between(range(len(p10)), p10, p90, alpha=0.15, color='gray', label='Random 10th-90th Percentile')
    plt.fill_between(range(len(p25)), p25, p75, alpha=0.25, color='gray', label='Random 25th-75th Percentile')
    plt.plot(p50, color='gray', linewidth=1.5, linestyle='--', label='Random median')

    # other strategies
    plt.plot(results['actual'], color='black', linewidth=2, label='Buy and Hold')
    plt.plot(results['model'], color='blue', linewidth=2, label='Model Strategy')
    # plt.plot(results['smart'], color='green', linewidth=2, label='Smart Model Strategy')
    plt.plot(results['naive'], color='orange', linewidth=2, label='Naive Momentum')
    
    plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
    plt.legend()
    plt.title(f"Strategy Returns — {TICKER}")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Trading Days")
    plt.tight_layout()
    plt.show()

    # print where model stands in random distribution
    model_final = results['model'][-1]
    pct_beaten = np.mean(sims[:, -1] < model_final) * 100
    print(f"Model beats {pct_beaten:.1f}% of random strategies")


def plot_attention(model, X_test, n_samples=200):
    all_weights = []
    for i in range(n_samples):
        sample = X_test[i:i+1] # gotta use i:i+1 to preserve shape (1, seq_len, num_features)
        weights = get_attention_weights(model, sample)
        all_weights.append(weights)
    
    mean_weights = np.mean(all_weights, axis=0)
    std_weights = np.std(all_weights, axis=0)

    plt.figure()
    plt.bar(range(len(mean_weights)), mean_weights, yerr=std_weights, alpha=0.7)
    plt.xlabel("Timestep in Sequence")
    plt.ylabel("Mean Attention Weight")
    plt.title("Average Attention Weights Across 200 Test Samples")
    plt.show()


def main():
    # get data and add features
    raw_data, vix_data = download_data()
    data = add_features(raw_data, vix_data)

    # prepare data for training
    train_data, test_data, x_scaler, y_scaler = split_and_scale(data)
    X_train, y_train, X_test, y_test = prepare_tensors(train_data, test_data, x_scaler, y_scaler)

    model = LSTMModel(input_size=len(FEATURES), hidden_size=HIDDEN_SIZE).to(device)

    model = train(model, X_train, y_train, X_test, y_test)
    y_test_inv, test_outputs_inv = evaluate(model, X_test, y_test, y_scaler)

    actual_prices = test_data['Close'].values[SEQUENCE_LENGTH:]
    results = backtest(actual_prices, test_outputs_inv)

    print_metrics(results)


    plot_predictions(y_test_inv, test_outputs_inv)
    plot_cumulative_returns(results)
    plot_attention(model, X_test)

    analyze_predictions(y_test_inv, test_outputs_inv)

if __name__ == "__main__":
    main()