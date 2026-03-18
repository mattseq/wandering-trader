# Wandering Trader

A machine learning project for stock direction prediction and strategy backtesting.

The main work in this repo is:

- **LSTM + Attention** (deep sequence model)
- **Random Forest** (tree-based classifier)

Gradient boosting models are also included.

## Project Structure

- `lstm/lstm.py` - LSTM with attention, feature engineering, training loop, and backtest plots
- `random_forest/random_forest.py` - Random Forest classifier and strategy evaluation
- `gradient_boosting/gboost.py` - Gradient Boosting classifier experiments
- `gradient_boosting/xgboost_model.py` - XGBoost classifier experiments

## What This Project Does

1. Downloads OHLCV market data from Yahoo Finance
2. Builds technical and volatility-driven features (returns, moving averages, RSI, MACD, Bollinger bands, ATR, VIX features)
3. Trains predictive models for next-move direction or next-step price
4. Converts predictions into trading signals
5. Compares strategy performance vs buy-and-hold

## Models

## 1) LSTM + Attention (Main Model)

The LSTM workflow is in `lstm/lstm.py`.

What it does:

- Uses sliding sequences of recent bars as input
- Predicts the next close value
- Converts predicted next-close vs current-close into a trading signal
- Backtests that signal stream against buy-and-hold

Feature set:

- Price and range: Close, High, Low, HL range, OC diff
- Momentum: Return_1, Return_5, Return_10
- Trend: MA5, MA10, MACD, MACD signal
- Volatility/risk: rolling volatility, ATR, Bollinger bands
- Volume context: Volume, Volume_Change
- External regime context: VIX close and short-horizon VIX returns

Practical highlights:

- Sequence modeling with configurable sequence length
- Attention layer over LSTM outputs to weight informative timesteps
- Direction-aware custom loss (`directional_loss`)
- Early stopping + learning-rate scheduler
- Backtesting with multiple baselines:
  - Buy and hold
  - Model strategy
  - Naive momentum
  - Agreement (smart) strategy
- Visualization of:
  - Predicted vs actual prices
  - Cumulative returns and random-signal benchmark bands
  - Attention weights across sequence positions

## 2) Random Forest (Primary Tree Model)

Main script is `random_forest/random_forest.py`.

What it does:

- Builds tabular technical features per bar
- Trains a classifier for direction (positive/negative forward return)
- Retrains on rolling windows to mimic live updating
- Converts class predictions into long/cash trades

Feature set:

- Returns across multiple horizons
- Moving averages and MA distance features
- Volatility features from rolling return windows
- Volume rolling features
- VIX level and VIX return features

Practical highlights:

- Expanding window retraining approach
- Exponential sample weighting to prioritize recent history
- Reports classification and strategy metrics (accuracy, cumulative/annualized return, Sharpe)
- Diagnostic plots for correct/incorrect prediction periods and return distributions

## 3) Gradient Boosting (Experimental)

Located in `gradient_boosting/`:

- `gboost.py`
- `xgboost_model.py`

Status:

- These are experiments and are not tuned as heavily as the LSTM and Random Forest workflows

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install

```bash
pip install -r requirements.txt
```

## Run

### LSTM

```bash
python lstm/lstm.py
```

Notes:

- Script prompts for ticker input
- Uses recent date windows

### Random Forest

```bash
python random_forest/random_forest.py
```

### Gradient Boosting

```bash
python gradient_boosting/gboost.py
python gradient_boosting/xgboost_model.py
```

## Releases

Prebuilt Windows executables will be published in GitHub Releases for:

- LSTM
- Random Forest

## Current Focus and Status

- Most mature and actively worked areas: **LSTM** and **Random Forest**
- Gradient boosting scripts are experimental
