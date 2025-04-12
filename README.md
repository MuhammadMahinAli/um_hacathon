# Backtesting Framework

A robust framework for developing, testing, and evaluating trading strategies using machine learning and financial data.

## Overview

This backtesting framework allows you to develop, train, and evaluate trading strategies using historical price data and various indicators. The framework includes components for data fetching, preprocessing, strategy development, backtesting, and performance evaluation.

## Installation

1. Clone the repository:
```
git clone https://github.com/MuhammadMahinAli/um_hacathon
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Requirements

The framework depends on the following Python libraries:
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- hmmlearn
- scipy
- statsmodels
- cybotrade-datasource (for fetching data)

## Project Structure

- `ML/` - Main module containing the framework
  - `backtesting.py` - Core backtesting engine
  - `strategy.py` - Strategy implementation and signal generation
  - `data_process.py` - Data preparation and feature engineering
  - `fetch_data.py` - Data fetching from APIs
  - `data/` - Directory for stored data files
- `example_usage.py` - Example script demonstrating the complete workflow

## How to Use

### 1. Fetching Data

The framework can fetch data from Cybotrade API, which provides access to CryptoQuant and Coinglass data. To fetch data:

```python
from ML.fetch_data import fetch_all_data_sync

# This will fetch data and save it to ML/data/cryptoquant_metrics.csv
df_raw = fetch_all_data_sync()
```

Note: You need to set your API key in a `.env` file or as an environment variable:
```
API_KEY=your_api_key_here
```

### 2. Data Processing and Feature Engineering

```python
from ML.data_process import prepare_features
import pandas as pd

# Load data from CSV
df = pd.read_csv("ML/data/cryptoquant_metrics.csv")

# Process data and split into train/test sets
data_dict = prepare_features(df)
df_train = data_dict["train"]
df_test = data_dict["test"]
```

### 3. Building a Trading Strategy

The framework includes a hybrid strategy that combines XGBoost and Hidden Markov Models (HMM):

```python
from ML.strategy import HybridStrategy

# Initialize and train strategy
strategy = HybridStrategy(n_hmm_components=5, min_trade_freq=0.05)
strategy.fit(df_train)

# Generate trading signals on test data
signals = strategy.generate_signals(df_test)

# Calculate strategy performance metrics
performance = strategy.calculate_performance(df_test, signals)
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
```

### 4. Running a Backtest

```python
from ML.backtesting import Backtest

# Initialize and run backtest
backtest = Backtest(
    data=df_test,
    signals=signals,
    initial_capital=10000,
    fee_rate=0.0004,
    risk_free_rate=0.01
)

# Run the backtest
results_df = backtest.run_backtest()

# Evaluate performance metrics
metrics = backtest.evaluate_performance()
print(f"Final portfolio value: ${results_df['equity'].iloc[-1]:.2f}")
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### 5. Visualizing Results

```python
# Plot the backtest results
backtest.plot_results()
```

## Running the Complete Example

The example_usage.py script demonstrates the end-to-end workflow:

```
python example_usage.py
```

This script will:
1. Fetch data from the API
2. Process and engineer features
3. Train an ML strategy
4. Run a backtest
5. Evaluate and display performance metrics
6. Plot the results

## Creating Custom Strategies

To create your own strategy, you can either extend the `HybridStrategy` class or create a new strategy class. Your custom strategy should implement at least these methods:

- `fit(data)`: Train your strategy on historical data
- `generate_signals(data)`: Generate trading signals (1: buy, 0: hold, -1: sell)
- `calculate_performance(data, signals)`: Calculate performance metrics

## Backtest Parameters

The `Backtest` class accepts the following parameters:

- `data`: DataFrame containing at least timestamp and price columns
- `signals`: List of trade signals (1: buy, 0: hold, -1: sell)
- `initial_capital`: Starting portfolio value (default: 10000)
- `fee_rate`: Fee per transaction as a decimal (default: 0.06%)
- `risk_free_rate`: Annual risk-free rate as a decimal (default: 0.0)

## Performance Metrics

The framework calculates various performance metrics:

- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Max Drawdown
- Total Return
- Annualized Return
- Win Rate
- Trade Frequency 
