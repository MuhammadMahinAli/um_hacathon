import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

# Try to import talib, but make it optional
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Will use alternative calculations for technical indicators.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_features(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Transform raw merged data into clean, model-ready features for machine learning and backtesting.
    
    Args:
        data (pd.DataFrame): Merged DataFrame containing data from various sources
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing train and test DataFrames
                                {"train": df_train, "test": df_test}
    """
    logger.info("Starting data preprocessing...")
    
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    
    if 'timestamp' not in data.columns:
        raise ValueError("Input DataFrame must contain a 'timestamp' column")
    
    # 1. Data Preprocessing
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Drop duplicates based on timestamp
    df = df.drop_duplicates(subset=['timestamp'])
    
    # Sort chronologically
    df = df.sort_values('timestamp')
    
    # Identify price column from Coinglass data
    price_cols = [col for col in df.columns if 'coinglass_price_ohlc-history_c' in col]
    if not price_cols:
        # Try to find any price-related column
        price_cols = [col for col in df.columns if ('price' in col.lower() and 'close' in col.lower()) or 
                     ('ohlc' in col.lower() and 'c' in col.lower())]
    
    if not price_cols:
        raise ValueError("No suitable price column found in the data")
    
    # Use the first identified price column
    price_col = price_cols[0]
    logger.info(f"Using {price_col} as the price column")
    
    # Add standardized price column
    df['price'] = df[price_col].astype(float)
    
    # 2. Feature Engineering
    # Calculate returns and log returns
    df['return'] = df['price'].pct_change()
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    
    # Create lagged returns (1-day, 3-day, 5-day)
    for lag in [1, 3, 5]:
        df[f'lag_return_{lag}'] = df['return'].shift(lag)
    
    # Calculate rolling statistics
    windows = [7, 14, 30]
    
    for window in windows:
        # Simple Moving Average
        df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
        
        # Exponential Moving Average
        df[f'ema_{window}'] = df['price'].ewm(span=window, adjust=False).mean()
        
        # Volatility (rolling standard deviation)
        df[f'volatility_{window}'] = df['return'].rolling(window=window).std()
    
    # RSI calculation
    if TALIB_AVAILABLE:
        df['rsi_14'] = talib.RSI(df['price'].values, timeperiod=14)
    else:
        # Fallback RSI calculation
        delta = df['price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Feature for future return (target variable)
    df['future_return_24h'] = df['return'].shift(-24)  # 24-hour ahead return
    df['future_binary_24h'] = (df['future_return_24h'] > 0).astype(int)
    
    # 3. Dataset Splitting
    # Define the split point (use 70% for training, 30% for testing)
    split_idx = int(len(df) * 0.7)
    split_date = df.iloc[split_idx]['timestamp']
    
    logger.info(f"Split date: {split_date} (70% of data)")
    
    # Split the dataset
    df_train = df[df['timestamp'] < split_date].copy()
    df_test = df[df['timestamp'] >= split_date].copy()
    
    logger.info(f"Training set: {df_train.shape}, from {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
    logger.info(f"Testing set: {df_test.shape}, from {df_test['timestamp'].min()} to {df_test['timestamp'].max()}")
    
    # Fill any remaining NaNs with appropriate values
    numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df_train[col].isna().any() or df_test[col].isna().any():
            # Use median for the column
            train_median = df_train[col].median()
            df_train[col] = df_train[col].fillna(train_median)
            df_test[col] = df_test[col].fillna(train_median)
    
    return {
        "train": df_train,
        "test": df_test
    }

if __name__ == "__main__":
    # Load the merged data from fetch_data
    df = pd.read_csv("data/cryptoquant_metrics.csv")  # make sure this file exists

    # Run feature preparation
    output = prepare_features(df)

    # Print summary
    print("Train set shape:", output["train"].shape)
    print("Test set shape:", output["test"].shape) 