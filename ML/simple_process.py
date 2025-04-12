import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_features(data):
    """
    Simplified data preparation function
    """
    logger.info("Starting data preprocessing...")
    
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    
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
    
    if price_cols:
        # Use the first identified price column
        price_col = price_cols[0]
        logger.info(f"Using {price_col} as the price column")
        
        # Add standardized price column
        df['price'] = df[price_col]
        
        # Calculate basic returns
        df['return'] = df['price'].pct_change()
        
        # Fill NaNs with 0 for returns
        df['return'] = df['return'].fillna(0)
    else:
        logger.warning("No price column found, skipping return calculations")
    
    # Dataset Splitting - use 70% of data for training, 30% for testing
    # Sort by timestamp first
    df = df.sort_values('timestamp')
    
    # Determine the split point (70% of data)
    split_idx = int(len(df) * 0.7)
    split_date = df.iloc[split_idx]['timestamp']
    
    logger.info(f"Split date: {split_date} (70% of data)")
    
    # Print date range info
    logger.info(f"Data date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Split the dataset
    df_train = df[df['timestamp'] < split_date].copy()
    df_test = df[df['timestamp'] >= split_date].copy()
    
    logger.info(f"Training set: {df_train.shape}, from {df_train['timestamp'].min() if not df_train.empty else 'N/A'} to {df_train['timestamp'].max() if not df_train.empty else 'N/A'}")
    logger.info(f"Testing set: {df_test.shape}, from {df_test['timestamp'].min() if not df_test.empty else 'N/A'} to {df_test['timestamp'].max() if not df_test.empty else 'N/A'}")
    
    return {
        "train": df_train,
        "test": df_test
    }

if __name__ == "__main__":
    # Load the merged data from fetch_data
    print("Loading data file...")
    df = pd.read_csv("data/cryptoquant_metrics.csv")
    print(f"Loaded data shape: {df.shape}")
    print(f"First few columns: {df.columns[:5].tolist()}")
    
    # Run simplified feature preparation
    print("Running feature preparation...")
    output = prepare_features(df)
    
    # Print summary
    print("Train set shape:", output["train"].shape)
    print("Test set shape:", output["test"].shape) 