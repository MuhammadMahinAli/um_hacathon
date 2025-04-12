import os
import pandas as pd
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the cybotrade_datasource module
try:
    import cybotrade_datasource
    from cybotrade_datasource import query_paginated
    USING_REAL_API = True
except ImportError:
    logger.error("cybotrade_datasource import failed - exiting as fallback data is not allowed")
    raise ImportError("cybotrade_datasource is required for fetching data")

# Load environment variables
load_dotenv()
API_KEY = os.environ.get('API_KEY') or os.getenv('API_KEY') or os.getenv('DATASET_API_KEY') or "JiR47aozqWLOiwaQh8nxtCVWxKVcX6KNh7VKPqsLw03aF7M7"

if not API_KEY:
    raise ValueError("API key not found. Please set API_KEY or DATASET_API_KEY in your .env file.")

# Ensure the data directory exists
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Data fetch interval in seconds
FETCH_INTERVAL = 10

async def fetch_data_from_api(topic, start_date, end_date, description=""):
    """
    Generic function to fetch data from Cybotrade API
    
    Args:
        topic (str): API topic string formatted for cybotrade
        start_date (datetime): Start date for data
        end_date (datetime): End date for data
        description (str): Description of the data being fetched
        
    Returns:
        pandas.DataFrame: Fetched data
    """
    logger.info(f"Fetching {description or topic}")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    try:
        data = await query_paginated(
            api_key=API_KEY,
            topic=topic,
            start_time=start_date,
            end_time=end_date
        )
        
        if not data:
            logger.error(f"No data received for {description or topic}")
            raise ValueError(f"No data received from API for {description or topic}")
            
        df = pd.DataFrame(data)
        logger.info(f"Successfully fetched {len(df)} records for {description or topic}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {description or topic}: {str(e)}")
        raise

async def fetch_cryptoquant_topic(topic, start_date, end_date):
    """
    Fetch data for a specific CryptoQuant topic using cybotrade_datasource
    
    Args:
        topic (str): The CryptoQuant topic to fetch
        start_date (datetime): Start date for the query
        end_date (datetime): End date for the query
        
    Returns:
        pandas.DataFrame: The fetched data as a DataFrame
    """
    logger.info(f"Fetching data for topic: {topic}")
    
    try:
        return await fetch_data_from_api(
            topic=topic,
            start_date=start_date,
            end_date=end_date,
            description=f"CryptoQuant topic: {topic}"
        )
        
    except Exception as e:
        logger.error(f"Error fetching {topic}: {str(e)}")
        raise  # No fallback data, propagate the error

async def fetch_coinglass_topic(endpoint, start_date, end_date, params=None):
    """
    Fetch data for a specific Coinglass endpoint using cybotrade_datasource
    
    Args:
        endpoint (str): The Coinglass endpoint to fetch
        start_date (datetime): Start date for the query
        end_date (datetime): End date for the query
        params (dict): Additional parameters for the endpoint
        
    Returns:
        pandas.DataFrame: The fetched data as a DataFrame
    """
    if params is None:
        params = {}
    
    # Format the topic for cybotrade (provider|endpoint)
    formatted_topic = f"coinglass|{endpoint}"
    
    # Add any additional parameters as query parameters
    if params:
        param_str = "&".join([f"{k}={v}" for k, v in params.items()])
        formatted_topic = f"{formatted_topic}?{param_str}"
    
    logger.info(f"Fetching data for Coinglass endpoint: {formatted_topic}")
    
    try:
        return await fetch_data_from_api(
            topic=formatted_topic,
            start_date=start_date,
            end_date=end_date,
            description=f"Coinglass endpoint: {endpoint}"
        )
        
    except Exception as e:
        logger.error(f"Error fetching Coinglass {endpoint}: {str(e)}")
        raise  # No fallback data, propagate the error

async def fetch_all_data():
    """
    Fetch and merge all data from CryptoQuant and Coinglass
    """
    # Define date range
    end_date = datetime.now(timezone.utc)
    start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
    end_date_for_fetch = datetime(2024, 1, 1, tzinfo=timezone.utc)
    
    # Define the CryptoQuant topics to fetch
    cryptoquant_topics = [
        'cryptoquant|btc/inter-entity-flows/miner-to-exchange?from_miner=f2pool&to_exchange=binance&window=day',
        'cryptoquant|btc/exchange-flows/inflow?exchange=binance&window=day',
        'cryptoquant|btc/exchange-flows/outflow?exchange=binance&window=day',
        'cryptoquant|btc/flow-indicator/exchange-whale-ratio?exchange=binance&window=day',
        'cryptoquant|btc/flow-indicator/fund-flow-ratio?exchange=binance&window=day',
        'cryptoquant|btc/flow-indicator/stablecoins-ratio?exchange=binance&window=day',
        'cryptoquant|btc/market-indicator/mvrv?window=day',
        'cryptoquant|btc/market-indicator/sopr-ratio?window=day'
    ]
    
    # Define the Coinglass endpoints to fetch
    coinglass_endpoints = [
        {
            "endpoint": "futures/openInterest/ohlc-history",
            "params": {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "1d"}
        },
        {
            "endpoint": "futures/fundingRate/ohlc-history",
            "params": {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "1d"}
        },
        {
            "endpoint": "futures/liquidation/v2/history",
            "params": {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "1d"}
        },
        {
            "endpoint": "price/ohlc-history",
            "params": {"exchange": "Binance", "symbol": "BTCUSDT", "type": "futures", "interval": "1h", "limit": "10"}
        }
    ]
    
    # Create a dictionary to store all DataFrames
    all_dataframes = {}
    
    # Fetch CryptoQuant topics
    for topic in cryptoquant_topics:
        try:
            # Extract a short name from the topic
            short_name = topic.split('|')[-1].split('?')[0].replace('/', '_')
            df = await fetch_cryptoquant_topic(topic, start_date, end_date_for_fetch)
            
            if not df.empty:
                # Save individual DataFrame to CSV for inspection
                df_path = DATA_DIR / f"{short_name}.csv"
                df.to_csv(df_path, index=False)
                logger.info(f"Saved {short_name} to {df_path}")
                
                all_dataframes[short_name] = df
        except Exception as e:
            logger.error(f"Error processing CryptoQuant topic {topic}: {str(e)}")
            # Don't use fallback data, but continue with other topics
    
    # Fetch Coinglass endpoints
    for endpoint_info in coinglass_endpoints:
        try:
            endpoint = endpoint_info["endpoint"]
            params = endpoint_info["params"]
            
            # Extract a short name from the endpoint
            short_name = f"coinglass_{endpoint.replace('/', '_')}"
            df = await fetch_coinglass_topic(endpoint, start_date, end_date_for_fetch, params)
            
            if not df.empty:
                # Save individual DataFrame to CSV for inspection
                df_path = DATA_DIR / f"{short_name}.csv"
                df.to_csv(df_path, index=False)
                logger.info(f"Saved {short_name} to {df_path}")
                
                all_dataframes[short_name] = df
        except Exception as e:
            logger.error(f"Error processing Coinglass endpoint {endpoint}: {str(e)}")
            # Don't use fallback data, but continue with other endpoints
    
    if not all_dataframes:
        logger.error("No data was fetched for any topic or endpoint.")
        raise ValueError("Failed to fetch any data")
    
    # Debug print of all dataframes before merging
    logger.info("\nDataFrames before merging:")
    for name, df in all_dataframes.items():
        logger.info(f"{name}: {df.shape} - Columns: {df.columns.tolist()}")
    
    # Initialize a list to store processed DataFrames
    dfs_to_merge = []
    
    # Process each DataFrame before merging
    for name, df in all_dataframes.items():
        if df.empty:
            continue
            
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Ensure we have a timestamp/date column for merging
        time_col = None
        for col_name in ['start_time', 'time', 'timestamp', 'date']:
            if col_name in processed_df.columns:
                time_col = col_name
                break
        
        if time_col:
            # Convert time column to datetime if it's not already
            processed_df[time_col] = pd.to_datetime(processed_df[time_col])
            
            # Standardize time column name to 'timestamp' for merging
            if time_col != 'timestamp':
                processed_df.rename(columns={time_col: 'timestamp'}, inplace=True)
            
            # Add prefix to all columns except timestamp
            for col in processed_df.columns:
                if col != 'timestamp':
                    processed_df.rename(columns={col: f"{name}_{col}"}, inplace=True)
            
            # Add to list of DataFrames to merge
            dfs_to_merge.append(processed_df)
        else:
            logger.warning(f"WARNING: No suitable time column found in {name}. Available columns: {processed_df.columns.tolist()}")
    
    if not dfs_to_merge:
        logger.error("No DataFrames to merge. Check if timestamp columns exist.")
        raise ValueError("No suitable data for merging found")
    
    # Merge all DataFrames on the timestamp column
    logger.info(f"\nMerging {len(dfs_to_merge)} DataFrames on timestamp column...")
    
    # Start with the first DataFrame
    merged_df = dfs_to_merge[0]
    
    # Merge with the rest one by one
    for i, df in enumerate(dfs_to_merge[1:], 2):
        logger.info(f"Merging DataFrame {i}/{len(dfs_to_merge)}...")
        merged_df = pd.merge(merged_df, df, on='timestamp', how='outer')
    
    # Sort by timestamp
    merged_df = merged_df.sort_values('timestamp')
    
    # Add a data source column to indicate this is real API data
    merged_df['data_source'] = 'REAL'
    
    # Save to CSV
    output_path = DATA_DIR / "cryptoquant_metrics.csv"
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved merged data to {output_path}")
    
    # Print some statistics
    logger.info(f"\nMerged DataFrame shape: {merged_df.shape}")
    logger.info(f"Number of columns: {len(merged_df.columns)}")
    logger.info("\nColumns in the merged DataFrame:")
    for col in merged_df.columns:
        logger.info(f"  - {col}")
    
    return merged_df

async def continuous_data_fetch():
    """
    Continuously fetch data at regular intervals
    """
    logger.info(f"Starting continuous data fetch with {FETCH_INTERVAL} second interval")
    
    while True:
        try:
            logger.info("Fetching fresh data...")
            await fetch_all_data()
            logger.info(f"Data fetch completed. Waiting {FETCH_INTERVAL} seconds for next fetch...")
        except Exception as e:
            logger.error(f"Error in data fetch cycle: {str(e)}")
            logger.error("API data fetch failed. No fallback data will be used.")
        
        # Wait for the next fetch cycle
        await asyncio.sleep(FETCH_INTERVAL)

def fetch_all_data_sync():
    """
    Synchronous wrapper for fetch_all_data
    """
    return asyncio.run(fetch_all_data())

if __name__ == "__main__":
    try:
        # For single fetch use:
        # df = fetch_all_data_sync()
        # print(f"Fetched data with {len(df)} rows and {len(df.columns)} columns")
        
        # For continuous fetching:
        asyncio.run(continuous_data_fetch())
    except Exception as e:
        logger.error(f"Error in main: {str(e)}") 