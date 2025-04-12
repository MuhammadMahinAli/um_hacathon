import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available. Plotting functionality will be disabled.")

class Backtest:
    """
    A standalone class for backtesting trading strategies.
    
    This class simulates trading based on provided price data and strategy signals,
    and calculates key performance metrics for strategy evaluation.
    """
    
    def __init__(self, data: pd.DataFrame, signals: List[int], initial_capital=10000, fee_rate=0.0006, risk_free_rate=0.0):
        """
        Initialize the Backtest class.
        
        Args:
            data (pd.DataFrame): DataFrame containing at least timestamp and price columns
            signals (List[int]): List of trade signals (1: buy, 0: hold, -1: sell)
            initial_capital (float): Starting portfolio value
            fee_rate (float): Fee per transaction as a decimal (default: 0.06%)
            risk_free_rate (float): Annual risk-free rate as a decimal (default: 0.0)
        """
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        
        if 'timestamp' not in data.columns:
            raise ValueError("data must contain a 'timestamp' column")
        
        if 'price' not in data.columns:
            raise ValueError("data must contain a 'price' column")
        
        if len(signals) != len(data):
            raise ValueError(f"Length of signals ({len(signals)}) must match length of data ({len(data)})")
        
        if not all(s in [-1, 0, 1] for s in signals):
            raise ValueError("signals must only contain values -1, 0, or 1")
        
        # Store inputs
        self.data = data.copy()
        self.signals = signals.copy()
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.risk_free_rate = risk_free_rate
        
        # Ensure price is numeric
        self.data['price'] = pd.to_numeric(self.data['price'], errors='coerce')
        
        # Ensure timestamp is in datetime format
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Sort by timestamp
        self.data = self.data.sort_values('timestamp')
        
        # Initialize results
        self.results = None
    
    def run_backtest(self) -> pd.DataFrame:
        """
        Run the backtest simulation.
        
        Returns:
            pd.DataFrame: DataFrame with backtest results including timestamp, price,
                         position, returns, and equity_curve
        """
        logger.info("Running backtest simulation...")
        
        # Create a results DataFrame
        results = self.data[['timestamp', 'price']].copy()
        
        # Add signals to results
        results['signal'] = self.signals
        
        # Initialize portfolio metrics
        results['position'] = 0
        results['capital'] = self.initial_capital
        results['shares'] = 0
        results['fees'] = 0
        results['equity'] = self.initial_capital
        results['returns'] = 0
        
        # Track current position and capital
        current_position = 0
        available_capital = self.initial_capital
        shares_held = 0
        
        # Simulate trading
        for i in range(len(results)):
            # Get the current signal and price
            signal = results['signal'].iloc[i]
            price = results['price'].iloc[i]
            
            # Determine the new position based on the signal
            new_position = signal  # 1 (long), 0 (no position), -1 (short)
            
            # Skip if the position doesn't change and we're not executing a trade
            if new_position == current_position:
                # Update the results for this row
                results['position'].iloc[i] = current_position
                results['capital'].iloc[i] = available_capital
                results['shares'].iloc[i] = shares_held
                results['equity'].iloc[i] = available_capital + (shares_held * price if shares_held > 0 else 0)
                
                # Calculate returns (change in equity)
                if i > 0:
                    prev_equity = results['equity'].iloc[i-1]
                    if prev_equity > 0:
                        results['returns'].iloc[i] = results['equity'].iloc[i] / prev_equity - 1
                
                continue
            
            # Execute the trade - calculate fees and update capital/shares
            if current_position == 0:  # No position -> Taking a position
                if new_position == 1:  # Buy (go long)
                    # Calculate how many shares we can buy with our capital
                    shares_to_buy = available_capital / price
                    # Calculate fees
                    fee = price * shares_to_buy * self.fee_rate
                    # Adjust shares to account for fees
                    shares_to_buy = (available_capital - fee) / price
                    
                    # Update
                    shares_held = shares_to_buy
                    available_capital = 0  # All capital used to buy shares
                    results['fees'].iloc[i] = fee
                    
                elif new_position == -1:  # Sell (go short)
                    # Short selling not implemented in this basic version
                    # Just stay in cash
                    shares_held = 0
                    new_position = 0  # Force to no position
                    
            elif current_position == 1:  # Long -> Closing or changing position
                # Sell all shares
                sale_value = shares_held * price
                fee = sale_value * self.fee_rate
                available_capital = sale_value - fee
                shares_held = 0
                results['fees'].iloc[i] = fee
                
                # If new position is short, implement that
                if new_position == -1:
                    # Short selling not implemented
                    new_position = 0  # Force to no position
                
            elif current_position == -1:  # Short -> Closing or changing position
                # Not implemented - just go to cash
                shares_held = 0
                new_position = 0
                
                # If new position is long, implement that
                if new_position == 1:
                    # Calculate how many shares we can buy with our capital
                    shares_to_buy = available_capital / price
                    # Calculate fees
                    fee = price * shares_to_buy * self.fee_rate
                    # Adjust shares to account for fees
                    shares_to_buy = (available_capital - fee) / price
                    
                    # Update
                    shares_held = shares_to_buy
                    available_capital = 0  # All capital used to buy shares
                    results['fees'].iloc[i] += fee
            
            # Update current position
            current_position = new_position
            
            # Update the results for this row
            results['position'].iloc[i] = current_position
            results['capital'].iloc[i] = available_capital
            results['shares'].iloc[i] = shares_held
            results['equity'].iloc[i] = available_capital + (shares_held * price if shares_held > 0 else 0)
            
            # Calculate returns (change in equity)
            if i > 0:
                prev_equity = results['equity'].iloc[i-1]
                if prev_equity > 0:
                    results['returns'].iloc[i] = results['equity'].iloc[i] / prev_equity - 1
        
        # Calculate cumulative equity curve
        results['equity_curve'] = (1 + results['returns']).cumprod()
        
        # Store results for later use
        self.results = results
        
        logger.info(f"Backtest completed. Final equity: {results['equity'].iloc[-1]:.2f}")
        
        return results
    
    def evaluate_performance(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the backtest.
        
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        if self.results is None:
            self.run_backtest()
        
        # Extract returns series
        returns = self.results['returns']
        
        # Calculate metrics
        
        # Determine data frequency and set appropriate annualization factor
        if len(self.results) > 1:
            time_diff = self.results['timestamp'].diff().median()
            if time_diff.total_seconds() <= 86400:  # Daily or intraday
                ann_factor = 252  # Trading days per year
            elif time_diff.total_seconds() <= 604800:  # Weekly
                ann_factor = 52
            elif time_diff.total_seconds() <= 2678400:  # Monthly
                ann_factor = 12
            else:
                ann_factor = 1  # Assuming yearly data
        else:
            ann_factor = 252  # Default to daily
        
        # Daily risk-free rate
        daily_rf_rate = (1 + self.risk_free_rate) ** (1 / ann_factor) - 1
        
        # 1. Sharpe Ratio (annualized, with risk-free rate)
        excess_returns = returns - daily_rf_rate
        daily_excess_returns_mean = excess_returns.mean()
        daily_returns_std = returns.std()
        
        if daily_returns_std > 0:
            sharpe_ratio = np.sqrt(ann_factor) * daily_excess_returns_mean / daily_returns_std
        else:
            sharpe_ratio = 0  # No volatility case
            logger.warning("Standard deviation of returns is 0, Sharpe ratio set to 0")
        
        # 2. Sortino Ratio (only penalizes downside volatility)
        downside_returns = returns[returns < daily_rf_rate] - daily_rf_rate
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        
        if downside_std > 0:
            sortino_ratio = np.sqrt(ann_factor) * daily_excess_returns_mean / downside_std
        else:
            sortino_ratio = 0  # No downside volatility case
            logger.warning("No downside volatility detected, Sortino ratio set to 0")
        
        # 3. Calmar Ratio (return / max drawdown)
        # 3. Max Drawdown
        equity_curve = self.results['equity_curve'].values
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve / peak) - 1
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio (annualized return / absolute max drawdown)
        # Calculate annualized return first
        total_return = self.results['equity'].iloc[-1] / self.initial_capital - 1
        days = (self.results['timestamp'].iloc[-1] - self.results['timestamp'].iloc[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = total_return
            
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # 4. Trade Frequency
        # Count transitions between different positions
        position_changes = (self.results['position'] != self.results['position'].shift(1)).sum()
        trade_frequency = position_changes / len(self.results)
        
        # 5. Information Ratio (vs. benchmark)
        # Not implemented as we would need benchmark data
        
        # 6. Win Rate
        daily_wins = (returns > 0).sum()
        win_rate = daily_wins / len(returns) if len(returns) > 0 else 0
        
        # Criteria checks
        meets_sharpe = sharpe_ratio >= 1.8
        meets_drawdown = max_drawdown >= -0.4  # -40%
        meets_trade_freq = trade_frequency >= 0.03  # 3%
        meets_all_criteria = meets_sharpe and meets_drawdown and meets_trade_freq
        
        # Return all metrics
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "trade_frequency": trade_frequency,
            "win_rate": win_rate,
            "meets_sharpe_criteria": meets_sharpe,
            "meets_drawdown_criteria": meets_drawdown,
            "meets_trade_freq_criteria": meets_trade_freq,
            "meets_all_criteria": meets_all_criteria,
            "annualization_factor": ann_factor
        }
    
    def plot_results(self, save_path=None):
        """
        Plot the equity curve and price with buy/sell markers.
        
        Args:
            save_path (str, optional): Path to save the plot to. If None, display the plot.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Cannot plot results.")
            return
        
        if self.results is None:
            self.run_backtest()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot equity curve
        ax1.plot(self.results['timestamp'], self.results['equity'], label='Portfolio Value', color='blue')
        ax1.set_title('Backtest Results')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot price and signals
        ax2.plot(self.results['timestamp'], self.results['price'], label='Price', color='black')
        
        # Add buy/sell markers
        buy_signals = self.results[self.results['signal'] == 1]
        sell_signals = self.results[self.results['signal'] == -1]
        
        ax2.scatter(buy_signals['timestamp'], buy_signals['price'], 
                   marker='^', color='green', s=100, label='Buy Signal')
        ax2.scatter(sell_signals['timestamp'], sell_signals['price'], 
                   marker='v', color='red', s=100, label='Sell Signal')
        
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


# Testing block
if __name__ == "__main__":
    # Add parent directory to path to enable imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        # Import necessary modules
        from ML.data_process import prepare_features
        from ML.strategy import HybridStrategy
        
        # Load data
        logger.info("Loading data...")
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cryptoquant_metrics.csv")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found at: {data_file}")
            
        df = pd.read_csv(data_file)
        data = prepare_features(df)
        
        # Train the strategy
        logger.info("Training strategy...")
        strategy = HybridStrategy(n_hmm_components=3, min_trade_freq=0.03)
        strategy.fit(data["train"])
        
        # Generate signals for the test set
        logger.info("Generating signals...")
        signals = strategy.generate_signals(data["test"])
        
        # Run backtest
        logger.info("Running backtest...")
        backtest = Backtest(
            data=data["test"], 
            signals=signals,
            initial_capital=10000,
            fee_rate=0.0006,
            risk_free_rate=0.02  # 2% annual risk-free rate
        )
        
        results = backtest.run_backtest()
        metrics = backtest.evaluate_performance()
        
        # Print results
        print("\n===== BACKTEST RESULTS =====")
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Trade Frequency: {metrics['trade_frequency']:.2%}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Annualization Factor: {metrics['annualization_factor']}")
        
        print("\n===== SUCCESS CRITERIA =====")
        print(f"Sharpe Ratio ≥ 1.8: {'✅' if metrics['meets_sharpe_criteria'] else '❌'}")
        print(f"Max Drawdown ≥ -40%: {'✅' if metrics['meets_drawdown_criteria'] else '❌'}")
        print(f"Trade Frequency ≥ 3%: {'✅' if metrics['meets_trade_freq_criteria'] else '❌'}")
        print(f"All Criteria Met: {'✅' if metrics['meets_all_criteria'] else '❌'}")
        
        # Plot results if matplotlib is available
        if MATPLOTLIB_AVAILABLE:
            backtest.plot_results()
    
    except Exception as e:
        logger.error(f"Error in backtest execution: {str(e)}") 