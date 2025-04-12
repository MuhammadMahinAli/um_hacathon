import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import xgboost as xgb
import sys
import os

# Add parent directory to path to enable importing from data_process
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ML.data_process import prepare_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HybridStrategy:
    """
    A strategy that combines XGBoost and Hidden Markov Models (HMM) for trading signal generation.
    
    The strategy uses HMM to identify market regimes and includes that information
    as a feature in an XGBoost classifier to predict future price movements.
    """
    
    def __init__(self, n_hmm_components: int = 3, min_trade_freq: float = 0.03):
        """
        Initialize the HybridStrategy.
        
        Args:
            n_hmm_components (int): Number of HMM components/regimes
            min_trade_freq (float): Minimum required trade frequency (0.03 = 3%)
        """
        self.xgb_model = None
        self.hmm_model = None
        self.scaler = StandardScaler()
        self.n_hmm_components = n_hmm_components
        self.min_trade_freq = min_trade_freq
        self.hmm_features = ['log_return', 'volatility_14', 'rsi_14']
        self.regime_feature_name = 'market_regime'
        
    def _fit_hmm(self, data: pd.DataFrame) -> Tuple[GaussianHMM, np.ndarray]:
        """
        Fit an HMM model on the price features to detect market regimes.
        
        Args:
            data (pd.DataFrame): Training data with price features
            
        Returns:
            Tuple[GaussianHMM, np.ndarray]: Fitted HMM model and regime predictions
        """
        logger.info("Training HMM model for regime detection...")
        
        # Extract features for HMM
        hmm_data = data[self.hmm_features].copy()
        
        # Normalize features
        hmm_data_scaled = self.scaler.fit_transform(hmm_data)
        
        # Create and fit HMM model
        hmm = GaussianHMM(
            n_components=self.n_hmm_components,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        try:
            hmm.fit(hmm_data_scaled)
            # Predict the hidden states/regimes
            hidden_states = hmm.predict(hmm_data_scaled)
            logger.info(f"HMM model trained successfully. Found {self.n_hmm_components} regimes.")
            
            # Distribution of regimes
            regime_counts = np.bincount(hidden_states)
            regime_percentages = regime_counts / len(hidden_states)
            logger.info(f"Regime distribution: {regime_percentages}")
            
            return hmm, hidden_states
            
        except Exception as e:
            logger.error(f"Error fitting HMM model: {str(e)}")
            raise
    
    def _predict_regimes(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict market regimes using the trained HMM model.
        
        Args:
            data (pd.DataFrame): Data to predict regimes for
            
        Returns:
            np.ndarray: Predicted regimes
        """
        if self.hmm_model is None:
            raise ValueError("HMM model not trained. Call fit() first.")
        
        # Extract and scale features
        hmm_data = data[self.hmm_features].copy()
        hmm_data_scaled = self.scaler.transform(hmm_data)
        
        # Predict regimes
        return self.hmm_model.predict(hmm_data_scaled)
    
    def _prepare_features(self, data: pd.DataFrame, regimes: np.ndarray = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for the XGBoost model, including market regime if available.
        
        Args:
            data (pd.DataFrame): Data to prepare features from
            regimes (np.ndarray, optional): Market regimes if available
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target vector y
        """
        # Features to use
        base_features = [
            'return', 'log_return', 
            'lag_return_1', 'lag_return_3', 'lag_return_5',
            'sma_7', 'sma_14', 'sma_30', 
            'ema_7', 'ema_14', 'ema_30',
            'volatility_7', 'volatility_14', 'volatility_30',
            'rsi_14'
        ]
        
        # Create a copy to avoid modifying the original
        X = data[base_features].copy()
        
        # Add market regime feature if provided
        if regimes is not None:
            X[self.regime_feature_name] = regimes
        
        # Target is binary future return
        y = data['future_binary_24h']
        
        return X, y
    
    def fit(self, data: pd.DataFrame) -> 'HybridStrategy':
        """
        Train the hybrid model using XGBoost and HMM.
        
        Args:
            data (pd.DataFrame): Training data with price features and future returns
            
        Returns:
            HybridStrategy: Self for method chaining
        """
        logger.info("Training hybrid strategy model...")
        
        try:
            # 1. Fit HMM to identify market regimes
            self.hmm_model, regimes = self._fit_hmm(data)
            
            # 2. Prepare features including market regimes
            X, y = self._prepare_features(data, regimes)
            
            # 3. Split data for XGBoost training
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # 4. Train XGBoost model
            logger.info("Training XGBoost model...")
            self.xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            # Simplify the training without early stopping
            self.xgb_model.fit(X_train, y_train)
            
            val_score = self.xgb_model.score(X_val, y_val)
            logger.info(f"XGBoost model trained successfully. Validation score: {val_score:.4f}")
            
            # Feature importance
            feature_importance = self.xgb_model.feature_importances_
            features = X.columns
            importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)
            logger.info(f"Top 5 important features: {importance_df.head(5)}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error training hybrid strategy: {str(e)}")
            raise
    
    def generate_signals(self, data: pd.DataFrame) -> List[int]:
        """
        Generate trading signals based on model predictions.
        
        Args:
            data (pd.DataFrame): Data to generate signals for
            
        Returns:
            List[int]: List of signals (1: buy, 0: hold, -1: sell)
        """
        logger.info("Generating trading signals...")
        
        try:
            # 1. Predict regimes
            regimes = self._predict_regimes(data)
            
            # 2. Prepare features
            X, _ = self._prepare_features(data, regimes)
            
            # 3. Get XGBoost predictions with probabilities
            probs = self.xgb_model.predict_proba(X)
            
            # 4. Generate signals based on confidence
            # If model has high confidence in positive return, buy (1)
            # If model has high confidence in negative return, sell (-1)
            # Otherwise, hold (0)
            signals = np.zeros(len(data), dtype=int)
            
            # Probability threshold for high confidence
            positive_threshold = 0.6  # Confidence for buying
            negative_threshold = 0.6  # Confidence for selling
            
            for i in range(len(probs)):
                # probs[i][1] is probability of positive return (class 1)
                if probs[i][1] >= positive_threshold:
                    signals[i] = 1  # Buy
                elif probs[i][0] >= negative_threshold:  # probs[i][0] is probability of negative return (class 0)
                    signals[i] = -1  # Sell
                else:
                    signals[i] = 0  # Hold
            
            # 5. Ensure minimum trade frequency
            trade_ratio = np.sum(signals != 0) / len(signals)
            logger.info(f"Initial trade ratio: {trade_ratio:.2%}")
            
            if trade_ratio < self.min_trade_freq:
                logger.info(f"Trade ratio below minimum {self.min_trade_freq:.2%}. Adjusting signals...")
                
                # Lower the threshold to increase trade frequency
                while trade_ratio < self.min_trade_freq:
                    # Reduce thresholds by 5% each time
                    positive_threshold *= 0.95
                    negative_threshold *= 0.95
                    
                    # Recreate signals
                    for i in range(len(probs)):
                        if probs[i][1] >= positive_threshold:
                            signals[i] = 1  # Buy
                        elif probs[i][0] >= negative_threshold:
                            signals[i] = -1  # Sell
                        else:
                            signals[i] = 0  # Hold
                    
                    trade_ratio = np.sum(signals != 0) / len(signals)
                    
                    # Emergency break to prevent infinite loop
                    if positive_threshold < 0.5 or negative_threshold < 0.5:
                        logger.warning("Hit minimum threshold. Forcing trades...")
                        
                        # Get the indices sorted by confidence
                        pos_indices = np.argsort(-probs[:, 1])  # Highest positive conf first
                        neg_indices = np.argsort(-probs[:, 0])  # Highest negative conf first
                        
                        # Calculate how many trades we need to add
                        trades_needed = int(self.min_trade_freq * len(signals)) - np.sum(signals != 0)
                        
                        if trades_needed > 0:
                            # Add half as buys, half as sells from highest confidence predictions
                            buy_adds = trades_needed // 2
                            sell_adds = trades_needed - buy_adds
                            
                            # Add buy signals
                            pos_counter = 0
                            while buy_adds > 0 and pos_counter < len(pos_indices):
                                idx = pos_indices[pos_counter]
                                if signals[idx] == 0:  # Only change holds to trades
                                    signals[idx] = 1
                                    buy_adds -= 1
                                pos_counter += 1
                            
                            # Add sell signals
                            neg_counter = 0
                            while sell_adds > 0 and neg_counter < len(neg_indices):
                                idx = neg_indices[neg_counter]
                                if signals[idx] == 0:  # Only change holds to trades
                                    signals[idx] = -1
                                    sell_adds -= 1
                                neg_counter += 1
                                
                        break
            
            # Final statistics
            final_trade_ratio = np.sum(signals != 0) / len(signals)
            buy_ratio = np.sum(signals == 1) / len(signals)
            sell_ratio = np.sum(signals == -1) / len(signals)
            
            logger.info(f"Final trade frequency: {final_trade_ratio:.2%}")
            logger.info(f"Buy signals: {buy_ratio:.2%}, Sell signals: {sell_ratio:.2%}")
            
            return signals.tolist()
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def calculate_performance(self, data: pd.DataFrame, signals: List[int]) -> Dict[str, float]:
        """
        Calculate the strategy performance.
        
        Args:
            data (pd.DataFrame): Price data
            signals (List[int]): Generated signals
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Convert signals to numpy array
        signals_array = np.array(signals)
        
        # Convert daily returns to numpy array
        returns = data['return'].values
        
        # Calculate strategy returns (signal from previous day * today's return)
        strategy_returns = np.roll(signals_array, 1) * returns
        strategy_returns[0] = 0  # First day has no previous signal
        
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + strategy_returns) - 1
        
        # Calculate metrics
        total_return = cum_returns[-1]
        
        # Annualized return (assuming 252 trading days)
        n_days = len(returns)
        ann_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Calculate daily volatility and annualize it
        daily_vol = np.std(strategy_returns)
        ann_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        sharpe_ratio = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(1 + cum_returns)
        drawdown = (1 + cum_returns) / peak - 1
        max_drawdown = np.min(drawdown)
        
        # Trade frequency
        trade_freq = np.sum(signals_array != 0) / len(signals_array)
        
        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trade_frequency": trade_freq
        }


if __name__ == "__main__":
    # Load and prepare data
    logger.info("Loading data...")
    try:
        # Use os.path to build a more robust path
        data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cryptoquant_metrics.csv")
        logger.info(f"Attempting to load data from: {data_file}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found at: {data_file}")
            
        df = pd.read_csv(data_file)
        data = prepare_features(df)
        
        # Train the strategy on the training data
        strategy = HybridStrategy(n_hmm_components=3, min_trade_freq=0.03)
        strategy.fit(data["train"])
        
        # Generate signals on the test data
        signals = strategy.generate_signals(data["test"])
        
        # Calculate and print performance metrics
        performance = strategy.calculate_performance(data["test"], signals)
        
        print("\n===== STRATEGY PERFORMANCE =====")
        print(f"Total Return: {performance['total_return']:.2%}")
        print(f"Annualized Return: {performance['annualized_return']:.2%}")
        print(f"Annualized Volatility: {performance['annualized_volatility']:.2%}")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"Trade Frequency: {performance['trade_frequency']:.2%}")
        
        # Signal distribution
        signal_counts = np.bincount(np.array(signals) + 1, minlength=3)  # +1 to make -1,0,1 -> 0,1,2
        print("\n===== SIGNAL DISTRIBUTION =====")
        print(f"Buy signals (1): {signal_counts[2]} ({signal_counts[2]/len(signals):.2%})")
        print(f"Hold signals (0): {signal_counts[1]} ({signal_counts[1]/len(signals):.2%})")
        print(f"Sell signals (-1): {signal_counts[0]} ({signal_counts[0]/len(signals):.2%})")
        print(f"Total trades: {signal_counts[0] + signal_counts[2]} ({(signal_counts[0] + signal_counts[2])/len(signals):.2%})")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}") 