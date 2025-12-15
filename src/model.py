import pandas as pd
import numpy as np
from config import DATA_PROCESSED, TEST_SIZE

def load_features():
    """Load engineered features."""
    df = pd.read_csv(DATA_PROCESSED / "processed.csv", index_col=0, parse_dates=True)
    return df


def train_test_split(df, target_col='target_volatility', test_size=TEST_SIZE):
    "Split data chronologically for time series."
    
    # Define features (exclude target, returns(because they are used to build volatility, not predict it directly), OHLCV(beacause prices are non-stationary and useless for volatility prediction))
    exclude_cols = [target_col, 'log_return', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Time series split (chronological)
    split_idx = int(len(df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test
