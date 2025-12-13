import pandas as pd
import numpy as np
from data_ingestion import get_data


def calculate_volatility(df, window=20) :
    """Calculate realized volatility from price data."""
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    
    # Realized volatility (annualized rolling std)
    # There are roughly 252 trading days in a year.
    df['volatility'] = df['returns'].rolling(window).std() * np.sqrt(252)

    # Remove NaN rows
    df = df.dropna()
    print(df.isna().sum().sort_values(ascending=False))
    return df


