import pandas as pd
import numpy as np
from data_ingestion import get_data
from config import VOL_WINDOW, VOL_TARGET_HORIZON, DATA_PROCESSED
from sklearn.preprocessing import StandardScaler


def add_log_returns(df, price_col="Close"):
    "Compute log returns from price data(preferrable a it is closer to a normal distribution ) "
    df = df.copy()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def add_volatility_features(df, VOL_WINDOW):
    "Create volatility-related features based on log returns"
    df = df.copy()

    ## Rolling historical volatility (annualized)
    """There are roughly 252 trading days in a year,
    rolling window here is just take the last 20 days, compute then slide one step and repeat"""
    df[f"volatility_{VOL_WINDOW}d"] = (
        df["log_return"].rolling(VOL_WINDOW).std() * np.sqrt(252)
    )

    # ARCH style features
    """absolute returns can sometimes predict future volatility better, 
    while squared returns are closer to the variance based theory.
    abs_return and return_squared measure the size of past moves, 
    ignoring direction, exactly what ARCH uses to explain current volatility"""

    df["abs_return"] = df["log_return"].abs() # make everything positive, e.g abs(-5) = 5
    df["return_squared"] = df["log_return"] ** 2 #  blows up return to make it +ve

    # Lagged returns
    """This lets each row use past values (previous days' returns) as information/
     to help predict what might happen next"""
    for lag in [1, 5, 10, 20]: 
        df[f"lag_{lag}"] = df["log_return"].shift(lag) #every value slides down by the amount of {lag} row and the top is filled with NaN

    # Smoothed volatility proxy
    """gives a smoother measure of recent volatility. 
    instead of looking at one noisy day's move, it looks at the average size of moves over the last 20 days
    This helps the model see how turbulent the market has been recently rather than reacting to a single spike."""

    df[f"rolling_abs_return_mean_{VOL_WINDOW}d"] = (
        df["abs_return"].rolling(VOL_WINDOW).mean()
    )

    return df


def add_target_volatility(df, VOL_TARGET_HORIZON):
    " Target = future realized volatility over the next horizon(20) trading days"
    df = df.copy()

    df["target_volatility"] = (
        df["log_return"]
        .rolling(VOL_TARGET_HORIZON)
        .std()
        .shift(-VOL_TARGET_HORIZON) # .shift(-horizon) takes the value that was originally aligned with the end of a window and slides it up so it sits at the start of that window, turning it into a "future" label for that row
        * np.sqrt(252)
    )

    return df


def add_sentiment_features(df, sentiment_df):
    """
    Merges combined sentiment (FinBERT + Google Trends) with price data.
    Creates lagged and scaled sentiment features for EGARCH modeling.
    EGARCH models asymmetric volatility response to good/bad news.
    
    For dates without sentiment (pre-2024), fills with 0 (neutral sentiment).
    This allows EGARCH to train on full SPY history while using sentiment when available.
    """
    df = df.copy()
    df['date'] = df.index
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    
    # Aggregate multiple sentiment scores per day (in case of duplicates)
    daily_sentiment = sentiment_df.groupby('date')['sentiment'].mean().reset_index()
    
    # Merge sentiment into price data (left join keeps all SPY dates)
    df = df.merge(daily_sentiment, on='date', how='left')
    
    # Fill missing sentiment with 0 (neutral) instead of forward fill
    # This treats pre-sentiment-era data as having neutral news sentiment
    df['sentiment'] = df['sentiment'].fillna(0)
    
    # Lag sentiment by 1 day: yesterday's news affects today's volatility
    df['sentiment_lag1'] = df['sentiment'].shift(1)
    
    # Scale sentiment for EGARCH optimizer stability
    # Fit scaler only on non-zero sentiment to avoid skewing by zeros
    scaler = StandardScaler()
    
    # Mask for dates with actual sentiment data (non-zero after fillna)
    has_sentiment = sentiment_df['date'].isin(df['date'])
    
    if has_sentiment.any():
        # Fit scaler on actual sentiment values only
        actual_sentiment = df.loc[df['date'].isin(sentiment_df['date']), 'sentiment_lag1'].dropna()
        if len(actual_sentiment) > 0:
            scaler.fit(actual_sentiment.values.reshape(-1, 1))
            # Transform all values (zeros will become negative after standardization, which is fine)
            df['sentiment_scaled'] = scaler.transform(df['sentiment_lag1'].values.reshape(-1, 1))
        else:
            # Fallback: just use unscaled if no data
            df['sentiment_scaled'] = df['sentiment_lag1']
    else:
        # No sentiment data at all - use zeros
        df['sentiment_scaled'] = 0
    
    return df.set_index('date')


def engineer_features(df, sentiment_df, VOL_WINDOW, VOL_TARGET_HORIZON):
    "full pipeline"

    df = add_log_returns(df)
    df = add_volatility_features(df, VOL_WINDOW)
    df = add_sentiment_features(df, sentiment_df)
    df = add_target_volatility(df, VOL_TARGET_HORIZON)

    # only drop rows where FEATURES are missing.
    # We KEEP rows where 'target_volatility' is NaN (these are our "Live Prediction" rows).
    
    # Identify feature columns (everything except the target)
    feature_cols = [c for c in df.columns if c != 'target_volatility']
    
    # Drop rows where any FEATURE is NaN (usually the very beginning of the data)
    df = df.dropna(subset=feature_cols)

    # Save to CSV
    df.to_csv(DATA_PROCESSED/'processed.csv')
    print(f"✅ Processed data saved. Range: {df.index.min().date()} to {df.index.max().date()}")

    return df