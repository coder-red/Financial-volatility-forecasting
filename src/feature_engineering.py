import pandas as pd
import numpy as np
from data_ingestion import get_data
from config import VOL_WINDOW, HORIZON


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
    """This lets each row use past values (previous days’ returns) as information/
     to help predict what might happen next"""
    for lag in [1, 5, 10, 20]: 
        df[f"lag_{lag}"] = df["log_return"].shift(lag) #every value slides down by the amount of {lag} row and the top is filled with NaN

    # Smoothed volatility proxy
    """gives a smoother measure of recent volatility. 
    instead of looking at one noisy day’s move, it looks at the average size of moves over the last 20 days
    This helps the model see how turbulent the market has been recently rather than reacting to a single spike."""

    df[f"rolling_abs_return_mean_{VOL_WINDOW}d"] = (
        df["abs_return"].rolling(VOL_WINDOW).mean()
    )

    return df



def add_target_volatility(df, HORIZON):
    """
    Create future realized volatility as prediction target.

    Target = volatility over the NEXT `HORIZON` trading days.
    """
    df = df.copy()

    df["target_volatility"] = (
        df["log_return"]
        .rolling(HORIZON)
        .std()
        .shift(-HORIZON) # .shift(-horizon) takes the value that was originally aligned with the end of a window and slides it up so it sits at the start of that window, turning it into a “future” label for that row
        * np.sqrt(252)
    )

    return df

