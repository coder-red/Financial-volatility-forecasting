import pandas as pd
import numpy as np
import yfinance as yf


def get_data(ticker: str = "SPY", start: str = "1993-01-01", end: str = "2025-12-11"):
	df = yf.download(ticker, start=start, end=end)
	df.to_csv(f"../data/raw/{ticker}.csv")


get_data()