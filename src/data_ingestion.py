import yfinance as yf
import pandas as pd
from config import DATA_RAW, TICKER, START_DATE, END_DATE

def download_data(ticker: str = TICKER, start: str = START_DATE, end: str = END_DATE):
    """Download SPY data from yfinance"""
    df = yf.download(ticker, start=start, end=end)
    
    # flatten multi index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # force numeric dtype
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")

    df.to_csv(f"{DATA_RAW}/{ticker}.csv")

    print("data downloaded")
    print(f"Data shape: {df.shape}")
    
    return df



def get_data():
    """Load SPY data (download if not exists)."""
    
    filepath = DATA_RAW / f"{TICKER}.csv"
    
    if filepath.exists():
        print(f"Loading existing data from {filepath}")
        df = pd.read_csv(filepath,
                        index_col=0,
                        parse_dates=True,
                        )
    else:                   
        print("Data not found. Downloading...")
        df = download_data()
    
    return df

