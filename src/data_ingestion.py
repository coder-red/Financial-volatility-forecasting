import yfinance as yf
import pandas as pd
from config import DATA_RAW, TICKER, START_DATE, HISTORICAL_END, RECENT_DAYS
from sentiment import fetch_daily_sentiment

def download_historical_data(ticker: str = TICKER, start: str = START_DATE, end: str = HISTORICAL_END):
    """Download SPY historical data (static, git-committed)"""
    df = yf.download(ticker, start=start, end=end)
    
    # flatten multi index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # force numeric dtype
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")

    df.to_csv(DATA_RAW / f"{ticker}_historical.csv")
    print("Historical data saved")
    print(f"Shape: {df.shape}")
    return df

def download_recent_data(ticker: str = TICKER, days: int = RECENT_DAYS):
    """Download recent SPY data (live, .gitignore)"""
    df = yf.download(ticker, period=f"{days}d", progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")
    
    recent_file = DATA_RAW / f"{ticker}_recent.csv"
    df.to_csv(recent_file)
    return df

def get_data(live_update: bool = True):
    """Load SPY data: historical (static) + recent (live optional)"""
    
    hist_file = DATA_RAW / f"{TICKER}_historical.csv"
    
    if hist_file.exists():
        print(f"Loading historical from {hist_file}")
        df_hist = pd.read_csv(hist_file, index_col=0, parse_dates=True)
    else:
        print("Historical missing. Downloading...")
        df_hist = download_historical_data()
    
    if live_update:
        df_recent = download_recent_data()
        # Smart merge (no duplicates)
        df = pd.concat([df_hist, df_recent]).drop_duplicates().sort_index()
        print(f"✅ Full data: {df.shape[0]} days (historical + {RECENT_DAYS}d live)")
    else:
        df = df_hist
        print(f"📁 Offline mode: {df.shape[0]} days historical")
    
    return df

def get_sentiment_data():
    """Fetch daily news sentiment aggregated by trading day"""
    sentiment_df = fetch_daily_sentiment(ticker="SPY", keyword="market")
    
    # Save live sentiment
    sentiment_df.to_csv(DATA_RAW / "news_sentiment.csv", index=False)
    print(f"Sentiment saved: {len(sentiment_df)} days")
    return sentiment_df
