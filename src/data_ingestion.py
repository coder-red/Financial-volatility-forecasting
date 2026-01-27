import yfinance as yf
import pandas as pd
from config import DATA_RAW, TICKER, START_DATE, END_DATE, VOL_WINDOW
from sentiment import fetch_daily_sentiment_multi_source, fetch_google_trends, combine_sentiment_sources, create_daily_vol


def download_data(ticker: str = TICKER, start: str = START_DATE, end: str = END_DATE):
    """Download SPY data and ensure it's saved in a clean machine-readable format."""
    print(f"Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end)
    
    # This line flattens yfinance's  Ticker to just 'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Save as CSV 
    filepath = DATA_RAW / f"{ticker}.csv"
    df.to_csv(filepath)
    print(f"✅ Data saved to {filepath}")
    
    return df


def get_data():
    """ Load SPY data from CSV or download if not present or outdated """
    filepath = DATA_RAW / f"{TICKER}.csv"
    
    if filepath.exists():
        # parse_dates=True is CRITICAL for time-series forecasting
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Check if the data is old (e.g if the last date is not 'yesterday')
        last_date = df.index[-1].strftime("%Y-%m-%d") # Get last date in 'YYYY-MM-DD' format
        if last_date < END_DATE:
            print(f"⚠️ Data is outdated (Last date: {last_date}). Re-downloading...")
            df = download_data()
    else:
        df = download_data()
    return df


def get_sentiment_data(vol_weighted=True):
    """
    Fetch and combine all sentiment sources for EGARCH modeling.
    Combines multi-source FinBERT news sentiment + Google Trends search volume.
    Optionally applies volatility weighting.
    Returns combined sentiment DataFrame.
    """
    print("\n Fetching multi-source FinBERT news sentiment...")
    news_df = fetch_daily_sentiment_multi_source(ticker=TICKER, keyword="market")
    news_df = news_df[news_df['date'] <= pd.to_datetime(END_DATE).date()]
    news_df.to_csv(DATA_RAW / "news_sentiment.csv", index=False)
    print(f" News sentiment: {len(news_df)} days")

    print("\n Fetching Google Trends data...")
    trends_df = fetch_google_trends()
    trends_df.to_csv(DATA_RAW / "google_trends.csv", index=False)
    print(f" Google Trends: {len(trends_df)} days")

    # To ensure daily_vol.csv exists
    if vol_weighted:
        print("\n Creating daily volatility CSV...")
        create_daily_vol(ticker=TICKER, window=VOL_WINDOW)  # window=20 for rolling vol

    print("\n Combining sentiment sources...")
    combined_df = combine_sentiment_sources(vol_weighted=vol_weighted)

    return combined_df
