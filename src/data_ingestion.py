import yfinance as yf
import pandas as pd
from config import DATA_RAW, TICKER, START_DATE, END_DATE

def download_data(ticker: str = TICKER, start: str = START_DATE, end: str = END_DATE):
	"download SPY data from yfinance"
	df = yf.download(ticker, start=start, end=end)
	df.to_csv(f"{DATA_RAW}/{ticker}.csv")

	print("data downloaded")
	print(f"Data shape: {df.shape}")
	
	return df



def get_data():
    """Load SPY data (download if not exists)."""
    
    filepath = DATA_RAW / f"{TICKER.lower()}_prices.csv"
    
    if filepath.exists():
        print(f"Loading existing data from {filepath}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        print("Data not found. Downloading...")
        df = download_data()
    
    return df