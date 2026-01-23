from pathlib import Path

# Get project root (2 levels up from this file)
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DATA_RAW = BASE_DIR / "data"/"raw"
DATA_PROCESSED = BASE_DIR / "data"/"processed"

# Other directories
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_RAW, DATA_PROCESSED, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)



# Model parameters
TICKER = "SPY"  # The asset for modelling (S&P 500 ETF)
START_DATE = "1993-01-01" # Data start date
END_DATE = "2025-12-11" # Data end date 
HISTORICAL_END = "2025-12-11"  # Static historical cutoff (git committed)
RECENT_DAYS = 60               # Live data window (fast download)
VOL_WINDOW = 20   # Looks at the past 20 days to compute features (Rolling window), 20 is the approximate number of trading days in a month
TEST_SIZE = 0.2 # Proportion of data for testing
RANDOM_STATE = 42 # Seed for random number generator to ensure reproducibility
VOL_TARGET_HORIZON = 20 #To predict volatility over the next 20 days 