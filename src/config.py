from pathlib import Path

# Get project root (2 levels up from this file)
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DATA_RAW = BASE_DIR / "data"/"raw"
DATA_SAMPLE = BASE_DIR / "data"/"data_sample"
DATA_PROCESSED = BASE_DIR / "data"/"processed"

# Other directories
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_RAW, DATA_SAMPLE, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)



# Model parameters
TICKER = "SPY"  # The asset for modelling (S&P 500 ETF)
START_DATE = "1993-01-01" # Data start date
END_DATE = "2025-12-11" # Data end date 
VOL_WINDOW = 20   # Rolling window (in days) for volatility features, 20 is the approximate number of trading days in a month.
TEST_SIZE = 0.2 # Proportion of data for testing
RANDOM_STATE = 42 # Seed for random number generator to ensure reproducibility