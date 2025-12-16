import pandas as pd
import numpy as np
from xgboost import XGBRegressor 
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import DATA_PROCESSED, TEST_SIZE



def load_features():
    """Load engineered features"""
    df = pd.read_csv(DATA_PROCESSED / "processed.csv", index_col=0, parse_dates=True)
    return df




def train_test_split(df, target_col='target_volatility', test_size=TEST_SIZE):
    """Split data chronologically for time series"""
    
    # Define features (exclude target, # exclude log_return (already embedded in volatility target and including it risks leakage), OHLCV(beacause prices are non-stationary and unsuitable for volatility prediction))
    exclude_cols = [target_col, 'log_return', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Time series split (chronological)
    split_idx = int(len(df) * (1 - test_size)) # split_idx = 80% of df in int
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:] #X.iloc[:split_idx] = rows 0–80%(in int) of X becomes X_train, the rest X_test
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test






def train_xgboost(X_train, y_train):
    """Train XGBoost model
    XGBoost predicts volatility based on features and complex patterns"""
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model






def train_garch(returns_series, p=1, q=1):
    """Train GARCH(p,q) model on returns,
    GARCH predicts volatility based on the past returns"""
    
    # GARCH models were mathematically derived assuming percentage returns so ..
    # convert returns from  decimal to percentage for GARCH
    returns_pct = returns_series * 100
    
    # Fit GARCH model
    model = arch_model(returns_pct, vol='Garch', p=p, q=q)
    # p = persistence / averaging of past volatility over time
    # q = reaction to the magnitude of recent shocks (returns)
    fitted_model = model.fit(disp='off') # disp ='off' = Suppresses optimizer spam and cleaner logs
    
    return fitted_model

def forecast_garch(fitted_model, horizon=1):
    """forecast volatility using fitted GARCH model"""
    
    forecast = fitted_model.forecast(horizon=horizon)
    volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
    
    # Convert back to decimal and annualize
    volatility_forecast = volatility_forecast / 100 * np.sqrt(252)
    
    return volatility_forecast[0]





def evaluate_models(y_true, y_pred, model_name="Model"):
    """Evaluate model performance."""
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}
