import pandas as pd
import numpy as np
from xgboost import XGBRegressor 
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import DATA_PROCESSED, TEST_SIZE
from pathlib import Path


def load_features():
    """Load engineered features"""
    df = pd.read_csv(DATA_PROCESSED / "processed.csv", index_col=0, parse_dates=True)
    return df


def train_test_split(df, target_col='target_volatility', test_size=TEST_SIZE):
    """Split data chronologically for time series"""
    
    # Define features (exclude target, {log_return} (already embedded in volatility target and including it risks leakage), OHLCV(beacause prices are non-stationary and unsuitable for volatility prediction))
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

######################## XGBOOST  #################################

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


def walk_forward_xgboost(X_train, X_test, y_train, y_test, retrain_frequency=20):
    """
    Walk forward validation retrains the model every 20 days on growing historical data 
    while predicting one day ahead through the test period
    """
    predictions = []
    
    print(f"Starting walk-forward validation...")
    print(f"Initial train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Retraining every {retrain_frequency} days")
    
    for i in range(len(X_test)): # Loops through each day (i is the day to be predicted)
        
        if i % retrain_frequency == 0:  #checks if the current day number is a multiple of 20 retraining only on days 0, 20, 40, 60 etc
            print(f"Day {i}/{len(X_test)}: Retraining model (train size: {len(X_train) + i})...") #shows progress e.g on day 20 train size grows to 500+20=520 days
            
            # Expanding window: use all data and stop before day i
            X_train_current = pd.concat([X_train, X_test.iloc[:i]])
            y_train_current = pd.concat([y_train, y_test.iloc[:i]])
            
            model = train_xgboost(X_train_current, y_train_current)
        
        # Predict next day
        X_current = X_test.iloc[[i]]
        pred = model.predict(X_current)[0]
        predictions.append(pred)
    
    print(f" Walk forward complete: {len(predictions)} predictions")
    return np.array(predictions)


################################## GARCH ##############################

def train_garch(returns_series, p=1, q=1):
    """Train GARCH(p,q) model on returns,
    GARCH predicts volatility based on the past returns"""
    
    # GARCH models were mathematically derived assuming percentage returns so ..
    # convert returns from  decimal to percentage for GARCH
    returns_pct = returns_series * 100
    
    # Fit GARCH model
    model = arch_model(returns_pct, vol='Garch', p=p, q=1)
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



############################ EGARCH #################################

def train_egarch(returns, exog):
    """
    Train EGARCH: Exponential GARCH with asymmetric leverage effects.
    Captures asymmetry: bad news increases volatility more than good news.
    Sentiment enters mean equation to model news impact on returns.
    
    returns: pd.Series of log returns
    exog: np.ndarray (T x k) - sentiment features (scaled)
    """
    model = arch_model(
        returns * 100,  # Convert to percentage
        mean="ARX",  # Mean equation with exogenous sentiment
        lags=0,
        x=exog,
        vol="EGARCH",  # Exponential GARCH (asymmetric volatility response)
        p=1,  # Lag order for conditional variance
        o=1,  # Asymmetry parameter (leverage effect)
        q=1,  # Lag order for squared innovations
        rescale=False
    )
    res = model.fit(disp="off")
    return res


def forecast_egarch(model_res, exog_next):
    """
    1-step ahead volatility forecast from EGARCH with sentiment.
    exog_next: np.ndarray (1 x k) - next period's sentiment value
    """
    forecast = model_res.forecast(horizon=1, x=exog_next)
    var = forecast.variance.values[-1,0]
    # Return annualized volatility
    return np.sqrt(var) / 100 * np.sqrt(252)




def evaluate_models(y_true, y_pred, model_name="Model"):
    """Evaluate model performance."""
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n {model_name} Performance:")
    print(f" RMSE: {rmse:.4f}")
    print(f" MAE: {mae:.4f}")
    print(f" R-squared: {r2:.4f}")


    return {"model": model_name, "rmse": rmse, "mae": mae, "R_squared": r2}