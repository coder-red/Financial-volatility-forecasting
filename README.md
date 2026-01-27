<p align="center">
  <img src="assets/vol.png" alt="Project Banner" width="100%">
</p>

![Python version](https://img.shields.io/badge/Python%20version-3.10%2B-lightgrey)
![GitHub repo size](https://img.shields.io/github/repo-size/coder-red/Financial-volatility-forecasting)
![GitHub last commit](https://img.shields.io/github/last-commit/coder-red/Financial-volatility-forecasting)


# Key findings: 

**EGARCH-X outperformed GARCH(1,1) and XGBoost in all metrics except MAE:**

                                        | Model     | RMSE  | MAE    | R²     |
                                        |-----------|-------|--------|--------|
                                        | EGARCH-X  | 0.0978| 0.0572 | 0.2744 |
                                        | XGBoost   | 0.1001| 0.0562 | 0.2347 |
                                        | GARCH(1,1)| 0.1080| 0.0620 | 0.1158 |

Even modest improvements in volatility forecasting can reduce risk mispricing and improve capital allocation.

## Author

- [@coder-red](https://www.github.com/coder-red)

## Table of Contents

  - [Business context](#business-context)
  - [Data source](#data-source)
  - [Methods](#methods)
  - [Tech Stack](#tech-stack)
  - [Quick glance at the results](#quick-glance-at-the-results)
  - [Lessons learned and recommendation](#lessons-learned-and-recommendation)
  - [Limitation and what can be improved](#limitation-and-what-can-be-improved)
  - [Repository structure](#repository-structure)


## Business context
This model predicts future market volatility. The model benchmarks GARCH against EGARCH-X and XGBoost. By incorporating exogenous sentiment data into EGARCH, the model is designed to catch "panic" moves that price data alone misses. Portfolio managers, traders and financial institutions use this to manage risk, price options, and decide how much exposure to take in financial markets. 

## Data source

- Daily SPY price data is downloaded from Yahoo Finance using the yfinance Python library. It spans from January 1st, 1993 to current date.

- RSS data for exogenous sentiment is downloaded from Yahoo Finance, Bloomberg, cnbc, ft, and wall street journal. 

- Google Trends for current market trends

## Methods

- **Sentiment Engineering (NLP):** Extracted contextual sentiment from financial headlines using FinBERT. The model was optimized via ONNX to reduce inference latency, allowing for rapid processing of large-scale historical RSS archives.
- **Feature engineering and sentiment extraction:** Integrated RSS feeds and Google Trends as exogenous variables, applied NLP based sentiment scoring to quantify market "panic"
- **Benchmarking: Evaluated three distinct models:** GARCH(1,1), EGARCH-X, and XGBoost.
- **Chronological Train/Test Split:** Used a fixed  split i.e 80% train / 20% test to preserve the time dependent structure of the market data and prevent random shuffling.
- **Look-ahead Bias Prevention:** All exogenous features (Sentiments and trends) were lagged to ensure predictions rely strictly on information available at the time of the forecast.


## Tech Stack

- **Python:** Core logic (refer to requirement.txt for the packages used in this project)
- **FinBERT + ONNX Runtime:** Leveraged FinBERT (specialized BERT for finance) exported to ONNX format for high-speed sentiment inference on RSS and news data.
- **Scikit-learn and XGBoost:** machine learning & evaluation
- **Arch Library:** Used for GARCH(1,1) as the baseline and EGARCH-X for modelling with exogenous inputs
- **NLP & APIs:** yfinance for market data; Bloomberg, cnbc, ft, and wall street journal, Google Trends for exogenous inputs



## Quick glance at the results

Line chart comparing forecasts of XGBoost vs GARCH vs EGARCH-X

![Line chart](assets/master_vol_comparison.png)

Bar Chart of error comparison of XGBoost vs GARCH

![Bar Chart](assets/error_comparison.png)

Feature importance.

![Bar Chart](assets/Feature_importance.png)

- ***Metrics used: rmse, mae, R²***


### Model Evaluation Strategy

**Primary Metric: RMSE (Root Mean Squared Error)**
Volatility forecasting requires precise predictions since small errors can compound in risk calculations.RMSE penalizes large forecast errors more heavily than MAE, making it best for identifying models that avoid dangerous outliers in volatility estimates.


**Supporting Metrics: MAE (Mean Absolute Error), R²**
- **MAE** shows the model's average size of forecasting error.
- **R²** indicates how much variation in volatility the model explains.


## Lessons Learned and Recommendations

**What I found:**

- **EGARCH-X vs. XGBoost Performance:** While XGBoost is better at capturing non-linearities, EGARCH-X performed better due to its assymetry modelling combined with reacting to exogenous sentiment

- **Walk-forward validation with XGBoost performed better:** I compared with standard xgboost and walk-forward validation performed better. This might be because retraining could have added more signal. SPY volatility dynamics were stable during the test period

- **Historical volatility dominates prediction:** The 20-day rolling mean of absolute returns (`rolling_abs_return_mean_20d`) was by far the strongest predictor. This confirms volatility persistence.  This is because instead of looking at one noisy day’s move, it looks at the average size of moves over the last 20 days. This helps the model see how turbulent the market has been recently rather than reacting to a single spike.

- **ARCH-style features (abs_return, return_squared) underperformed expectations:** `return_squared` had zero importance in XGBoost. This is likely due to the presence of lagged volatility feature which makes it add little incremental information. The model already captures volatility dynamics through historical rolling volatility.

- **Lagged returns showed limited value:** Lagged returns added very limited incremental value because the rolling volatility feature already captures past returns. Since `rolling_abs_return_mean_20d` is calculated from the last 20 days of returns, individual lagged returns become redundant.



**Recommendation:**
- Recommendation would be to regularly re train the model on new data and use a simple check to see if the market is in a calm or crazy period, then use settings that fit that period.


## Limitation and What Can Be Improved
**Limitation**
- The model is mostly looking at what happened yesterday to predict today. If there is a major sudden market crash or spike, the model may be one day late to react because it hasn't seen the news/pattern yet.


**What Can Be Improved**
- Dynamic Re-training: Implement an automated pipeline to regularly re-train the model on a sliding window.

## Repository structure

<details>
  <summary><strong>Repository Structure (click to expand)</strong></summary>

```text

Financial-volatility-forecasting/
├── assets/                         # Images used in the README
│   ├── master_vol_comparison.png
│   ├── error_comparison.png
│   ├── Feature_importance.png
│   └── vol.png
├── data/                            # All data (raw, processed)
│   ├── processed/
│   │   └── processed.csv
│   └── raw/
│       ├── combined_sentiment.csv
│       ├── daily_vol.csv
│       ├── google_trends.csv
│       ├── news_sentiment.csv
│       └── SPY.csv     
├── notebooks/                         # Jupyter notebooks for analysis + modelling + interpretation
│    ├── 01_eda.ipynb
│    ├── 02_garch.ipynb
│    ├── 03_xgboost.ipynb
│    ├── 04_model_benchmarking.ipynb
│    └── 05_egarch.ipynb
│   
│
├── results/                            # Generated plots and outputs
│    ├── figures/
│    │   ├── eda/
│    │   │   ├── correlations.png
│    │   │   ├── log_returns.png
│    │   │   ├── sentiment_distribution.png
│    │   │   ├── sentiment_vol_scatter.png
│    │   │   ├── target_volatility.png
│    │   │   └── volatility_features.png
│    │   ├── egarch/
│    │   │   └── EGARCH_predicted_vs_True_Volatility.png
│    │   ├── garch/
│    │   │   ├── GARCH_Forecast_vs_Target_Volatility.png
│    │   │   └── GARCH_predicted_vs_True_Volatility.png
│    │   └── xgboost/
│    │       ├── Feature_importance.png
│    │       └── Predicted_vs_True_Volatility.png
│    ├── metrics/
│    │   ├── egarch_metrics.csv
│    │   ├── garch_metrics.csv
│    │   ├── model_comparison.csv
│    │   └── xgboost_metrics.csv
│    └── preds/
│        ├── egarch_preds.csv
│        ├── garch_preds.csv
│        ├── master_vol_comparison.png
│        ├── model_comparison.html
│        └── xgboost_preds.csv
│
│
├── src/                                     # Python modules
│   ├── __init__.py
│   ├── config.py                            # Paths and constants
│   ├── data_ingestion.py                    # Data Ingestion
│   ├── feature_engineering.py               # Feature engineering functions
│   ├── model.py                             # Training + evaluation
│   └── sentiment.py                         
├── .gitignore                               # Files/folders ignored by git
├── README.md                                # Project overview
├── requirements.txt                         
└── setup.py                               