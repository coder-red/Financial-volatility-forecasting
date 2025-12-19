![Banner](assets/vol.png)

![Python version](https://img.shields.io/badge/Python%20version-3.10%2B-lightgrey)
![GitHub repo size](https://img.shields.io/github/repo-size/coder-red/Financial-volatility-forecasting)
![GitHub last commit](https://img.shields.io/github/last-commit/coder-red/Financial-volatility-forecasting)


# Key findings: 

**XGBoost outperformed GARCH(1,1) across all metrics:**

| Model     | RMSE  | MAE    | R²     |
|-----------|-------|--------|--------|
| GARCH(1,1)| 0.1078| 0.0622 | 0.1125 |
| XGBoost   | 0.1001| 0.0562 | 0.2347 |


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
This model predicts future market volatility.Portfolio managers, traders and financial institutions use this to manage risk, price options, and decide how much exposure to take in financial markets. 

## Data source

- Daily SPY price data was downloaded from Yahoo Finance using the yfinance Python library. It spans from January 1st, 1993 to December 11, 2025 covering approximately 8,300 trading days.


## Methods

- Data cleaning,preprocessing and Feature engineering to create predictive variables
- Exploratory data analysis
- Model training and evaluation with XGBoost and GARCH

## Tech Stack

- Python (refer to requirement.txt for the packages used in this project)
- Scikit-learn, XGBoost (machine learning & evaluation)
- GARCH 


## Quick glance at the results

Target distribution between the features.

![Bar chart](assets/target_dist.png)

Summary bar of major features

![Bar chart](assets/shap_summary_bar.png)

Confusion matrix of LightGBM.

![Confusion matrix](assets/confusion_matrix_lgbm_tuned.png)

ROC curve of LightGBM.

![ROC curve](assets/roc_curve.png)

models (with default parameters)

| Model     	         |    AUC-ROC score     |
|----------------------|----------------------|
| LightGBM(tuned)      | 72.57% 	            |
| XGboost  (tuned)     | 72.42% 	            |
| Logistic Regression  | 69.80% 	            |


- ***Metrics used: rmse, mae, R²***


### Model Evaluation Strategy

**Primary Metric: ROC-AUC**
Credit risk data is very imbalanced, so ROC-AUC is best here as it measures how well the model does in separating defaulters from non defaulters

**Supporting Metrics: Precision, Recall, F1**
- **Recall** is critical as missing a high-risk borrower leads to real financial loss.
- **Precision** helps ensure we don’t wrongly reject too many good borrowers
- **F1** balances both precision and recall.


## Lessons Learned and Recommendations

**What I found:**
- Based on the analysis in this project it was found that loan amount, existing debt ratio, and age were the strongest predictors of default
- Hyperparameter tuning barely helped improve the model performance, for example XGBoost went from 0.722349 to 0.724171 AUC and it took over 30 minutes to train. This suggests that features matter more than tuning
- For imbalanced data, AUC-ROC matters way more than accuracy, and the 0.5 threshold doesn't work (except for logistic regression), the optimal threshold was 0.121

**Recommendations:**
- Recommendation would be to focus more on the loan amount when deciding since they carry the most risk and also accept that precision will be low, you'll reject some good customers to catch defaults

## Limitations and What Can Be Improved
- Low precision means 80% of rejected applicants are false positives (lost customers)
- Hyperparameter tuning with Optuna takes 1+ hours
- Get more data 
- Monitor model performance over time and retrain quarterly

## Repository structure

<details>
  <summary><strong>Repository Structure (click to expand)</strong></summary>

```text














├── data
│   ├── processed
│   └── raw
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_garch.ipynb
│   └── 03_ml_vol.ipynb
├── results
│   ├── metrics
│   └── plots
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   └── model.py
├── .gitignore
├── README.md
└── requirements.txt