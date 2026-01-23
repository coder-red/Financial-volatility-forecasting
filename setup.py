from setuptools import setup, find_packages

setup(
    name="financial_volatility_forecasting",
    version="0.1.0",
    description="Financial volatility forecasting using GARCH, GARCH-X and XGBoost",
    author="Your Name",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "yfinance",
        "matplotlib",
        "scikit-learn",
        "arch",
        "xgboost",
        "transformers",
        "torch",
        "feedparser",
    ],
)
