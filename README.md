📊 WealthWise – Intelligent Financial Planning System
🎯 1. Project Title & Objective

WealthWise is an AI-driven personal finance decision-support system designed to help users make informed financial decisions through:

Asset Growth Prediction

Investment Allocation Recommendation

Retirement Readiness Scoring

Expenses & Savings Analysis

Objective

To build an integrated system that uses Machine Learning + Financial Analytics to simplify financial planning for individuals by providing data-driven predictions, risk-aligned investment recommendations, and long-term wealth forecasts.

📁 2. Dataset Details

WealthWise uses a combination of:

A. Market & Asset Datasets

Raw price data collected from Yahoo Finance for 7 key tickers:

AAPL

MSFT

^GSPC (S&P 500 Index)

GC=F (Gold Futures)

BTC-USD (Bitcoin)

ETH-USD (Ethereum)

^TNX (US Treasury Yield)

After merging → stored as:
✔ raw_all_assets.csv
✔ Preprocessed into: processed_all_assets.csv

B. User Financial Data (SQLite Database)

Stored user-specific inputs:

Income, Expenses, Savings

Assets (Stock/Gold/Crypto/Mutual Funds etc.)

Expected Rate of Return

Age, Goals & Time Horizon

C. Generated Technical Indicators

Preprocessing also included computation of:

Moving Averages (MA5, MA20, MA50)

RSI-14

MACD line, signal & histogram

Normalized volatility

Date-time encodings (Month, Day, Year, Day of Week)

🤖 3. Algorithm / Models Used
A. Asset Growth Prediction Model

Random Forest Regressor

XGBoost Regressor

Hyperparameter tuning using RandomizedSearchCV

80/20 Train–Test Split

Final model selected based on highest R² performance.

B. Investment Allocation Model

Random Forest Regression

Learns optimal allocation for:

Equity

Debt

Gold

Real Estate

Based on user's:

Income

Age

Savings

Debt

Time horizon

Risk level

C. Retirement Readiness Score

Multiple Linear Regression

Predicts readiness score (0–100)

D. Expenses & Savings Prediction

Simple regression & rule-based analytics

Stores all calculations in SQLite database

📊 4. Results (Accuracy & Graphs)
A. Asset Prediction Model Performance
Metric	Random Forest	XGBoost
R² Score	~0.85–0.90	~0.88–0.92
MAE	Low (good)	Lower (best)
RMSE	Moderate	Lower

✔ XGBoost usually selected as best-performing model
✔ Model captures trends well due to technical indicators

B. Visualization Outputs

The system generates:

Multi-year asset growth graph (1–50 years)

Portfolio total value chart

Investment allocation pie chart

Retirement readiness indicator gauge

Examples include:

Line charts using Chart.js

User-specific forecasts for each asset

Year-by-year table of values

📌 5. Conclusion

WealthWise successfully integrates machine learning, financial indicators, and user-centered analytics into a unified system capable of:

Predicting future asset values with strong accuracy

Recommending optimal investment distribution

Assessing retirement preparation

Helping users plan expenses & savings effectively

The platform demonstrates that ML-enhanced financial planning can significantly improve decision-making for beginner and intermediate investors.

🚀 6. Future Scope

We aim to extend WealthWise with:

Live API integration (NSE/BSE, Crypto APIs, AlphaVantage)

SIP, SWP, and STP investment forecasting

Goal-based financial planning (Education, Home, Marriage)

Advanced ML (LSTM, Prophet, Transformer models)

Risk simulation using Monte-Carlo Analysis

Mobile App version (Flutter/React Native)

📚 7. References

Yahoo Finance API – Historical Market Data

Scikit-Learn Documentation

XGBoost: A Scalable Tree Boosting System

Technical Indicators – Investopedia

Python Flask Framework Documentation

Chart.js Visualization Library

SQLite Documentation
