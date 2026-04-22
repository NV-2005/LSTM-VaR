#Forecasting Value-at-Risk with Deep Quantile Regression

This project implements a Deep Learning framework to forecast **Value-at-Risk (VaR)** for the S&P 500. By combining traditional financial feature engineering (GARCH, RiskMetrics) with the non-linear capabilities of **Long Short-Term Memory (LSTM)** networks, this model directly predicts the 99% VaR threshold using **Deep Quantile Regression**.

## Project Overview

Value-at-Risk (VaR) is a standard metric in financial risk management used to estimate the potential loss of a portfolio at a given confidence level. Traditional methods often fail to capture complex non-linear dependencies or adapt quickly to changing volatility regimes.

This project benchmarks an LSTM-based Quantile Regression model against four industry-standard baselines to evaluate predictive accuracy and regulatory compliance.

## Methodology

### 1. Data and Feature Engineering
The model utilizes **30 years of S&P 500 data** augmented with the following risk factors:
* **Rolling GARCH(1,1) Volatility:** Statistical volatility measure refitted every 30 days.
* **RiskMetrics (EWMA) Volatility:** Exponentially weighted moving average (Lambda = 0.94).
* **VIX Index:** Implied volatility normalized to a daily scale.
* **RSI (Relative Strength Index):** 14-day momentum indicator.

### 2. Model Architecture and Hyperparameters
The architecture was optimized following the grid search methodology outlined in the *Journal of Financial Econometrics (Babii et al., 2022)*, favoring a regularized, shallow network to prevent overfitting:

* **Architecture:** 2-Stack LSTM
* **Units per Layer:** 10
* **Hidden Activation:** ReLU
* **Output Activation:** Linear
* **Regularization:** L2 Weight Decay (Lambda = 0.01) and 20% Dropout
* **Optimizer:** Adam (Learning Rate = 0.001)
* **Batch Size:** 20

## Backtesting and Comparative Analysis

The models were evaluated over **2,209 out-of-sample trading days** targeting a **1.0% Alpha** (99% Confidence Level).

### Global Model Comparison

| Model | Actual Breaches | Failure Rate (%) | Kupiec POF P-Value | Status |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM (Deep Quantile)** | **21** | **0.95%** | **0.8142** | **PASSED** |
| Quantile Regression | 25 | 1.13% | 0.5422 | **PASSED** |
| Polynomial Regression | 34 | 1.54% | 0.0183 | FAILED |
| Historical Simulation | 35 | 1.58% | 0.0110 | FAILED |
| Parametric (Normal) | 54 | 2.44% | 0.0000 | FAILED |

### Statistical Validation
1.  **Kupiec Proportion of Failures (POF):** The LSTM achieved a P-value of 0.8142, indicating its failure rate is statistically indistinguishable from the 1.0% target.
2.  **Regulatory Compliance:** With a 0.95% failure rate, the LSTM falls into the **Basel Green Zone**, satisfying international standards for market risk capital requirements.
3.  **Independence:** Analysis of consecutive hits confirmed that the LSTM significantly reduced breach clustering compared to Polynomial and Historical models.

## Getting Started

### Prerequisites
* Python 3.8+
* TensorFlow 2.x
* arch (for GARCH modeling)
* yfinance
* scikit-learn
* pandas/numpy

### Installation
```bash
pip install tensorflow yfinance arch pandas numpy scikit-learn matplotlib
```

### Usage
1.  Generate volatility features using the GARCH and RiskMetrics modules.
2.  Train the LSTM using the custom Pinball Loss function.
3.  Execute the backtesting script to generate comparative POF and independence statistics.
