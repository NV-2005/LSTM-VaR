# Forecasting Value-at-Risk with Deep Quantile Regression

This project implements a Deep Learning framework to forecast **Value-at-Risk (VaR)** for the S&P 500. By combining traditional financial feature engineering (GARCH, RiskMetrics) with the non-linear capabilities of **Long Short-Term Memory (LSTM)** networks, this model directly predicts the 99% VaR threshold using **Deep Quantile Regression**.

## Project Overview

Value-at-Risk (VaR) is a standard metric in financial risk management used to estimate the potential loss of a portfolio over a specific time frame at a given confidence level. Traditional methods (Parametric, Historical Simulation) often fail to capture complex non-linear dependencies or adapt quickly to changing volatility regimes.

This project overcomes those limitations by:
1.  **Hybrid Feature Engineering:** Feeding the LSTM with both raw returns and powerful statistical volatility estimates (GARCH, EWMA).
2.  **Deep Quantile Regression:** Optimizing the network using a custom **Pinball Loss** function to strictly learn the 1st percentile of the return distribution, rather than the mean.
3.  **Robust Backtesting:** Validating the model over nearly 24 years of data (including the 2008 and 2020 crises) to ensure regulatory compliance and statistical validity.

## Methodology

### 1. Data and Feature Engineering
The model utilizes **30 years of S&P 500 data** augmented with the following risk factors:
* **Log Returns:** The primary target for prediction.
* **Rolling GARCH(1,1) Volatility:** A statistical volatility measure refitted every 30 days and updated recursively to capture volatility clustering.
* **RiskMetrics (EWMA) Volatility:** An exponentially weighted moving average (Lambda = 0.94) providing a baseline industry-standard risk measure.
* **VIX Index:** Market sentiment gauge, normalized to a daily volatility scale to capture implied volatility expectations.
* **RSI (Relative Strength Index):** A 14-day momentum indicator used to detect potential mean-reversion or trend-exhaustion points.

### 2. Model Architecture
The core model is a stacked LSTM network built with TensorFlow/Keras:
* **Input Layer:** 60-day sliding window of 5 input features.
* **Hidden Layers:**
    * LSTM Layer (64 units) with Dropout (0.2)
    * LSTM Layer (32 units) with Dropout (0.2)
* **Output Layer:** Dense (1 unit) predicting the VaR threshold directly.
* **Loss Function:** Custom **Quantile Loss** (Pinball Loss) at Alpha = 0.01. This asymmetric loss function penalizes underestimation of risk significantly more than overestimation.

## Results and Backtesting

The model was backtested on **6,223 trading days** of out-of-sample data.

| Metric | Result | Description |
| :--- | :--- | :--- |
| Confidence Level | 99% | Expected breaches only 1% of the time |
| Expected Breaches | 62 | Theoretical number of failures over the test period |
| Actual Breaches | **57** | Actual number of times losses exceeded VaR |
| Failure Rate | **0.92%** | Empirical alpha compared to 1.0% target |

### Regulatory Compliance (Basel Committee)
With a failure rate of **0.92%**, this model falls strictly within the **Green Zone** of the Basel Traffic Light approach. This indicates the model is statistically sound and would not require additional capital multipliers under standard market risk frameworks.

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
1.  Run the data acquisition and feature engineering scripts to generate the volatility surfaces.
2.  Execute the training notebook to compile the LSTM with the custom Quantile Loss function.
3.  Utilize the backtesting module to generate POF (Proportion of Failures) tests and independence metrics.
