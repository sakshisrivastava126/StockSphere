# StockSphere

StockSphere is a machine learning based demand forecasting and inventory optimization system designed for retail businesses. The project leverages historical transaction data to predict future demand, reduce forecasting errors, and support inventory planning decisions.

The primary objective is to help retailers minimize losses caused by stockouts and overstocking by generating accurate demand forecasts and data-driven replenishment recommendations.

---

## Problem Statement

Inventory planning is one of the most critical challenges in retail operations.

Underestimating demand can lead to stockouts, lost sales, and poor customer experience, while overestimating demand increases holding costs and ties up capital in excess inventory.

StockSphere addresses this problem by forecasting future demand using historical sales patterns and translating those forecasts into actionable inventory recommendations.

---

## Features

* Retail demand forecasting using machine learning
* Advanced time-series feature engineering
* Multi-horizon forecasting strategy
* Business-aware inventory recommendations
* Forecast evaluation using industry-standard metrics
* Demand trend and prediction visualizations
* Safety stock calculation for inventory planning

---

## Dataset

The project uses retail transaction data containing:

* Invoice information
* Store branch details
* Product categories
* Quantity sold
* Transaction date and time
* Unit prices
* Payment methods
* Customer ratings
* Profit margins

The raw transactional data is cleaned, standardized, and aggregated into daily demand series before model training.

---

## Project Architecture

```text
Raw Retail Transactions
          |
          v
Data Cleaning & Validation
          |
          v
Daily Demand Aggregation
          |
          v
Feature Engineering
          |
          v
Forecasting Models
   |               |
   |               |
LightGBM       Prophet
   |               |
   +-------+-------+
           |
           v
Demand Forecast
           |
           v
Inventory Recommendation Engine
           |
           v
Safety Stock Calculation
```

---

## Feature Engineering

The forecasting pipeline incorporates multiple categories of time-series features:

### Temporal Features

* Day of week
* Month
* Quarter
* Week of year
* Weekend indicators
* Cyclical date encodings

### Lag Features

* 1-day lag
* 3-day lag
* 5-day lag
* 7-day lag
* 14-day lag
* 30-day lag

### Rolling Statistics

* Rolling averages
* Rolling standard deviations
* Rolling minimum and maximum values

### Trend Features

* Demand momentum
* Demand acceleration
* Short-term trend shifts

### Volatility Features

* Demand variability
* Spike detection
* Coefficient of variation

### Holiday Features

* Holiday indicators
* Pre-holiday and post-holiday effects

---

## Models

### LightGBM

LightGBM serves as the primary forecasting model for capturing short-term demand fluctuations and local patterns.

### Prophet

Prophet is used to model long-term trends, seasonality, and holiday effects.

### Hybrid Forecasting Strategy

The project combines the strengths of both approaches:

* Short horizons: LightGBM
* Medium horizons: Blended forecasts
* Long horizons: Prophet

This allows the system to remain responsive to short-term changes while maintaining stable long-term forecasts.

---

## Evaluation Metrics

Model performance is evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* WAPE (Weighted Absolute Percentage Error)

In addition to statistical metrics, StockSphere incorporates business-aware cost evaluation by assigning higher penalties to stockouts than overstock situations.

---

## Inventory Optimization Logic

Forecasts are converted into inventory recommendations using:

```text
Recommended Inventory
=
Forecasted Demand
+
Safety Stock
```

The system prioritizes reducing stockout risk while maintaining reasonable inventory levels.

---

## Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* Prophet

### Visualization

* Matplotlib
* Jupyter Notebook

### Development Tools

* Git
* GitHub

---

## Project Structure

```text
StockSphere/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── notebooks/
│
├── src/
│   ├── data_processing/
│   ├── feature_engineering/
│   ├── models/
│   └── forecasting/
│
├── reports/
├── visualizations/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Future Improvements

* Store-level demand forecasting
* Category-level inventory planning
* FastAPI prediction service
* Interactive React dashboard
* Automated model retraining pipeline
* Database integration
* Forecast monitoring and drift detection

---

## Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/StockSphere.git
cd StockSphere
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Training Pipeline

```bash
python src/train.py
```

### Generate Forecasts

```bash
python src/predict.py
```

---

## License

This project is intended for educational and research purposes.
