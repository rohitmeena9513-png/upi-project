# 📈 UPI Transaction Forecasting

A time-series forecasting project for analyzing and forecasting **monthly UPI (Unified Payments Interface) transaction volume in India** using classical statistical models and machine-learning regression models.

The project compares **Naive, Seasonal Naive, ARIMA, SARIMA, Random Forest, and XGBoost** models and uses the best-performing model to forecast UPI transaction volume for the next six months.

---

## 📌 Project Overview

Digital payments have grown rapidly in India, making UPI transaction volume an important metric for understanding payment-system adoption and planning infrastructure capacity.

This project analyzes monthly UPI data and builds forecasting models to:

- Understand long-term UPI growth.
- Identify trend and seasonality.
- Test whether the series is stationary.
- Build classical time-series forecasting models.
- Build machine-learning forecasting models using lag and rolling features.
- Compare models using RMSE, MAE, and MAPE.
- Generate a six-month future forecast.
- Extract business insights from the forecasting results.

---

## 🎯 Objectives

- Analyze historical UPI transaction volume and value.
- Perform exploratory time-series analysis.
- Identify trend, seasonality, and volatility.
- Perform the Augmented Dickey-Fuller (ADF) stationarity test.
- Apply first-order and seasonal differencing.
- Analyze ACF and PACF.
- Establish Naive and Seasonal Naive baselines.
- Build and tune ARIMA and SARIMA models.
- Engineer lag, rolling-statistic, and calendar features.
- Train Random Forest and XGBoost regressors.
- Compare all forecasting approaches.
- Forecast UPI volume for the next six months.
- Translate model results into business recommendations.

---

## 📊 Dataset

The project uses:

```text
upi_forecasting_dataset.csv
```

The dataset contains **79 monthly observations and 21 columns** covering several Indian digital-payment instruments.

Examples include:

- UPI
- IMPS
- NEFT
- RTGS
- Mobile Payments
- Internet Payments
- AePS
- PPI Wallets

### Main UPI Variables

| Variable | Description |
|---|---|
| `UPI_Volume_Lakh` | Monthly UPI transaction volume in lakh transactions |
| `UPI_Value_RsCr` | Monthly UPI transaction value in ₹ crore |
| `UPI_Avg_Ticket_Size_Rs` | Average value per UPI transaction |

The dataset covers:

**November 2019 → May 2026**

---

## 🧹 Data Cleaning

The preprocessing pipeline includes:

1. Parsing the `Date` column.
2. Sorting observations chronologically.
3. Setting `Date` as the time-series index.
4. Assigning a monthly frequency (`MS`).
5. Checking duplicate observations.
6. Handling missing values.
7. Checking for negative numeric values.

The dataset initially contained missing values in:

```text
Num_UPI_QR_Lakh
```

These were handled using linear interpolation followed by backfilling.

No duplicate rows or negative numeric values were found.

---

# 🔍 Exploratory Data Analysis

The project analyzes:

### UPI Volume

The target series is:

```text
UPI_Volume_Lakh
```

It increased from approximately:

**12,188 lakh transactions in Nov-2019**

to:

**232,019 lakh transactions in May-2026**

representing approximately **1,804% growth** over the sample period.

### UPI Value

The project also tracks:

```text
UPI_Value_RsCr
```

### Average Ticket Size

Average transaction size is calculated as:

```python
UPI_Avg_Ticket_Size_Rs =
(UPI_Value_RsCr × 10^7) /
(UPI_Volume_Lakh × 10^5)
```

The average ticket size across the sample is approximately:

**₹1,589 per transaction.**

---

# 📈 Trend Analysis

Month-over-month and year-over-year growth are calculated.

```python
upi['MoM_Growth_%'] = upi['UPI_Volume_Lakh'].pct_change() * 100
upi['YoY_Growth_%'] = upi['UPI_Volume_Lakh'].pct_change(12) * 100
```

### Key Result

- Average YoY growth: **65.29%**
- Latest YoY growth: **24.22%**

The analysis shows strong long-term structural growth in UPI transaction volume.

---

# 🔄 Seasonality Analysis

A 12-month seasonal pattern is investigated using:

- Calendar-month boxplots
- Seasonal decomposition
- Rolling statistics

The analysis indicates clear **12-month seasonality**.

Observed seasonal behavior includes:

- Lower volumes around February, partly reflecting the shorter month.
- Increasing activity toward the end of the year.
- Stronger volumes around festive and quarter-end periods.

---

# 📉 Stationarity Analysis

The Augmented Dickey-Fuller (ADF) test is used to determine whether the UPI volume series is stationary.

### Original Series

```text
ADF Statistic: -0.1768
p-value: 0.9412
Conclusion: Non-Stationary
```

### First Difference

```text
ADF Statistic: -1.7881
p-value: 0.3864
Conclusion: Non-Stationary
```

### First + Seasonal Difference (lag 12)

```text
ADF Statistic: -8.8398
p-value: 0.0000
Conclusion: Stationary
```

Therefore, first-order and seasonal differencing are used as part of the SARIMA modeling approach.

---

# 📊 ACF & PACF

Autocorrelation and partial autocorrelation plots are generated to understand temporal dependencies and help guide ARIMA/SARIMA model selection.

The analysis uses up to **24 lags**.

---

# ✂️ Train-Test Split

A chronological split is used because randomly shuffling time-series observations would introduce future information into the training data.

### Training Set

```text
Nov-2019 → May-2025
67 months
```

### Test Set

```text
Jun-2025 → May-2026
12 months
```

No random shuffling is performed.

---

# 🧪 Baseline Models

Two simple forecasting baselines are established.

### 1. Naive Forecast

The last observed training value is repeated for every future test month.

### 2. Seasonal Naive

The value from 12 months earlier is used as the forecast.

### Baseline Results

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Naive | 26,009.71 | 22,525.10 | 10.42% |
| Seasonal Naive | 46,343.92 | 46,248.05 | 22.27% |

---

# 📐 ARIMA

An ARIMA model is selected by searching across:

```text
p = 0 to 3
d = 1
q = 0 to 3
```

The model with the lowest AIC was:

```text
ARIMA(3, 1, 2)
```

### AIC

```text
1274.43
```

### Test Performance

| Metric | Value |
|---|---:|
| RMSE | **6,343.71** |
| MAE | **4,570.02** |
| MAPE | **2.25%** |

---

# 📐 SARIMA

Because the data exhibits strong 12-month seasonality, SARIMA models are also evaluated.

The search considers multiple non-seasonal and seasonal configurations.

### Best Model

```text
SARIMA(2,1,2) × (0,1,1,12)
```

### AIC

```text
740.78
```

### Test Performance

| Metric | Value |
|---|---:|
| RMSE | **3,109.11** |
| MAE | **2,679.49** |
| MAPE | **1.30%** |

SARIMA is the **best-performing model in the project**.

---

# 🤖 Machine Learning Forecasting

Machine-learning models are trained using engineered time-series features.

## Lag Features

The following lag variables are created:

```text
lag_1
lag_2
lag_3
lag_6
lag_12
```

## Rolling Features

Rolling mean and standard deviation are calculated over:

```text
3 months
6 months
12 months
```

## Calendar Features

The model also receives:

- Month
- Quarter
- Month sine
- Month cosine
- Time index

The sine/cosine transformation represents the cyclical nature of calendar months.

---

# 🌲 Random Forest

A `RandomForestRegressor` is tuned using `GridSearchCV` with `TimeSeriesSplit`.

### Search Parameters

- `n_estimators`
- `max_depth`
- `min_samples_leaf`

### Best Parameters

```text
n_estimators: 400
max_depth: None
min_samples_leaf: 1
```

### Test Performance

| Metric | Value |
|---|---:|
| RMSE | **33,677.39** |
| MAE | **30,735.67** |
| MAPE | **14.34%** |

---

# 🚀 XGBoost

An `XGBRegressor` is tuned using `GridSearchCV` and time-series cross-validation.

### Best Parameters

```text
learning_rate: 0.1
max_depth: 4
n_estimators: 400
subsample: 0.8
```

### Test Performance

| Metric | Value |
|---|---:|
| RMSE | **27,601.32** |
| MAE | **24,507.67** |
| MAPE | **11.39%** |

XGBoost performs better than Random Forest, but it is still outperformed by ARIMA and SARIMA.

---

# 🏆 Model Comparison

The final comparison is:

| Rank | Model | RMSE | MAE | MAPE |
|---:|---|---:|---:|---:|
| 🥇 | **SARIMA(2,1,2) × (0,1,1,12)** | **3,109.11** | **2,679.49** | **1.30%** |
| 🥈 | ARIMA(3,1,2) | 6,343.71 | 4,570.02 | 2.25% |
| 🥉 | Naive | 26,009.71 | 22,525.10 | 10.42% |
| 4 | XGBoost | 27,601.32 | 24,507.67 | 11.39% |
| 5 | Random Forest | 33,677.39 | 30,735.67 | 14.34% |
| 6 | Seasonal Naive | 46,343.92 | 46,248.05 | 22.27% |

### 🏆 Best Model

**SARIMA(2,1,2) × (0,1,1,12)**

It achieves the lowest error across all three evaluation metrics.

---

# 🔎 Feature Importance

Feature importance is analyzed for the Random Forest and XGBoost models.

The most important features are dominated by:

- Recent UPI volume (`lag_1`)
- Previous-year UPI volume (`lag_12`)
- Short-term rolling means

This indicates that both **recent momentum** and **same-month historical behavior** are important predictors of UPI transaction volume.

---

# 🔮 Future UPI Forecast

The best-performing SARIMA model is retrained using the full available series and used to forecast the next **6 months**.

### Forecast

| Month | Forecast UPI Volume (Lakh Transactions) |
|---|---:|
| Jun-2026 | 229,897 |
| Jul-2026 | 239,933 |
| Aug-2026 | 245,555 |
| Sep-2026 | 242,825 |
| Oct-2026 | 254,330 |
| Nov-2026 | 250,946 |

### Month-over-Month Forecast Growth

| Month | MoM Growth |
|---|---:|
| Jul-2026 | +4.37% |
| Aug-2026 | +2.34% |
| Sep-2026 | -1.11% |
| Oct-2026 | +4.74% |
| Nov-2026 | -1.33% |

The forecast suggests continued high UPI transaction volumes with moderate month-to-month fluctuations.

---

# 💼 Business Insights

### 1. Strong Long-Term Growth

UPI volume increased by approximately **1,804%** between Nov-2019 and May-2026.

### 2. Sustained Adoption

Average YoY growth across the sample was approximately **65.3%**, indicating substantial structural growth in UPI usage.

### 3. Strong Seasonality

A clear annual seasonal pattern is present, with activity generally building toward festive and quarter-end periods.

### 4. SARIMA Outperforms ML Models

The best SARIMA model achieved a **1.30% MAPE**, substantially lower than the machine-learning models tested.

### 5. Future Volume Remains High

The six-month forecast remains around **230,000–254,000 lakh transactions per month**, indicating continued high transaction-system demand.

---

# 💡 Recommendations

### Capacity Planning

Scale UPI/NPCI infrastructure ahead of seasonal peaks identified through the seasonality analysis.

### Model Monitoring

Track forecasting error monthly and retrain the model when performance deteriorates beyond the historical baseline.

### Broader Payment Dashboard

UPI volume is analyzed alongside other digital-payment instruments. A combined dashboard could provide earlier signals of changes in digital-payment behavior.

### Extend Forecasting Targets

The same pipeline can be applied to:

- `UPI_Value_RsCr`
- `UPI_Avg_Ticket_Size_Rs`

This would distinguish transaction-count growth from transaction-value and transaction-size trends.

---

# 🛠️ Tech Stack

### Programming

- Python
- Jupyter Notebook

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Statistical Modeling

- Statsmodels
- ARIMA
- SARIMA
- ADF Test
- ACF/PACF
- Seasonal Decomposition

### Machine Learning

- Scikit-learn
- Random Forest
- GridSearchCV
- TimeSeriesSplit
- XGBoost

---

# 📁 Project Structure

```text
UPI-Transaction-Forecasting/
│
├── UPI_Transaction_Forecasting-2.ipynb
├── upi_forecasting_dataset.csv
├── README.md
│
└── images/
    ├── upi_trend.png
    ├── seasonality.png
    ├── decomposition.png
    ├── acf_pacf.png
    ├── model_comparison.png
    └── future_forecast.png
```

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/UPI-Transaction-Forecasting.git
cd UPI-Transaction-Forecasting
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels xgboost jupyter
```

### 3. Add the dataset

Place:

```text
upi_forecasting_dataset.csv
```

in the project directory.

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
UPI_Transaction_Forecasting-2.ipynb
```

and execute the notebook cells sequentially.

---

# 🔮 Future Improvements

- Add more recent UPI observations as they become available.
- Perform rolling-origin / walk-forward validation.
- Tune SARIMA hyperparameters using a broader search.
- Compare SARIMA with Prophet, ETS, TBATS, and other forecasting methods.
- Test LSTM/GRU deep-learning models.
- Build multivariate forecasting models using other payment instruments.
- Add external variables such as GDP, internet penetration, smartphone adoption, and policy changes.
- Deploy the forecasting pipeline as an interactive dashboard.
- Automate monthly data updates and model retraining.

---

## 👨‍💻 Author

**Your Name**

⭐ If you found this project useful, consider starring the repository.
