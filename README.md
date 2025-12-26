# 📊 FCF Forecasting System

Unified Free Cash Flow forecasting system using Nixtla's StatsForecast and MLForecast libraries.

## 🎯 What This Does

Forecasts **annual Free Cash Flow (FCF)** for stocks using:
- Historical 10-K (annual) and 10-Q (quarterly) data from SEC EDGAR filings
- Multiple forecasting models (statistical, ML, and deep learning)
- Quarterly data as exogenous variables (optional)

**Target**: Predict next 4 years of FCF to enhance DCF valuation models.

---

## 📁 Project Structure

```
project/
├── config.py                    # Centralized configuration
├── model.py                     # Model factory (StatsForecast, MLForecast, Chronos)
├── train_predict.py             # Unified training script
├── fcf_extractor.py             # Download FCF data from SEC EDGAR
├── 1_exploration_analysis.py    # Plot historical data
├── 2_model_comparison_analysis.ipynb  # Compare all models (Jupyter)
├── tickers.txt                  # List of CIK,TICKER pairs
├── data/
│   ├── {TICKER}_fcf_10Q.csv    # Quarterly FCF data
│   └── {TICKER}_fcf_10K.csv    # Annual FCF data
└── result/
    ├── {TICKER}_fcf_annual_predictions_{MODEL}.csv
    └── {TICKER}_fcf_10K_with_models.png
```

---

## 📝 Usage

### Basic Commands

```bash
# 1. Download FCF data from SEC
python fcf_extractor.py

# 2. Train a single model for one ticker
python train_predict.py AAPL autoarima

# 3. Train all models for one ticker
python train_predict.py NVDA

# 4. Compare all models (Jupyter notebook)
jupyter notebook model_comparison_analysis.ipynb
```

### Available Models

| Model | Type | Description |
|-------|------|-------------|
| `autoarima` | Statistical | Auto-tuned ARIMA |
| `arima` | Statistical | Fixed ARIMA(1,0,1) |
| `exponential_smoothing` | Statistical | AutoETS |
| `moving_average` | Statistical | Window Average (3 years) |
| `lightgbm` | ML | Gradient Boosting with lags |
| `chronos2` | DL | Pretrained Transformer |

---

## ⚙️ Configuration

All settings are in `config.py`. Key parameters:

### Data & Train/Test Split
```python
DATA_DIR = "data"
RESULT_DIR = "result"
TEST_YEARS = 4  # Last 4 years for testing
MIN_TRAIN_YEARS = 3  # Minimum training years required
```

### Model Selection
```python
MODELS = {
    "autoarima": "statsforecast",
    "arima": "statsforecast",
    "exponential_smoothing": "statsforecast",
    "moving_average": "statsforecast",
    "lightgbm": "mlforecast",
    "chronos2": "chronos",
}
```

### Quarterly Covariates
```python
USE_QUARTERLY_COVARIATES = True  # Use Q1-Q4 as features
QUARTERLY_LAG = -1  # Use previous year's quarters
```

### Model-Specific Settings

**AutoARIMA:**
```python
AUTOARIMA_CONFIG = {
    "seasonal": False,
    "max_p": 3,  # Max AR terms
    "max_q": 3,  # Max MA terms
    "max_d": 2,  # Max differencing
}
```

**LightGBM:**
```python
LIGHTGBM_CONFIG = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "lags": [1, 2, 3],  # Use last 3 years
}
```

See `config.py` for complete settings.

---

## 🔧 Common Tasks

### Add a New Ticker

Edit `tickers.txt`:
```
0000789019,MSFT
0001318605,TSLA
0001652044,GOOG
YOUR_CIK,YOUR_TICKER  # Add here
```

Then run:
```bash
python fcf_extractor.py
python train_predict.py YOUR_TICKER
```

### Change Test Period

In `config.py`:
```python
TEST_YEARS = 3  # Was 4 - use 3 years for testing
```

### Disable Quarterly Covariates

In `config.py`:
```python
USE_QUARTERLY_COVARIATES = False
```

### Tune AutoARIMA

In `config.py`:
```python
AUTOARIMA_CONFIG = {
    "max_p": 5,  # Was 3 - search more AR terms
    "max_q": 5,  # Was 3 - search more MA terms
}
```

### Increase LightGBM Complexity

In `config.py`:
```python
LIGHTGBM_CONFIG = {
    "n_estimators": 500,  # Was 100
    "max_depth": 8,      # Was 5
    "lags": [1, 2, 3, 4, 5],  # Was [1, 2, 3]
}
```

---

## 🐛 Troubleshooting

### ModuleNotFoundError: No module named 'statsforecast'
```bash
conda install -c conda-forge statsforecast mlforecast lightgbm
```

### Chronos model download fails
Chronos downloads ~1GB on first run. Ensure stable internet or use smaller model:
```python
# In config.py
CHRONOS_CONFIG["model_name"] = "amazon/chronos-t5-small"  # ~200MB
```

---

## 📝 File Formats

### tickers.txt
```
# Format: CIK,TICKER
0000789019,MSFT
0001318605,TSLA
0001652044,GOOG
```

### Output CSV (predictions)
```csv
fy,actual,predicted
2022,98765432100,95432109876
2023,102345678900,99876543210
2024,106789012345,103456789012
2025,110123456789,107890123456
```

---

## 🎓 Key Concepts

### Why Use Quarterly Data as Covariates?

Annual FCF = sum of quarterly cash flows. Using Q1-Q4 from previous year as features helps models capture:
- Seasonality patterns
- Quarterly momentum
- Business cycle effects

**Example**: When predicting FY2024, we use Q1-Q4 from FY2023 as inputs.

### Why Multiple Models?

Different models excel at different patterns:
- **ARIMA**: Stable trends, autocorrelation
- **ETS**: Smooth exponential trends
- **LightGBM**: Complex non-linear patterns
- **Chronos**: Transfer learning from many series

**Ensemble** (combining models) often outperforms individual models.

---

## 🚀 Quick Start Checklist

- [ ] Create `tickers.txt` with your stocks
- [ ] Download data: `python fcf_extractor.py`
- [ ] Test one model: `python train_predict.py AAPL autoarima`
- [ ] Run comparison: Open `model_comparison_analysis.ipynb`
- [ ] Review results in `result/` folder

---

## 🤝 Contributing

Contributions welcome! When adding features:
1. Update `config.py` for new parameters
2. Add model to `model.py` factory

---

## 🚀 Installation

### Mac (Apple Silicon M1/M2/M3 or Intel)

```bash
# Install Miniforge for Apple Silicon
# Download from: https://github.com/conda-forge/miniforge/releases/latest
# File: Miniforge3-MacOSX-arm64.sh (Apple Silicon) or Miniforge3-MacOSX-x86_64.sh (Intel)

# Create environment
conda create -n fcf python=3.10
conda activate fcf

# Install dependencies
conda install -c conda-forge --file requirements.txt
```

### Linux / Windows

```bash
# Using conda
conda create -n fcf python=3.10
conda activate fcf
conda install -c conda-forge --file requirements.txt

# OR using venv + pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

**Happy Forecasting! 🚀**
