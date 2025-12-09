# forecastly

Financial forecasting tools for SEC EDGAR data analysis and Free Cash Flow prediction.

## Installation

### Mac M1/M2/M3 (Apple Silicon)

**Important**: Use native ARM64 Python to avoid OpenMP conflicts and segmentation faults.

1. **Remove old Miniconda/Anaconda** (if installed under Rosetta):
```bash
rm -rf /usr/local/Caskroom/miniconda
rm -rf ~/miniconda3
rm -rf ~/miniforge3  # if exists
```

2. **Install Miniforge ARM64**:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh
rm Miniforge3-MacOSX-arm64.sh  # cleanup installer
```

Close and reopen your terminal.

3. **Create environment**:
```bash
# Verify ARM64
python -c "import platform; print(platform.machine())"  # Must print: arm64

# Create environment with Python 3.12
conda create -n forecasting python=3.12 -y
conda activate forecasting

# Install dependencies in correct order (avoids OpenMP conflicts)
conda install -c conda-forge "numpy<2.0" lightgbm -y
conda install -c conda-forge pandas scikit-learn matplotlib requests -y
pip install darts
pip install polars-lts-cpu  # optional, for better CPU compatibility
```

### Mac Intel

**Note**: Intel Macs may also experience OpenMP conflicts. Follow similar steps to Apple Silicon:

1. **Install Miniconda or Miniforge**:
```bash
# For Intel Mac, use standard Miniconda
curl -L -O "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
bash Miniconda3-latest-MacOSX-x86_64.sh
rm Miniconda3-latest-MacOSX-x86_64.sh
```

Close and reopen your terminal.

2. **Create environment**:
```bash
conda create -n forecasting python=3.12 -y
conda activate forecasting

# Install dependencies in correct order (avoids OpenMP conflicts)
conda install -c conda-forge "numpy<2.0" lightgbm -y
conda install -c conda-forge pandas scikit-learn matplotlib requests -y
pip install darts
```

### Windows / Linux

```bash
# Create environment
conda create -n forecasting python=3.12 -y
conda activate forecasting

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Extract FCF Data from SEC EDGAR

```bash
python fetch_fcf_data.py
```

This will read tickers from `tickers.txt` and save data to `/data` folder.

### 2. Analyze FCF Time Series

Run the Jupyter notebook:
```bash
jupyter notebook fcf_analysis.ipynb
```

Plots are saved to `/result` folder.

### 3. Train FCF Forecasting Model

```bash
python train_predict/train_predict_lightgbm.py AAPL
```

Predicts annual FCF using quarterly data as covariates.

## Project Structure

```
forecastly/
├── data/                      # FCF datasets (10-Q and 10-K)
├── result/                    # Prediction outputs and plots
├── train_predict/             # Forecasting models
│   └── train_predict_lightgbm.py  # LightGBM forecasting model
├── fetch_fcf_data.py          # SEC EDGAR data extraction
├── fcf_analysis.ipynb         # Time series visualization
├── tickers.txt                # List of stock tickers to process
└── requirements.txt           # Python dependencies
```

## Notes

- **Apple Silicon users**: Always use conda-forge channel to avoid OpenMP conflicts
- **NumPy version**: Darts requires NumPy < 2.0
- **Polars**: Use `polars-lts-cpu` on Apple Silicon for better compatibility