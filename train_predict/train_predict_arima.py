#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free Cash Flow Forecasting with ARIMA(1,0,1) (pmdarima)
Annual FCF prediction using Quarterly FCF as lagged exogenous variables

This model uses pmdarima's ARIMA with fixed order (1,0,1) and exogenous regressors (X).
Quarterly FCF data (Q1, Q2, Q3, Q4) is lagged by 1 year, so when predicting
FY_t, we use Q1-Q4 values from FY_{t-1} as exogenous features.

ARIMA(1,0,1) = AR(1) + MA(1), no differencing
- p=1: One autoregressive term (depends on previous year's FCF)
- d=0: No differencing (assume stationary or handle via scaling)
- q=1: One moving average term (depends on previous forecast error)

This approach is comparable to LightGBM with past_covariates.
"""

# Fix OpenMP conflict on macOS (must be BEFORE any imports that load OpenMP)
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pmdarima import ARIMA
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level from train_predict/
DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "result"

# Ensure directories exist
RESULT_DIR.mkdir(exist_ok=True)


def load_fcf_data(ticker: str):
    """Load FCF data for a given ticker with proper cleaning.

    Returns:
        quarterly_df: DataFrame with quarterly FCF data (10-Q)
        annual_df: DataFrame with annual FCF data (10-K)
    """
    logger.info(f"Loading data for {ticker}...")

    quarterly_file = DATA_DIR / f"{ticker}_fcf_10Q.csv"
    annual_file = DATA_DIR / f"{ticker}_fcf_10K.csv"

    if not quarterly_file.exists():
        raise FileNotFoundError(f"Quarterly file not found: {quarterly_file}")
    if not annual_file.exists():
        raise FileNotFoundError(f"Annual file not found: {annual_file}")

    quarterly_df = pd.read_csv(quarterly_file)
    annual_df = pd.read_csv(annual_file)

    # Convert dates
    quarterly_df['end'] = pd.to_datetime(quarterly_df['end'])
    annual_df['end'] = pd.to_datetime(annual_df['end'])

    # Remove rows with NaN in fy or Free Cash Flow
    quarterly_df = quarterly_df[quarterly_df['fy'].notna() & quarterly_df['Free Cash Flow'].notna()]
    annual_df = annual_df[annual_df['fy'].notna() & annual_df['Free Cash Flow'].notna()]

    # Convert fy to integer
    quarterly_df['fy'] = quarterly_df['fy'].astype(int)
    annual_df['fy'] = annual_df['fy'].astype(int)

    # Sort by fy, then by end date (to keep most recent)
    quarterly_df = quarterly_df.sort_values(['fy', 'fp', 'end'])
    annual_df = annual_df.sort_values(['fy', 'end'])

    # Proper deduplication - keep LAST entry (most recent end date) for each (fy, fp) or fy
    quarterly_df = quarterly_df.drop_duplicates(subset=['fy', 'fp'], keep='last')
    annual_df = annual_df.drop_duplicates(subset=['fy'], keep='last')

    # Sort by fy after deduplication
    quarterly_df = quarterly_df.sort_values('fy').reset_index(drop=True)
    annual_df = annual_df.sort_values('fy').reset_index(drop=True)

    logger.info(f"Loaded {len(quarterly_df)} quarterly records and {len(annual_df)} annual records")
    logger.info(f"Annual FY range: {annual_df['fy'].min()} - {annual_df['fy'].max()}")
    logger.info(f"Quarterly FY range: {quarterly_df['fy'].min()} - {quarterly_df['fy'].max()}")

    # Show available quarters per FY
    quarters_per_fy = quarterly_df.groupby('fy')['fp'].apply(list).to_dict()
    logger.info(f"Quarters available per FY: {quarters_per_fy}")

    return quarterly_df, annual_df


def prepare_data(quarterly_df: pd.DataFrame, annual_df: pd.DataFrame, test_years: int = 4):
    """Prepare data for ARIMA training.

    Creates:
    - Annual FCF series (target)
    - Quarterly FCF as lagged exogenous variables (X)
    
    The quarterly data is lagged by 1 year: when predicting FY_t, 
    we use Q1-Q4 from FY_{t-1}.

    Args:
        quarterly_df: Quarterly FCF data
        annual_df: Annual FCF data
        test_years: Number of years to use for test set (default: 4)

    Returns:
        train_y: Training target series
        test_y: Test target series
        train_X: Training exogenous variables (lagged quarterly)
        test_X: Test exogenous variables (lagged quarterly)
        fy_mapping: Dictionary mapping index to actual FY values
        feature_names: List of feature column names used
    """
    logger.info("Preparing data with lagged quarterly exogenous variables...")

    # Create annual DataFrame
    annual_clean = annual_df[['fy', 'end', 'Free Cash Flow']].copy()
    annual_clean = annual_clean.sort_values('fy').reset_index(drop=True)
    fy_values = annual_clean['fy'].tolist()

    # Create mapping from index to FY
    fy_mapping = {i: fy for i, fy in enumerate(fy_values)}

    logger.info(f"FY mapping: {fy_mapping}")

    # Create pivot table for quarterly data
    quarterly_pivot = quarterly_df.pivot_table(
        values='Free Cash Flow',
        index='fy',
        columns='fp',
        aggfunc='first'
    )

    # Ensure all quarters are present as columns
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        if quarter not in quarterly_pivot.columns:
            quarterly_pivot[quarter] = np.nan

    # Reorder columns
    quarterly_pivot = quarterly_pivot[['Q1', 'Q2', 'Q3', 'Q4']]

    # Align quarterly data with annual FY values
    quarterly_aligned = quarterly_pivot.reindex(fy_values).reset_index(drop=True)

    logger.info(f"Annual data: {len(annual_clean)} fiscal years")
    logger.info(f"Quarterly data aligned: {len(quarterly_aligned)} fiscal years")

    # Check which quarters have data
    quarters_with_data = [col for col in quarterly_aligned.columns if quarterly_aligned[col].notna().any()]
    logger.info(f"Quarters with data: {quarters_with_data}")
    logger.info(f"Non-NaN counts per quarter: {quarterly_aligned.notna().sum().to_dict()}")

    # Create target series
    y = annual_clean['Free Cash Flow'].values

    # Create lagged exogenous variables
    # Lag by 1: when predicting FY_t, use Q1-Q4 from FY_{t-1}
    # This is equivalent to past_covariates with lag=-1 in Darts
    X = None
    feature_names = []
    
    if quarters_with_data:
        # Only use quarters that have data
        quarterly_for_X = quarterly_aligned[quarters_with_data].copy()
        
        # Shift by 1 to create lag (row i gets values from row i-1)
        quarterly_lagged = quarterly_for_X.shift(1)
        
        # Rename columns to indicate lag
        quarterly_lagged.columns = [f"{col}_lag1" for col in quarterly_lagged.columns]
        feature_names = list(quarterly_lagged.columns)
        
        X = quarterly_lagged.values
        
        logger.info(f"Created lagged exogenous features: {feature_names}")
        logger.info(f"Exogenous matrix shape: {X.shape}")
    else:
        logger.warning("No quarterly data available! Training without exogenous variables.")

    # Split train/test - last N years go to test
    split_point = len(y) - test_years

    if split_point < 3:
        raise ValueError(f"Not enough training data: only {split_point} years for training. Need at least 3.")

    train_y = y[:split_point]
    test_y = y[split_point:]

    train_X = None
    test_X = None
    if X is not None:
        train_X = X[:split_point]
        test_X = X[split_point:]

    # Log the actual FY split
    train_fys = [fy_mapping[i] for i in range(split_point)]
    test_fys = [fy_mapping[i] for i in range(split_point, len(y))]
    logger.info(f"Train FYs: {train_fys}")
    logger.info(f"Test FYs: {test_fys}")
    logger.info(f"Train samples: {len(train_y)}, Test samples: {len(test_y)}")

    return train_y, test_y, train_X, test_X, fy_mapping, feature_names


def train_arima_model(
        train_y: np.ndarray,
        test_y: np.ndarray,
        train_X: np.ndarray = None,
        test_X: np.ndarray = None,
        order: tuple = (1, 0, 1),
):
    """Train ARIMA model with fixed order and optional exogenous variables.

    Args:
        train_y: Training target series
        test_y: Test target series
        train_X: Training exogenous variables (optional)
        test_X: Test exogenous variables (optional)
        order: ARIMA order tuple (p, d, q). Default: (1, 0, 1)

    Returns:
        model: Trained ARIMA model
        predictions: Predictions on test set
        y_scaler: Scaler for target variable
        X_scaler: Scaler for exogenous variables (or None)
    """
    p, d, q = order
    logger.info(f"Training ARIMA({p},{d},{q}) model...")
    logger.info(f"Training samples: {len(train_y)}")

    use_exog = train_X is not None and not np.all(np.isnan(train_X))
    
    # Scale the target variable
    y_scaler = StandardScaler()
    train_y_scaled = y_scaler.fit_transform(train_y.reshape(-1, 1)).flatten()

    # Scale and handle exogenous variables
    X_scaler = None
    train_X_clean = None
    test_X_clean = None
    
    if use_exog:
        # Check how many valid (non-NaN) rows we have in training exogenous
        valid_rows_train = ~np.any(np.isnan(train_X), axis=1)
        n_valid_train = np.sum(valid_rows_train)
        
        logger.info(f"Valid training rows with exogenous data: {n_valid_train}/{len(train_X)}")
        
        # We need at least a few valid rows to use exogenous
        if n_valid_train >= 3:
            # For training, we'll use only the portion where exogenous is valid
            # This means we lose the first row (due to lag)
            first_valid_idx = np.where(valid_rows_train)[0][0]
            
            train_y_for_fit = train_y_scaled[first_valid_idx:]
            train_X_for_fit = train_X[first_valid_idx:]
            
            # Scale exogenous variables (handle remaining NaNs by filling with 0 after scaling)
            X_scaler = StandardScaler()
            
            # Fit scaler on non-NaN values
            X_for_scaling = train_X_for_fit.copy()
            # Replace NaNs temporarily for fitting
            col_means = np.nanmean(X_for_scaling, axis=0)
            for j in range(X_for_scaling.shape[1]):
                X_for_scaling[np.isnan(X_for_scaling[:, j]), j] = col_means[j]
            
            X_scaler.fit(X_for_scaling)
            
            # Transform train and test
            train_X_clean = X_scaler.transform(X_for_scaling)
            
            # For test, also fill NaNs
            test_X_for_scaling = test_X.copy()
            for j in range(test_X_for_scaling.shape[1]):
                test_X_for_scaling[np.isnan(test_X_for_scaling[:, j]), j] = col_means[j]
            test_X_clean = X_scaler.transform(test_X_for_scaling)
            
            logger.info(f"Using exogenous variables. Training on {len(train_y_for_fit)} samples (after removing NaN rows)")
        else:
            logger.warning("Not enough valid exogenous data rows. Training without exogenous variables.")
            use_exog = False
            train_y_for_fit = train_y_scaled
    else:
        logger.info("No exogenous variables provided. Training on target series only.")
        train_y_for_fit = train_y_scaled

    # Create ARIMA model with fixed order
    logger.info(f"Fitting ARIMA{order}...")
    
    model = ARIMA(
        order=order,
        suppress_warnings=True,
        maxiter=100,
    )
    
    # Fit the model
    if use_exog and train_X_clean is not None:
        model.fit(train_y_for_fit, X=train_X_clean)
    else:
        model.fit(train_y_for_fit)
    
    logger.info(f"Model fitted successfully")
    logger.info(f"AIC: {model.aic():.2f}")
    
    # Forecast
    n_predict = len(test_y)
    logger.info(f"Forecasting {n_predict} periods...")
    
    if use_exog and test_X_clean is not None:
        predictions_scaled = model.predict(n_periods=n_predict, X=test_X_clean)
    else:
        predictions_scaled = model.predict(n_periods=n_predict)
    
    # Inverse transform predictions
    predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    return model, predictions, y_scaler, X_scaler

def save_predictions(predictions: np.ndarray, actuals: np.ndarray, ticker: str, fy_mapping: dict, train_size: int):
    """Save predictions and actuals to CSV.

    Args:
        predictions: Predicted values
        actuals: Actual values
        ticker: Stock ticker
        fy_mapping: Dictionary mapping index to FY
        train_size: Size of training set (to offset test indices)
    """
    # Map test indices to FY values
    test_fys = [fy_mapping[train_size + i] for i in range(len(predictions))]

    result_df = pd.DataFrame({
        'fy': test_fys,
        'actual': actuals,
        'predicted': predictions,
    })

    output_file = RESULT_DIR / f"{ticker}_fcf_annual_predictions_arima.csv"
    result_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to: {output_file}")


def train_and_predict(ticker: str, test_years: int = 4, order: tuple = (1, 0, 1)):
    """Main pipeline for training and prediction.

    Args:
        ticker: Stock ticker to process
        test_years: Number of years to use for test set (default: 4)
        order: ARIMA order tuple (p, d, q). Default: (1, 0, 1)
    """
    p, d, q = order
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Starting FCF forecasting pipeline for {ticker} (ARIMA({p},{d},{q}))")
    logger.info(f"{'=' * 80}\n")

    # 1. Load data
    quarterly_df, annual_df = load_fcf_data(ticker)

    # 2. Prepare data with lagged exogenous variables
    train_y, test_y, train_X, test_X, fy_mapping, feature_names = prepare_data(
        quarterly_df, annual_df, test_years
    )

    # 3. Train model
    logger.info(f"\n--- Training ARIMA({p},{d},{q}) Model ---")
    model, predictions, y_scaler, X_scaler = train_arima_model(
        train_y, test_y, train_X, test_X, order=order
    )

    # 4. Save predictions
    save_predictions(predictions, test_y, ticker, fy_mapping, len(train_y))

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Pipeline completed for {ticker}")
    logger.info(f"{'=' * 80}\n")
    
    return model, predictions


# --- Main Execution Block ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = input("Enter ticker symbol: ").strip().upper()

    try:
        # Fixed ARIMA(1,0,1) order
        train_and_predict(ticker, test_years=4, order=(1, 0, 1))
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
