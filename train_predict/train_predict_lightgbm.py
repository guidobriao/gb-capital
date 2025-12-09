#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free Cash Flow Forecasting with LightGBM
Annual FCF prediction using Quarterly FCF as past covariates

FIXES:
1. Proper deduplication - keep rows with valid FCF, prioritize most recent 'end' date
2. lags_past_covariates uses only strictly negative integers ([-1])
3. FY converted to integer RangeIndex for Darts compatibility
4. Filter out all-NaN covariate columns to avoid Scaler issues
5. OpenMP conflict fix for macOS
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
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel

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

    # FIX 1: Proper deduplication - keep LAST entry (most recent end date) for each (fy, fp) or fy
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


def prepare_time_series(quarterly_df: pd.DataFrame, annual_df: pd.DataFrame, test_years: int = 4):
    """Prepare Darts TimeSeries for training using integer RangeIndex.

    Creates:
    - Annual FCF with integer index (target)
    - Quarterly FCF (Q1, Q2, Q3, Q4) with integer index (past covariates)

    Args:
        quarterly_df: Quarterly FCF data
        annual_df: Annual FCF data
        test_years: Number of years to use for test set (default: 4)

    Returns:
        train_annual: Annual FCF TimeSeries for training (target)
        test_annual: Annual FCF TimeSeries for testing (target)
        train_quarterly_cov: Quarterly TimeSeries for training (covariates)
        test_quarterly_cov: Quarterly TimeSeries for testing (covariates)
        fy_mapping: Dictionary mapping index to actual FY values
    """
    logger.info("Preparing Darts TimeSeries with integer RangeIndex...")

    # Create annual DataFrame - use end date to derive the actual fiscal year
    # (SEC filings have fy as the filing year, not the period year)
    annual_clean = annual_df[['fy', 'end', 'Free Cash Flow']].copy()

    # Use integer index based on sorted FY values
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

    # Filter to only columns with data to avoid Scaler issues with all-NaN columns
    if not quarters_with_data:
        logger.warning("No quarterly data available! Training without covariates.")
        quarterly_aligned = None
    else:
        quarterly_aligned = quarterly_aligned[quarters_with_data]
        logger.info(f"Using covariates: {quarters_with_data}")

    # Create TimeSeries with integer RangeIndex
    # Annual target series
    annual_values = annual_clean['Free Cash Flow'].values.reshape(-1, 1)
    annual_series = TimeSeries.from_values(
        values=annual_values,
        columns=['Free Cash Flow']
    )

    # Quarterly covariates series (only if we have data)
    quarterly_cov_series = None
    if quarterly_aligned is not None:
        quarterly_values = quarterly_aligned.values
        quarterly_cov_series = TimeSeries.from_values(
            values=quarterly_values,
            columns=quarters_with_data
        )

    # Split train/test - last N years go to test
    split_point = len(annual_series) - test_years

    if split_point < 3:
        raise ValueError(f"Not enough training data: only {split_point} years for training. Need at least 3.")

    train_annual = annual_series[:split_point]
    test_annual = annual_series[split_point:]

    train_quarterly_cov = None
    test_quarterly_cov = None
    if quarterly_cov_series is not None:
        train_quarterly_cov = quarterly_cov_series[:split_point]
        test_quarterly_cov = quarterly_cov_series[split_point:]

    # Log the actual FY split
    train_fys = [fy_mapping[i] for i in range(split_point)]
    test_fys = [fy_mapping[i] for i in range(split_point, len(annual_series))]
    logger.info(f"Train FYs: {train_fys}")
    logger.info(f"Test FYs: {test_fys}")
    logger.info(f"Train annual: {len(train_annual)} periods, Test annual: {len(test_annual)} periods")

    return train_annual, test_annual, train_quarterly_cov, test_quarterly_cov, fy_mapping


def train_annual_model(
        train_annual: TimeSeries,
        train_quarterly_cov: TimeSeries = None,
        test_annual: TimeSeries = None,
        test_quarterly_cov: TimeSeries = None
):
    """Train LightGBM model for annual FCF prediction using quarterly FCF as covariates.

    Note: We don't use validation during training to avoid issues with small datasets.
    The test set is used only for final evaluation after training.

    Args:
        train_annual: Training annual FCF data (target)
        train_quarterly_cov: Training quarterly FCF data (past covariates), can be None
        test_annual: Test annual FCF data (for final evaluation only)
        test_quarterly_cov: Test quarterly FCF data (for final evaluation only)

    Returns:
        model: Trained LightGBM model
        predictions: Predictions on test set (if provided)
        annual_pipeline: Preprocessing pipeline for target
        cov_pipeline: Preprocessing pipeline for covariates (or None)
    """
    logger.info("Training annual FCF model...")
    logger.info(f"Training samples: {len(train_annual)}")

    use_covariates = train_quarterly_cov is not None
    if use_covariates:
        logger.info(f"Using {len(train_quarterly_cov.components)} covariate(s): {list(train_quarterly_cov.components)}")
    else:
        logger.info("No covariates available - training with target lags only")

    # Preprocessing pipeline for target
    annual_pipeline = Pipeline(
        transformers=[
            Scaler(),
        ],
        n_jobs=1,
    )

    # Transform target data
    train_transformed = annual_pipeline.fit_transform([train_annual])
    train_transformed = [t.astype("float32") for t in train_transformed]

    # Transform covariates if available
    cov_pipeline = None
    train_cov_transformed = None
    if use_covariates:
        cov_pipeline = Pipeline(
            transformers=[
                Scaler(),
            ],
            n_jobs=1,
        )
        train_cov_transformed = cov_pipeline.fit_transform([train_quarterly_cov])
        train_cov_transformed = [t.astype("float32") for t in train_cov_transformed]

    # Determine appropriate lags based on training data size
    # Need at least lags + output_chunk_length samples
    max_lags = max(1, len(train_annual) - 2)  # Leave room for output
    lags = min(3, max_lags)  # Use up to 3 years of lags

    logger.info(f"Using lags={lags} (max possible: {max_lags})")

    # Build model
    model_params = {
        "objective": "mae",
        "lags": lags,
        "output_chunk_length": 1,  # Predict 1 year ahead
        "verbose": -1,
        "random_state": 42,
    }

    if use_covariates:
        model_params["lags_past_covariates"] = [-1]  # Previous year's quarterly data

    model = LightGBMModel(**model_params)

    # Train
    fit_params = {
        "series": train_transformed,
    }
    if use_covariates:
        fit_params["past_covariates"] = train_cov_transformed

    model.fit(**fit_params)

    logger.info("Model training completed.")

    # Predict on test set if available
    predictions = None
    if test_annual is not None and len(test_annual) > 0:
        predict_params = {
            "n": len(test_annual),
            "series": train_transformed,
        }

        if use_covariates and test_quarterly_cov is not None:
            # Combine train and test covariates for prediction
            combined_cov = train_quarterly_cov.append(test_quarterly_cov)
            combined_cov_transformed = cov_pipeline.transform([combined_cov])
            predict_params["past_covariates"] = [t.astype("float32") for t in combined_cov_transformed]

        pred_transformed = model.predict(**predict_params)
        predictions = annual_pipeline.inverse_transform(pred_transformed)
        # inverse_transform returns a list, extract the single TimeSeries
        if isinstance(predictions, list):
            predictions = predictions[0]

    return model, predictions, annual_pipeline, cov_pipeline

def save_predictions(predictions: TimeSeries, actuals: TimeSeries, ticker: str, fy_mapping: dict, train_size: int):
    """Save predictions and actuals to CSV.

    Args:
        predictions: Predicted TimeSeries
        actuals: Actual TimeSeries
        ticker: Stock ticker
        fy_mapping: Dictionary mapping index to FY
        train_size: Size of training set (to offset test indices)
    """
    pred_values = predictions.values().flatten()
    actual_values = actuals.values().flatten()

    # Map test indices to FY values
    test_fys = [fy_mapping[train_size + i] for i in range(len(pred_values))]

    result_df = pd.DataFrame({
        'fy': test_fys,
        'actual': actual_values,
        'predicted': pred_values,
    })

    output_file = RESULT_DIR / f"{ticker}_fcf_annual_predictions_lightgbm.csv"
    result_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to: {output_file}")


def train_and_predict(ticker: str, test_years: int = 4):
    """Main pipeline for training and prediction.

    Args:
        ticker: Stock ticker to process
        test_years: Number of years to use for test set (default: 4)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Starting FCF forecasting pipeline for {ticker}")
    logger.info(f"{'=' * 80}\n")

    # 1. Load data
    quarterly_df, annual_df = load_fcf_data(ticker)

    # 2. Prepare TimeSeries with train/test split
    train_annual, test_annual, train_quarterly_cov, test_quarterly_cov, fy_mapping = prepare_time_series(
        quarterly_df, annual_df, test_years
    )

    # 3. Train model
    logger.info("\n--- Training Annual FCF Model ---")
    model, predictions, annual_pipeline, cov_pipeline = train_annual_model(
        train_annual,
        train_quarterly_cov,
        test_annual,
        test_quarterly_cov
    )

    # 4. Save predictions
    save_predictions(predictions, test_annual, ticker, fy_mapping, len(train_annual))

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Pipeline completed for {ticker}")
    logger.info(f"{'=' * 80}\n")


# --- Main Execution Block ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = input("Enter ticker symbol: ").strip().upper()

    try:
        train_and_predict(ticker, test_years=4)
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)