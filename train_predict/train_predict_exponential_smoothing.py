#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free Cash Flow Forecasting with Exponential Smoothing
Annual FCF prediction using Holt's Linear Trend method

This model uses exponential smoothing with trend component:
- Level smoothing (alpha parameter)
- Trend smoothing (beta parameter)
- Weighted average with exponentially decreasing weights for past observations
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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


def prepare_data(annual_df: pd.DataFrame, test_years: int = 4):
    """Prepare data for Exponential Smoothing.

    Args:
        annual_df: Annual FCF data
        test_years: Number of years to use for test set (default: 4)

    Returns:
        train_y: Training FCF values
        test_y: Test FCF values
        fy_mapping: Dictionary mapping index to actual FY values
    """
    logger.info("Preparing data for Exponential Smoothing...")

    # Create annual DataFrame
    annual_clean = annual_df[['fy', 'end', 'Free Cash Flow']].copy()
    annual_clean = annual_clean.sort_values('fy').reset_index(drop=True)
    fy_values = annual_clean['fy'].tolist()

    # Create mapping from index to FY
    fy_mapping = {i: fy for i, fy in enumerate(fy_values)}

    logger.info(f"FY mapping: {fy_mapping}")
    logger.info(f"Annual data: {len(annual_clean)} fiscal years")

    # Create target series
    y = annual_clean['Free Cash Flow'].values

    # Split train/test - last N years go to test
    split_point = len(y) - test_years

    if split_point < 3:
        raise ValueError(f"Not enough training data: only {split_point} years for training. Need at least 3.")

    train_y = y[:split_point]
    test_y = y[split_point:]

    # Log the actual FY split
    train_fys = [fy_mapping[i] for i in range(split_point)]
    test_fys = [fy_mapping[i] for i in range(split_point, len(y))]
    logger.info(f"Train FYs: {train_fys}")
    logger.info(f"Test FYs: {test_fys}")
    logger.info(f"Train samples: {len(train_y)}, Test samples: {len(test_y)}")

    return train_y, test_y, fy_mapping


def train_exponential_smoothing(
        train_y: np.ndarray,
        test_y: np.ndarray,
        trend: str = 'add',
        damped_trend: bool = False,
):
    """Generate rolling forecasts using Exponential Smoothing.

    Uses Holt's Linear Trend method (or Simple Exponential Smoothing if trend=None).
    For each test year, refit the model with all available data and predict one step ahead.

    Args:
        train_y: Training FCF values
        test_y: Test FCF values
        trend: Trend component ('add', 'mul', or None for simple smoothing)
        damped_trend: Whether to use damped trend (default: False)

    Returns:
        predictions: Predicted values for test set
        models: List of fitted models (one per prediction)
    """
    logger.info("Generating forecasts with Exponential Smoothing...")
    logger.info(f"Training samples: {len(train_y)}")
    logger.info(f"Trend: {trend}, Damped: {damped_trend}")

    predictions = []
    models = []

    # Rolling forecast: for each test year, refit and predict one step ahead
    for i in range(len(test_y)):
        logger.info(f"\n  Predicting year {i+1}/{len(test_y)}...")

        # Training data up to this point (includes previous test years)
        current_train = np.concatenate([train_y, test_y[:i]]) if i > 0 else train_y

        logger.info(f"    Using {len(current_train)} observations for training")

        try:
            # Fit Exponential Smoothing model
            # For annual data without seasonality, use trend='add' or None
            model = ExponentialSmoothing(
                current_train,
                trend=trend,
                damped_trend=damped_trend,
                seasonal=None,  # No seasonality for annual data
                initialization_method='estimated',
            )

            fitted_model = model.fit(optimized=True)

            # Log fitted parameters
            logger.info(f"    Smoothing level (alpha): {fitted_model.params['smoothing_level']:.4f}")
            if trend is not None:
                logger.info(f"    Smoothing trend (beta): {fitted_model.params['smoothing_trend']:.4f}")
                if damped_trend:
                    logger.info(f"    Damping trend (phi): {fitted_model.params['damping_trend']:.4f}")

            # Predict one step ahead
            pred = fitted_model.forecast(steps=1)[0]
            predictions.append(pred)
            models.append(fitted_model)

            logger.info(f"  ✓ Predicted: ${pred/1e9:.2f}B")

        except Exception as e:
            logger.error(f"  ⚠ Prediction failed for year {i+1}: {str(e)}")
            # Use last known value as fallback
            pred = current_train[-1]
            predictions.append(pred)
            models.append(None)
            logger.warning(f"    Using last value as fallback: ${pred/1e9:.2f}B")

    predictions = np.array(predictions)

    logger.info("\nForecast completed successfully")

    return predictions, models

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

    output_file = RESULT_DIR / f"{ticker}_fcf_annual_predictions_exponential_smoothing.csv"
    result_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to: {output_file}")


def train_and_predict(ticker: str, test_years: int = 4, trend: str = 'add', damped_trend: bool = False):
    """Main pipeline for training and prediction.

    Args:
        ticker: Stock ticker to process
        test_years: Number of years to use for test set (default: 4)
        trend: Trend component ('add', 'mul', or None) (default: 'add')
        damped_trend: Whether to use damped trend (default: False)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Starting FCF forecasting pipeline for {ticker} (Exponential Smoothing)")
    logger.info(f"{'=' * 80}\n")

    # 1. Load data
    quarterly_df, annual_df = load_fcf_data(ticker)

    # 2. Prepare data
    train_y, test_y, fy_mapping = prepare_data(annual_df, test_years)

    # 3. Generate forecasts
    logger.info("\n--- Forecasting Annual FCF with Exponential Smoothing ---")
    predictions, models = train_exponential_smoothing(train_y, test_y, trend=trend, damped_trend=damped_trend)

    # 4. Save predictions
    save_predictions(predictions, test_y, ticker, fy_mapping, len(train_y))

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Pipeline completed for {ticker}")
    logger.info(f"{'=' * 80}\n")

    return predictions, models


# --- Main Execution Block ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = input("Enter ticker symbol: ").strip().upper()

    # Optional: specify trend type
    trend = 'add'  # Options: 'add', 'mul', None
    damped_trend = False

    if len(sys.argv) > 2:
        trend_arg = sys.argv[2].lower()
        if trend_arg in ['add', 'mul', 'none']:
            trend = None if trend_arg == 'none' else trend_arg
            logger.info(f"Using custom trend parameter: {trend}")

    if len(sys.argv) > 3:
        damped_arg = sys.argv[3].lower()
        if damped_arg in ['true', '1', 'yes']:
            damped_trend = True
            logger.info(f"Using damped trend: {damped_trend}")

    try:
        train_and_predict(ticker, test_years=4, trend=trend, damped_trend=damped_trend)
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
