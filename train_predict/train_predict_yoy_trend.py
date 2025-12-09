#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free Cash Flow Forecasting with YoY Trend
Annual FCF prediction using Year-over-Year trend analysis

This model predicts based on:
1. Last year's value for the same fiscal year
2. Historical YoY growth trend (median)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Prepare data for YoY Trend forecasting.

    Args:
        annual_df: Annual FCF data
        test_years: Number of years to use for test set (default: 4)

    Returns:
        train_df: Training dataframe with fy and FCF
        test_df: Test dataframe with fy and FCF
        fy_mapping: Dictionary mapping index to actual FY values
    """
    logger.info("Preparing data for YoY Trend...")

    # Create annual DataFrame
    annual_clean = annual_df[['fy', 'end', 'Free Cash Flow']].copy()
    annual_clean = annual_clean.sort_values('fy').reset_index(drop=True)

    logger.info(f"Annual data: {len(annual_clean)} fiscal years")

    # Split train/test - last N years go to test
    split_point = len(annual_clean) - test_years

    if split_point < 2:
        raise ValueError(f"Not enough training data: only {split_point} years for training. Need at least 2.")

    train_df = annual_clean.iloc[:split_point].copy()
    test_df = annual_clean.iloc[split_point:].copy()

    # Create mapping from index to FY
    fy_mapping = {i: fy for i, fy in enumerate(annual_clean['fy'].tolist())}

    # Log the actual FY split
    logger.info(f"Train FYs: {train_df['fy'].tolist()}")
    logger.info(f"Test FYs: {test_df['fy'].tolist()}")
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    return train_df, test_df, fy_mapping


def yoy_trend_forecast(train_data: pd.DataFrame, target_fy: int):
    """
    YoY Trend: prediction based on last year's value and historical YoY trend.

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data with columns 'fy' and 'Free Cash Flow'
    target_fy : int
        Fiscal year to predict

    Returns:
    --------
    float : predicted value
    """
    # Find last year's value
    last_year_fy = target_fy - 1
    last_year_row = train_data[train_data['fy'] == last_year_fy]

    if len(last_year_row) == 0:
        # If no last year value, use mean
        logger.warning(f"No data for FY{last_year_fy}, using mean")
        return train_data['Free Cash Flow'].mean()

    last_year_value = last_year_row['Free Cash Flow'].values[0]

    # Calculate YoY trend from all available historical data
    yoy_changes = []

    for i in range(1, len(train_data)):
        current_fy = train_data.iloc[i]['fy']
        prev_fy = train_data.iloc[i-1]['fy']

        # Only consider consecutive fiscal years
        if current_fy == prev_fy + 1:
            current_fcf = train_data.iloc[i]['Free Cash Flow']
            prev_fcf = train_data.iloc[i-1]['Free Cash Flow']

            if prev_fcf > 0:  # Avoid division by zero
                yoy_change = (current_fcf - prev_fcf) / prev_fcf
                yoy_changes.append(yoy_change)

    if len(yoy_changes) > 0:
        # Use median of YoY trend (more robust than mean)
        avg_yoy_trend = np.median(yoy_changes)
        logger.info(f"    Historical YoY changes: {[f'{x:.2%}' for x in yoy_changes]}")
        logger.info(f"    Median YoY trend: {avg_yoy_trend:.2%}")
    else:
        avg_yoy_trend = 0.0
        logger.warning(f"    No YoY trend data available, using 0% growth")

    # Apply trend to last year's value
    prediction = last_year_value * (1 + avg_yoy_trend)

    logger.info(f"    FY{last_year_fy} value: ${last_year_value/1e9:.2f}B")
    logger.info(f"    Predicted FY{target_fy}: ${prediction/1e9:.2f}B ({avg_yoy_trend:+.2%} YoY)")

    return float(prediction)


def train_yoy_trend(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
):
    """Generate rolling forecasts using YoY Trend.

    For each test year, use all previous data to make a prediction
    based on last year's value and historical YoY trend.

    Args:
        train_df: Training dataframe with fy and FCF
        test_df: Test dataframe with fy and FCF

    Returns:
        predictions: Predicted values for test set
    """
    logger.info("Generating forecasts with YoY Trend...")
    logger.info(f"Training samples: {len(train_df)}")

    predictions = []
    test_fys = test_df['fy'].tolist()

    # Rolling forecast: for each test year, use all data up to that point
    for i, target_fy in enumerate(test_fys):
        logger.info(f"\n  Predicting FY{target_fy} ({i+1}/{len(test_fys)})...")

        # Training data up to this point (includes previous test years)
        if i > 0:
            # Include previous test observations
            prev_test = test_df.iloc[:i]
            current_train = pd.concat([train_df, prev_test], ignore_index=True)
        else:
            current_train = train_df.copy()

        # Make prediction
        pred = yoy_trend_forecast(current_train, target_fy)
        predictions.append(pred)

        logger.info(f"  ✓ Predicted FY{target_fy}: ${pred/1e9:.2f}B")

    predictions = np.array(predictions)

    logger.info("\nForecast completed successfully")

    return predictions

def save_predictions(predictions: np.ndarray, test_df: pd.DataFrame, ticker: str):
    """Save predictions and actuals to CSV.

    Args:
        predictions: Predicted values
        test_df: Test dataframe with actual values
        ticker: Stock ticker
    """
    result_df = pd.DataFrame({
        'fy': test_df['fy'].values,
        'actual': test_df['Free Cash Flow'].values,
        'predicted': predictions,
    })

    output_file = RESULT_DIR / f"{ticker}_fcf_annual_predictions_yoy_trend.csv"
    result_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to: {output_file}")


def train_and_predict(ticker: str, test_years: int = 4):
    """Main pipeline for training and prediction.

    Args:
        ticker: Stock ticker to process
        test_years: Number of years to use for test set (default: 4)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Starting FCF forecasting pipeline for {ticker} (YoY Trend)")
    logger.info(f"{'=' * 80}\n")

    # 1. Load data
    quarterly_df, annual_df = load_fcf_data(ticker)

    # 2. Prepare data
    train_df, test_df, fy_mapping = prepare_data(annual_df, test_years)

    # 3. Generate forecasts
    logger.info("\n--- Forecasting Annual FCF with YoY Trend ---")
    predictions = train_yoy_trend(train_df, test_df)

    # 4. Save predictions
    save_predictions(predictions, test_df, ticker)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Pipeline completed for {ticker}")
    logger.info(f"{'=' * 80}\n")

    return predictions


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
