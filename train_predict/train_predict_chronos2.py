#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free Cash Flow Forecasting with Chronos-2
Annual FCF prediction using historical annual FCF context

This model uses Amazon's Chronos-2 (chronos-t5-large) pretrained model
for time series forecasting.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

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
    """Prepare data for Chronos-2 training.

    Args:
        annual_df: Annual FCF data
        test_years: Number of years to use for test set (default: 4)

    Returns:
        train_y: Training FCF values
        test_y: Test FCF values
        fy_mapping: Dictionary mapping index to actual FY values
    """
    logger.info("Preparing data for Chronos-2...")

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


def load_chronos_model():
    """Load Chronos-2 pretrained model.

    Returns:
        pipeline: ChronosPipeline for forecasting
    """
    logger.info("Loading Chronos-2 model...")
    logger.info("  ⚠ This will download ~1GB model on first run")

    # Determine best device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        logger.info("  - Using GPU with bfloat16")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
        dtype = torch.float32
        logger.info("  - Using Apple Silicon MPS with float32")
    else:
        device = "cpu"
        dtype = torch.float32
        logger.info("  - Using CPU with float32")

    model_name = "amazon/chronos-t5-large"

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    logger.info(f"  ✓ Model loaded: {model_name}")
    return pipeline


def train_chronos_model(
        pipeline: ChronosPipeline,
        train_y: np.ndarray,
        test_y: np.ndarray,
):
    """Generate forecasts using Chronos-2 model.

    Args:
        pipeline: ChronosPipeline for forecasting
        train_y: Training FCF values
        test_y: Test FCF values

    Returns:
        predictions: Predicted values for test set
    """
    logger.info("Generating forecasts with Chronos-2...")
    logger.info(f"Training samples: {len(train_y)}")

    # Determine context length (use all training data, up to model's max)
    max_context_length = min(512, len(train_y))  # Chronos max is 512
    context_length = max_context_length

    logger.info(f"Using context length: {context_length} years")

    # Prepare context tensor
    context = torch.tensor(
        train_y[-context_length:],
        dtype=torch.float32
    )

    # Generate forecast
    n_predict = len(test_y)
    logger.info(f"Forecasting {n_predict} periods...")

    try:
        forecast = pipeline.predict(
            context,
            prediction_length=n_predict,
            num_samples=100,  # Generate 100 samples for uncertainty quantification
        )

        # Get median prediction (most robust)
        forecast_samples = forecast[0].numpy()
        predictions = np.median(forecast_samples, axis=0)

        logger.info("Forecast completed successfully")

    except Exception as e:
        logger.error(f"Forecast failed: {str(e)}")
        raise

    return predictions

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

    output_file = RESULT_DIR / f"{ticker}_fcf_annual_predictions_chronos2.csv"
    result_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to: {output_file}")


def train_and_predict(ticker: str, test_years: int = 4):
    """Main pipeline for training and prediction.

    Args:
        ticker: Stock ticker to process
        test_years: Number of years to use for test set (default: 4)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Starting FCF forecasting pipeline for {ticker} (Chronos-2)")
    logger.info(f"{'=' * 80}\n")

    # 1. Load data
    quarterly_df, annual_df = load_fcf_data(ticker)

    # 2. Prepare data
    train_y, test_y, fy_mapping = prepare_data(annual_df, test_years)

    # 3. Load Chronos-2 model
    logger.info("\n--- Loading Chronos-2 Model ---")
    pipeline = load_chronos_model()

    # 4. Generate forecasts
    logger.info("\n--- Forecasting Annual FCF ---")
    predictions = train_chronos_model(pipeline, train_y, test_y)

    # 5. Save predictions
    save_predictions(predictions, test_y, ticker, fy_mapping, len(train_y))

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Pipeline completed for {ticker}")
    logger.info(f"{'=' * 80}\n")

    return pipeline, predictions


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
