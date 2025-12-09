#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free Cash Flow Forecasting with Moving Average
Annual FCF prediction using quarterly moving averages

This model predicts annual FCF as the sum of quarterly predictions:
- Each quarter: average of last N observations of the same quarter (default N=3)
- If insufficient quarterly data: use average of last N annual observations
- Fallback: use all available observations
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


def prepare_data(quarterly_df: pd.DataFrame, annual_df: pd.DataFrame, test_years: int = 4):
    """Prepare data for Moving Average forecasting.

    Args:
        quarterly_df: Quarterly FCF data
        annual_df: Annual FCF data
        test_years: Number of years to use for test set (default: 4)

    Returns:
        train_quarterly: Training quarterly data
        test_quarterly: Test quarterly data
        train_annual: Training annual data
        test_annual: Test annual data
        fy_mapping: Dictionary mapping index to actual FY values
    """
    logger.info("Preparing data for Moving Average...")

    # Annual data
    annual_clean = annual_df[['fy', 'end', 'Free Cash Flow']].copy()
    annual_clean = annual_clean.sort_values('fy').reset_index(drop=True)

    # Quarterly data
    quarterly_clean = quarterly_df[['fy', 'fp', 'end', 'Free Cash Flow']].copy()
    quarterly_clean = quarterly_clean.sort_values(['fy', 'fp']).reset_index(drop=True)

    logger.info(f"Annual data: {len(annual_clean)} fiscal years")
    logger.info(f"Quarterly data: {len(quarterly_clean)} records")

    # Split train/test - last N years go to test
    split_fy = annual_clean['fy'].max() - test_years + 1

    train_quarterly = quarterly_clean[quarterly_clean['fy'] < split_fy].copy()
    test_quarterly = quarterly_clean[quarterly_clean['fy'] >= split_fy].copy()

    train_annual = annual_clean[annual_clean['fy'] < split_fy].copy()
    test_annual = annual_clean[annual_clean['fy'] >= split_fy].copy()

    # Create mapping from index to FY
    fy_mapping = {i: fy for i, fy in enumerate(annual_clean['fy'].tolist())}

    # Log the actual FY split
    logger.info(f"Train FYs: {sorted(train_annual['fy'].unique().tolist())}")
    logger.info(f"Test FYs: {sorted(test_annual['fy'].unique().tolist())}")
    logger.info(f"Train samples: {len(train_annual)} annual, {len(train_quarterly)} quarterly")
    logger.info(f"Test samples: {len(test_annual)} annual, {len(test_quarterly)} quarterly")

    return train_quarterly, test_quarterly, train_annual, test_annual, fy_mapping


def predict_quarter_moving_avg(train_quarterly: pd.DataFrame, target_fy: int, target_quarter: str, n_last: int = 3):
    """
    Predict a specific quarter using moving average of last N observations of the same quarter.

    Parameters:
    -----------
    train_quarterly : pd.DataFrame
        Training quarterly data with columns 'fy', 'fp', 'Free Cash Flow'
    target_fy : int
        Target fiscal year
    target_quarter : str
        Target quarter (Q1, Q2, Q3, Q4)
    n_last : int
        Number of last observations to average (default: 3)

    Returns:
    --------
    float : predicted value, or None if insufficient data
    """
    # Filter for the same quarter
    same_quarter = train_quarterly[train_quarterly['fp'] == target_quarter].copy()
    same_quarter = same_quarter[same_quarter['fy'] < target_fy]

    if len(same_quarter) >= n_last:
        # Average of last N observations of the same quarter
        last_n = same_quarter.tail(n_last)['Free Cash Flow'].values
        prediction = np.mean(last_n)
        logger.info(f"      {target_quarter}: Using last {n_last} quarters → ${prediction/1e9:.2f}B")
        return float(prediction)
    elif len(same_quarter) > 0:
        # If not enough observations, use all available
        prediction = same_quarter['Free Cash Flow'].mean()
        logger.info(f"      {target_quarter}: Using all {len(same_quarter)} available quarters → ${prediction/1e9:.2f}B")
        return float(prediction)
    else:
        # No data available
        logger.info(f"      {target_quarter}: No quarterly data available")
        return None


def predict_annual_moving_avg(train_annual: pd.DataFrame, target_fy: int, n_last: int = 3):
    """
    Predict annual FCF using moving average of last N annual observations.

    Parameters:
    -----------
    train_annual : pd.DataFrame
        Training annual data with columns 'fy', 'Free Cash Flow'
    target_fy : int
        Target fiscal year
    n_last : int
        Number of last observations to average (default: 3)

    Returns:
    --------
    float : predicted value
    """
    # Filter for years before target
    historical = train_annual[train_annual['fy'] < target_fy].copy()

    if len(historical) >= n_last:
        # Average of last N observations
        last_n = historical.tail(n_last)['Free Cash Flow'].values
        prediction = np.mean(last_n)
        logger.info(f"      Using last {n_last} annual observations → ${prediction/1e9:.2f}B")
    elif len(historical) > 0:
        # If not enough observations, use all available
        prediction = historical['Free Cash Flow'].mean()
        logger.info(f"      Using all {len(historical)} available annual observations → ${prediction/1e9:.2f}B")
    else:
        # Should not happen, but fallback to 0
        prediction = 0.0
        logger.warning(f"      No historical data available, using 0")

    return float(prediction)


def has_sufficient_quarter_coverage(
    train_quarterly: pd.DataFrame, target_fy: int, n_last: int = 3
) -> bool:
    """
    Controlla se, negli ultimi n_last anni prima di target_fy,
    ogni anno ha almeno 3 quarter disponibili.

    - Se un anno ha 0, 1 o 2 quarter → anno "monco" → ritorna False
    - Se tutti gli anni (fino a n_last) hanno >=3 quarter → True
    """
    historical = train_quarterly[train_quarterly['fy'] < target_fy].copy()
    years = sorted(historical['fy'].unique())

    if not years:
        return False

    # Prendo gli ultimi n_last anni storici disponibili
    years_to_check = years[-min(n_last, len(years)):]

    for fy in years_to_check:
        year_q = historical[historical['fy'] == fy]
        n_quarters = year_q['fp'].nunique()
        if n_quarters < 3:
            logger.info(
                f"    FY{fy} ha solo {n_quarters} quarter disponibili (<3) → anno monco, fallback annuale"
            )
            return False

    return True

def train_moving_average(
        train_quarterly: pd.DataFrame,
        test_quarterly: pd.DataFrame,
        train_annual: pd.DataFrame,
        test_annual: pd.DataFrame,
        n_last: int = 3,
):
    """Generate forecasts using Moving Average.

    Strategy:
    1. Try to predict each quarter using quarterly moving average
    2. Sum the 4 quarters to get annual prediction
    3. If quarterly data is insufficient, use annual moving average
    """

    logger.info("Generating forecasts with Moving Average...")
    logger.info(f"Using n_last={n_last} for moving average")

    predictions = []
    test_fys = sorted(test_annual['fy'].unique())

    for target_fy in test_fys:
        logger.info(f"\n  Predicting FY{target_fy}...")

        # 🔍 PRIMO CHECK: copertura trimestrale negli ultimi n_last anni
        if not has_sufficient_quarter_coverage(train_quarterly, target_fy, n_last):
            logger.info("    ⚠ Copertura trimestrale insufficiente negli ultimi anni")
            logger.info("    → Uso subito fallback annuale (moving average annuale)")
            annual_pred = predict_annual_moving_avg(train_annual, target_fy, n_last)
            predictions.append(annual_pred)
            logger.info(f"  ✓ Final prediction FY{target_fy}: ${annual_pred/1e9:.2f}B")
            continue

        # Se la copertura è ok, procedo con la logica trimestrale ESATTAMENTE come prima
        quarterly_predictions = {}
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            pred = predict_quarter_moving_avg(train_quarterly, target_fy, quarter, n_last)
            if pred is not None:
                quarterly_predictions[quarter] = pred

        # Check se ho abbastanza quarter previsti
        if len(quarterly_predictions) >= 3:
            # Quarterly approach: somma dei quarter disponibili
            annual_pred = sum(quarterly_predictions.values())
            logger.info(
                f"    ✓ Quarterly-based prediction: {len(quarterly_predictions)}/4 quarters available"
            )
            logger.info(f"    Sum of quarters: ${annual_pred/1e9:.2f}B")

            # Se manca 1 quarter, scala a 4 come prima
            if len(quarterly_predictions) < 4:
                scaling_factor = 4 / len(quarterly_predictions)
                annual_pred_scaled = annual_pred * scaling_factor
                logger.info(
                    f"    Scaled to 4 quarters: ${annual_pred_scaled/1e9:.2f}B "
                    f"(factor: {scaling_factor:.2f})"
                )
                annual_pred = annual_pred_scaled
        else:
            # Fallback annuale se, nonostante la copertura storica, non riesco a stimare ≥3 quarter
            logger.info(
                f"    ⚠ Insufficient quarterly data ({len(quarterly_predictions)}/4 quarters)"
            )
            logger.info("    Using annual moving average fallback:")
            annual_pred = predict_annual_moving_avg(train_annual, target_fy, n_last)

        predictions.append(annual_pred)
        logger.info(f"  ✓ Final prediction FY{target_fy}: ${annual_pred/1e9:.2f}B")

    predictions = np.array(predictions)

    logger.info("\nForecast completed successfully")

    return predictions

def save_predictions(predictions: np.ndarray, test_annual: pd.DataFrame, ticker: str):
    """Save predictions and actuals to CSV.

    Args:
        predictions: Predicted values
        test_annual: Test annual dataframe
        ticker: Stock ticker
    """
    result_df = pd.DataFrame({
        'fy': sorted(test_annual['fy'].unique()),
        'actual': test_annual['Free Cash Flow'].values,
        'predicted': predictions,
    })

    output_file = RESULT_DIR / f"{ticker}_fcf_annual_predictions_moving_average.csv"
    result_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to: {output_file}")


def train_and_predict(ticker: str, test_years: int = 4, n_last: int = 3):
    """Main pipeline for training and prediction.

    Args:
        ticker: Stock ticker to process
        test_years: Number of years to use for test set (default: 4)
        n_last: Number of last observations to average (default: 3)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Starting FCF forecasting pipeline for {ticker} (Moving Average)")
    logger.info(f"{'=' * 80}\n")

    # 1. Load data
    quarterly_df, annual_df = load_fcf_data(ticker)

    # 2. Prepare data
    train_quarterly, test_quarterly, train_annual, test_annual, fy_mapping = prepare_data(
        quarterly_df, annual_df, test_years
    )

    # 3. Generate forecasts
    logger.info("\n--- Forecasting Annual FCF with Moving Average ---")
    predictions = train_moving_average(
        train_quarterly, test_quarterly, train_annual, test_annual, n_last=n_last
    )

    # 4. Save predictions
    save_predictions(predictions, test_annual, ticker)

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

    # Optional: specify n_last parameter
    n_last = 3
    if len(sys.argv) > 2:
        try:
            n_last = int(sys.argv[2])
            logger.info(f"Using custom n_last parameter: {n_last}")
        except ValueError:
            logger.warning(f"Invalid n_last parameter, using default: {n_last}")

    try:
        train_and_predict(ticker, test_years=4, n_last=n_last)
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
