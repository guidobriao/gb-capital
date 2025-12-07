#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Free Cash Flow Forecasting with LightGBM
Quarterly FCF prediction using Annual FCF as covariate
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import Scaler
from darts.models import LightGBMModel
from lightgbm.callback import early_stopping, log_evaluation

# Setup logging
logging.basicConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "result"

# Ensure directories exist
RESULT_DIR.mkdir(exist_ok=True)


def load_fcf_data(ticker: str):
    """Load FCF data for a given ticker.
    
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
    
    # Sort by date
    quarterly_df = quarterly_df.sort_values('end').reset_index(drop=True)
    annual_df = annual_df.sort_values('end').reset_index(drop=True)
    
    logger.info(f"Loaded {len(quarterly_df)} quarterly records and {len(annual_df)} annual records")
    
    return quarterly_df, annual_df


def impute_missing_quarters(quarterly_df: pd.DataFrame, annual_df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing quarterly FCF values using annual FCF constraint.
    
    For each fiscal year, if some quarters are missing:
    - Calculate sum of available quarters
    - Distribute remaining FCF uniformly across missing quarters
    
    This is applied only on train data to teach the model the relationship.
    """
    logger.info("Imputing missing quarterly values using annual constraint...")
    
    # Create a copy to avoid modifying original
    df = quarterly_df.copy()
    
    # Group by fiscal year
    for fy in df['fy'].unique():
        if pd.isna(fy):
            continue
            
        # Get annual FCF for this fiscal year
        annual_fcf = annual_df[annual_df['fy'] == fy]['Free Cash Flow'].values
        if len(annual_fcf) == 0:
            continue
        annual_fcf = annual_fcf[0]
        
        # Get quarterly data for this fiscal year
        fy_mask = df['fy'] == fy
        fy_quarters = df[fy_mask].copy()
        
        # Count available and missing quarters
        available_fcf = fy_quarters['Free Cash Flow'].dropna()
        n_available = len(available_fcf)
        n_missing = 4 - n_available  # Assuming 4 quarters per year
        
        if n_missing > 0 and n_available > 0:
            # Calculate remaining FCF to distribute
            sum_available = available_fcf.sum()
            remaining_fcf = annual_fcf - sum_available
            fcf_per_missing = remaining_fcf / n_missing
            
            # Fill missing values
            df.loc[fy_mask & df['Free Cash Flow'].isna(), 'Free Cash Flow'] = fcf_per_missing
            
            logger.debug(f"FY {int(fy)}: Imputed {n_missing} quarters with {fcf_per_missing/1e9:.2f}B each")
    
    return df


def prepare_time_series(quarterly_df: pd.DataFrame, annual_df: pd.DataFrame, train_ratio: float = 0.8):
    """Prepare Darts TimeSeries for training.
    
    Returns:
        train_quarterly: Quarterly FCF TimeSeries for training
        test_quarterly: Quarterly FCF TimeSeries for testing
        train_annual: Annual FCF TimeSeries for training
        test_annual: Annual FCF TimeSeries for testing
    """
    logger.info("Preparing Darts TimeSeries...")
    
    # Create quarterly TimeSeries
    quarterly_series = TimeSeries.from_dataframe(
        df=quarterly_df,
        time_col='end',
        value_cols='Free Cash Flow',
        fill_missing_dates=False,
        freq='QS'  # Quarter Start
    )
    
    # Create annual TimeSeries
    annual_series = TimeSeries.from_dataframe(
        df=annual_df,
        time_col='end',
        value_cols='Free Cash Flow',
        fill_missing_dates=False,
        freq='YS'  # Year Start
    )
    
    # Split train/test (80/20)
    quarterly_split_point = int(len(quarterly_series) * train_ratio)
    annual_split_point = int(len(annual_series) * train_ratio)
    
    train_quarterly = quarterly_series[:quarterly_split_point]
    test_quarterly = quarterly_series[quarterly_split_point:]
    
    train_annual = annual_series[:annual_split_point]
    test_annual = annual_series[annual_split_point:]
    
    logger.info(f"Train quarterly: {len(train_quarterly)} periods, Test quarterly: {len(test_quarterly)} periods")
    logger.info(f"Train annual: {len(train_annual)} periods, Test annual: {len(test_annual)} periods")
    
    return train_quarterly, test_quarterly, train_annual, test_annual


def train_annual_model(train_annual: TimeSeries, val_annual: TimeSeries = None):
    """Train LightGBM model for annual FCF prediction.
    
    Args:
        train_annual: Training annual FCF data
        val_annual: Validation annual FCF data (optional)
    
    Returns:
        model: Trained LightGBM model
        predictions: Predictions on validation set (if provided)
    """
    logger.info("Training annual FCF model...")
    
    # Preprocessing pipeline
    annual_pipeline = Pipeline(
        transformers=[
            Scaler(),
        ],
        n_jobs=1,
    )
    
    train_transformed = annual_pipeline.fit_transform([train_annual])
    train_transformed = [t.astype("float32") for t in train_transformed]
    
    val_transformed = None
    if val_annual is not None and len(val_annual) > 0:
        val_transformed = annual_pipeline.transform([val_annual])
        val_transformed = [t.astype("float32") for t in val_transformed]
    
    # Build model
    encoders = {"datetime_attribute": {"future": ["month", "year"]}}
    model = LightGBMModel(
        objective="mae",
        lags=5,  # Use last 5 years
        output_chunk_length=1,  # Predict 1 year ahead
        add_encoders=encoders,
        verbose=-1,
        random_state=42,
    )
    
    # Train
    val_available = val_transformed is not None and len(val_transformed[0]) >= model.output_chunk_length
    
    model.fit(
        series=train_transformed,
        val_series=val_transformed if val_available else None,
        callbacks=[
            log_evaluation(period=10),
            early_stopping(stopping_rounds=10),
        ] if val_available else None,
    )
    
    logger.info("Annual model training completed.")
    
    # Predict on validation if available
    predictions = None
    if val_annual is not None:
        pred_transformed = model.predict(
            n=len(val_annual),
            series=train_transformed,
        )
        predictions = annual_pipeline.inverse_transform(pred_transformed)
    
    return model, predictions, annual_pipeline


def train_quarterly_model(
    train_quarterly: TimeSeries,
    train_annual: TimeSeries,
    val_quarterly: TimeSeries = None,
    annual_forecast: TimeSeries = None
):
    """Train LightGBM model for quarterly FCF prediction using annual FCF as covariate.
    
    Args:
        train_quarterly: Training quarterly FCF data
        train_annual: Training annual FCF data (used as future covariate)
        val_quarterly: Validation quarterly FCF data (optional)
        annual_forecast: Forecasted annual FCF for validation period (optional)
    
    Returns:
        model: Trained LightGBM model
        predictions: Predictions on validation set (if provided)
    """
    logger.info("Training quarterly FCF model with annual FCF as covariate...")
    
    # Preprocessing pipeline
    quarterly_pipeline = Pipeline(
        transformers=[
            Scaler(),
        ],
        n_jobs=1,
    )
    
    # Prepare annual FCF as future covariate
    # Resample annual to quarterly frequency by forward-filling
    train_annual_quarterly = train_annual.resample('QS')
    
    train_transformed = quarterly_pipeline.fit_transform([train_quarterly])
    train_transformed = [t.astype("float32") for t in train_transformed]
    
    future_cov_train = [train_annual_quarterly.astype("float32")]
    
    val_transformed = None
    future_cov_val = None
    
    if val_quarterly is not None and annual_forecast is not None:
        val_transformed = quarterly_pipeline.transform([val_quarterly])
        val_transformed = [t.astype("float32") for t in val_transformed]
        
        # Use forecasted annual FCF as future covariate for validation
        annual_forecast_quarterly = annual_forecast.resample('QS')
        future_cov_val = [annual_forecast_quarterly.astype("float32")]
    
    # Build model
    encoders = {"datetime_attribute": {"future": ["month", "quarter"]}}
    model = LightGBMModel(
        objective="mae",
        lags=12,  # Use last 12 quarters (3 years)
        lags_future_covariates=[0],  # Use current annual FCF
        output_chunk_length=4,  # Predict 4 quarters ahead (1 year)
        add_encoders=encoders,
        verbose=-1,
        random_state=42,
    )
    
    # Train
    val_available = val_transformed is not None and len(val_transformed[0]) >= model.output_chunk_length
    
    model.fit(
        series=train_transformed,
        future_covariates=future_cov_train,
        val_series=val_transformed if val_available else None,
        val_future_covariates=future_cov_val if val_available else None,
        callbacks=[
            log_evaluation(period=10),
            early_stopping(stopping_rounds=10),
        ] if val_available else None,
    )
    
    logger.info("Quarterly model training completed.")
    
    # Predict on validation if available
    predictions = None
    if val_quarterly is not None and annual_forecast is not None:
        # Combine train and forecast covariates for prediction
        combined_future_cov = train_annual_quarterly.append(annual_forecast_quarterly)
        
        pred_transformed = model.predict(
            n=len(val_quarterly),
            series=train_transformed,
            future_covariates=[combined_future_cov.astype("float32")],
        )
        predictions = quarterly_pipeline.inverse_transform(pred_transformed)
    
    return model, predictions, quarterly_pipeline


def calculate_metrics(predictions: TimeSeries, actuals: TimeSeries) -> dict:
    """Calculate evaluation metrics.
    
    Args:
        predictions: Predicted TimeSeries
        actuals: Actual TimeSeries
    
    Returns:
        metrics: Dictionary with MAE, RMSE, MAPE
    """
    pred_values = predictions.values().flatten()
    actual_values = actuals.values().flatten()
    
    # Remove NaN values from actuals (missing quarters in test set)
    mask = ~np.isnan(actual_values)
    pred_values = pred_values[mask]
    actual_values = actual_values[mask]
    
    if len(actual_values) == 0:
        logger.warning("No non-NaN values in actuals for metric calculation")
        return {}
    
    mae = np.mean(np.abs(pred_values - actual_values))
    rmse = np.sqrt(np.mean((pred_values - actual_values) ** 2))
    mape = np.mean(np.abs((actual_values - pred_values) / actual_values)) * 100
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
    }
    
    return metrics


def print_metrics_dict(metrics: dict, title: str = "Metrics"):
    """Print metrics in a formatted way."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{title}")
    logger.info(f"{'=' * 60}")
    for metric_name, value in metrics.items():
        if metric_name == 'MAPE':
            logger.info(f"{metric_name:>10}: {value:>12.2f}%")
        else:
            logger.info(f"{metric_name:>10}: ${value/1e9:>12.2f}B")
    logger.info(f"{'=' * 60}\n")


def save_predictions(predictions: TimeSeries, actuals: TimeSeries, ticker: str, data_type: str):
    """Save predictions and actuals to CSV.
    
    Args:
        predictions: Predicted TimeSeries
        actuals: Actual TimeSeries
        ticker: Stock ticker
        data_type: 'quarterly' or 'annual'
    """
    pred_df = predictions.pd_dataframe()
    actual_df = actuals.pd_dataframe()
    
    result_df = pd.DataFrame({
        'date': pred_df.index,
        'actual': actual_df.values.flatten(),
        'predicted': pred_df.values.flatten(),
    })
    
    output_file = RESULT_DIR / f"{ticker}_fcf_{data_type}_predictions.csv"
    result_df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to: {output_file}")


def train_and_predict(ticker: str, train_ratio: float = 0.8):
    """Main pipeline for training and prediction.
    
    Args:
        ticker: Stock ticker to process
        train_ratio: Ratio of data to use for training (default: 0.8)
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Starting FCF forecasting pipeline for {ticker}")
    logger.info(f"{'=' * 80}\n")
    
    # 1. Load data
    quarterly_df, annual_df = load_fcf_data(ticker)
    
    # 2. Impute missing quarters in training data using annual constraint
    quarterly_df_imputed = impute_missing_quarters(quarterly_df, annual_df)
    
    # 3. Prepare TimeSeries with train/test split
    train_quarterly, test_quarterly, train_annual, test_annual = prepare_time_series(
        quarterly_df_imputed, annual_df, train_ratio
    )
    
    # 4. Train annual model
    logger.info("\n--- Step 1: Training Annual FCF Model ---")
    annual_model, annual_val_pred, annual_pipeline = train_annual_model(
        train_annual, test_annual
    )
    
    # Evaluate annual model
    if annual_val_pred is not None:
        annual_metrics = calculate_metrics(annual_val_pred, test_annual)
        print_metrics_dict(annual_metrics, "Annual FCF Model Performance")
        save_predictions(annual_val_pred, test_annual, ticker, 'annual')
    
    # 5. Train quarterly model using forecasted annual FCF
    logger.info("\n--- Step 2: Training Quarterly FCF Model ---")
    quarterly_model, quarterly_val_pred, quarterly_pipeline = train_quarterly_model(
        train_quarterly,
        train_annual,
        test_quarterly,
        annual_val_pred if annual_val_pred is not None else None
    )
    
    # Evaluate quarterly model
    if quarterly_val_pred is not None:
        quarterly_metrics = calculate_metrics(quarterly_val_pred, test_quarterly)
        print_metrics_dict(quarterly_metrics, "Quarterly FCF Model Performance")
        save_predictions(quarterly_val_pred, test_quarterly, ticker, 'quarterly')
    
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
        train_and_predict(ticker, train_ratio=0.8)
    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
