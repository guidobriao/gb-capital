#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Comparison Analysis
Compares multiple ARIMA models and ensembles using train/test split
Uses train_predict.py for model training and evaluation
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Import from src/
from src.config import settings, load_tickers
from src.train_predict import train_all_models_for_ticker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default models to compare (can be overridden)
DEFAULT_MODELS = [
    'arima_110',  # ARIMA(1,1,0)
    'arima_111',  # ARIMA(1,1,1)
    'arima_210',  # ARIMA(2,1,0)
    'arima_211',  # ARIMA(2,1,1)
    'ensemble_110_111',  # Ensemble average
    'ensemble_110_210',  # Ensemble average
    'ensemble_111_211',  # Ensemble average
    'ensemble_210_211',  # Ensemble average
]


def run_comparison_for_ticker(ticker: str,
                              models_to_compare: list = None,
                              test_years: int = 4) -> dict:
    """
    Run model comparison for a single ticker using train/test split.

    Args:
        ticker: Stock ticker
        models_to_compare: List of model names (default: all ARIMA + ensembles)
        test_years: Number of years to use for test set

    Returns:
        dict: Results for all models
    """
    logger.info("=" * 80)
    logger.info(f"MODEL COMPARISON ANALYSIS - {ticker}")
    logger.info("=" * 80)

    if models_to_compare is None:
        models_to_compare = DEFAULT_MODELS

    logger.info(f"Models to compare: {', '.join(models_to_compare)}")
    logger.info(f"Test period: {test_years} years")
    logger.info("")

    try:
        # Use train_predict.py to train all models with train/test split
        results = train_all_models_for_ticker(
            ticker=ticker,
            test_years=test_years,
            data_dir=settings.DATA_DIR,
            result_dir=settings.RESULT_DIR
        )

        return results

    except Exception as e:
        logger.error(f"Error in model comparison for {ticker}: {str(e)}")
        return {'error': str(e)}


def create_comparison_summary(ticker: str, results: dict) -> pd.DataFrame:
    """
    Create summary DataFrame comparing all models.

    Args:
        ticker: Stock ticker
        results: Results dict from run_comparison_for_ticker

    Returns:
        DataFrame with comparison metrics
    """
    summary_data = []

    # Check if there's a top-level error (complete failure)
    if 'error' in results and len(results) == 1:
        # Complete failure - no models were trained
        summary_data.append({
            'Ticker': ticker,
            'Model': 'ALL',
            'Status': 'ERROR',
            'MAE ($B)': np.nan,
            'RMSE ($B)': np.nan,
            'MAPE (%)': np.nan,
            'sMAPE (%)': np.nan,
            'Error': results['error']
        })
    else:
        # Process individual model results
        for model_name, result in results.items():
            if isinstance(result, dict) and 'error' in result:
                summary_data.append({
                    'Ticker': ticker,
                    'Model': model_name,
                    'Status': 'ERROR',
                    'MAE ($B)': np.nan,
                    'RMSE ($B)': np.nan,
                    'MAPE (%)': np.nan,
                    'sMAPE (%)': np.nan,
                    'Error': result['error']
                })
            elif isinstance(result, dict) and 'metrics' in result:
                metrics = result['metrics']
                summary_data.append({
                    'Ticker': ticker,
                    'Model': model_name,
                    'Status': 'SUCCESS',
                    'MAE ($B)': metrics['MAE'] / 1e9,
                    'RMSE ($B)': metrics['RMSE'] / 1e9,
                    'MAPE (%)': metrics['MAPE'],
                    'sMAPE (%)': metrics['sMAPE'],
                    'Error': ''
                })

    df = pd.DataFrame(summary_data)

    # Sort by MAPE for successful models
    df_success = df[df['Status'] == 'SUCCESS'].sort_values('MAPE (%)')
    df_error = df[df['Status'] == 'ERROR']

    return pd.concat([df_success, df_error], ignore_index=True)


def save_comparison_results(ticker: str, summary_df: pd.DataFrame):
    """Save comparison results to CSV."""
    settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = settings.RESULT_DIR / f"{ticker}_model_comparison.csv"
    summary_df.to_csv(output_file, index=False)

    logger.info(f"\n✓ Comparison results saved to: {output_file}")


def print_comparison_table(summary_df: pd.DataFrame):
    """Print formatted comparison table."""
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON RESULTS")
    logger.info("=" * 80)

    # Successful models
    df_success = summary_df[summary_df['Status'] == 'SUCCESS']

    if not df_success.empty:
        logger.info("\n✓ Successful Models (sorted by MAPE):\n")

        # Format for display
        display_df = df_success[['Model', 'MAE ($B)', 'RMSE ($B)', 'MAPE (%)', 'sMAPE (%)']].copy()

        # Format numeric columns
        display_df['MAE ($B)'] = display_df['MAE ($B)'].apply(lambda x: f"{x:.2f}")
        display_df['RMSE ($B)'] = display_df['RMSE ($B)'].apply(lambda x: f"{x:.2f}")
        display_df['MAPE (%)'] = display_df['MAPE (%)'].apply(lambda x: f"{x:.1f}")
        display_df['sMAPE (%)'] = display_df['sMAPE (%)'].apply(lambda x: f"{x:.1f}")

        print(display_df.to_string(index=False))

        # Highlight best model
        best_model = df_success.iloc[0]
        logger.info(f"\n🏆 Best Model: {best_model['Model']} (MAPE: {best_model['MAPE (%)']:.1f}%)")

    # Failed models
    df_error = summary_df[summary_df['Status'] == 'ERROR']

    if not df_error.empty:
        logger.info(f"\n✗ Failed Models: {len(df_error)}")
        for _, row in df_error.iterrows():
            logger.info(f"  {row['Model']}: {row['Error']}")

    logger.info("\n" + "=" * 80)


def run_comparison_all_tickers(models_to_compare: list = None, test_years: int = 4):
    """
    Run model comparison for all tickers in tickers.txt.

    Args:
        models_to_compare: List of model names (default: all ARIMA + ensembles)
        test_years: Number of years to use for test set
    """
    companies = load_tickers()

    if not companies:
        logger.error("No tickers found in tickers.txt")
        return

    tickers = [c['TICKER'] for c in companies]

    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON ANALYSIS - ALL TICKERS")
    logger.info("=" * 80)
    logger.info(f"Processing {len(tickers)} tickers: {', '.join(tickers)}")
    logger.info(f"Test period: {test_years} years")
    logger.info("=" * 80 + "\n")

    all_summaries = []

    for ticker in tickers:
        # Run comparison for ticker
        results = run_comparison_for_ticker(ticker, models_to_compare, test_years)

        # Create summary
        summary_df = create_comparison_summary(ticker, results)
        all_summaries.append(summary_df)

        # Save results
        save_comparison_results(ticker, summary_df)

        # Print comparison table
        print_comparison_table(summary_df)

    # Combined summary for all tickers
    if all_summaries:
        logger.info("\n" + "=" * 80)
        logger.info("OVERALL SUMMARY - ALL TICKERS")
        logger.info("=" * 80)

        combined_df = pd.concat(all_summaries, ignore_index=True)

        # Save combined results
        combined_file = settings.RESULT_DIR / "all_tickers_model_comparison.csv"
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"\n✓ Combined results saved to: {combined_file}")

        # Average performance by model (across all tickers)
        df_success = combined_df[combined_df['Status'] == 'SUCCESS']

        if not df_success.empty:
            avg_by_model = df_success.groupby('Model').agg({
                'MAE ($B)': 'mean',
                'RMSE ($B)': 'mean',
                'MAPE (%)': 'mean',
                'sMAPE (%)': 'mean'
            }).round(2)

            avg_by_model = avg_by_model.sort_values('MAPE (%)')

            logger.info("\nAverage Performance by Model (across all tickers):\n")
            print(avg_by_model.to_string())

        logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        # Single ticker mode
        ticker = sys.argv[1].upper()

        # Optional: specify models
        models = DEFAULT_MODELS
        if len(sys.argv) > 2:
            models = sys.argv[2].split(',')

        # Optional: specify test years
        test_years = 4
        if len(sys.argv) > 3:
            test_years = int(sys.argv[3])

        results = run_comparison_for_ticker(ticker, models, test_years)
        summary_df = create_comparison_summary(ticker, results)
        save_comparison_results(ticker, summary_df)
        print_comparison_table(summary_df)

    else:
        # All tickers mode
        run_comparison_all_tickers()