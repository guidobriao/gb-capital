"""
FCF Forecasting System - 4 ARIMA Models + 4 Ensembles
Trains ARIMA(1,1,0), ARIMA(1,1,1), ARIMA(2,1,0), ARIMA(2,1,1)
Then creates 4 ensemble averages from pairs
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_fcf_data(name: str, data_dir: Path) -> tuple:
    """
    Load annual and quarterly FCF data from CSV (already processed by fcf_extractor.py)
    Args:
        name: Company name (not ticker)
    """

    # Load quarterly data (optional)
    quarterly_file = data_dir / f"{name}_fcf_10Q.csv"
    if quarterly_file.exists():
        df_q = pd.read_csv(quarterly_file)
        df_q['end'] = pd.to_datetime(df_q['end'])
    else:
        logger.warning(f"Quarterly file not found: {quarterly_file}")
        df_q = pd.DataFrame()

    # Load annual data (required)
    annual_file = data_dir / f"{name}_fcf_10K.csv"
    if not annual_file.exists():
        raise FileNotFoundError(f"Annual file not found: {annual_file}")

    df_a = pd.read_csv(annual_file)
    df_a['end'] = pd.to_datetime(df_a['end'])

    # Clean and deduplicate
    for df in [df_q, df_a]:
        if not df.empty:
            df.dropna(subset=['fy', 'Free Cash Flow'], inplace=True)
            df['fy'] = df['fy'].astype(int)
            df.sort_values(['fy', 'end'], inplace=True)
            df.drop_duplicates(subset=['fy'], keep='last', inplace=True)

    logger.info(f"Loaded {len(df_q)} quarterly records and {len(df_a)} annual records")
    logger.info(f"Annual FY range: {df_a['fy'].min()} - {df_a['fy'].max()}")

    return df_q, df_a


def prepare_quarterly_covariates(df_quarterly: pd.DataFrame, annual_fys: np.ndarray, 
                                 lag: int = -1) -> pd.DataFrame:
    """
    Prepare quarterly covariates aligned to annual fiscal years.
    Fixed: Handles NaN from lag, fills with forward fill then mean.
    """
    logger.info("Preparing quarterly covariates...")
    
    df_q = df_quarterly.copy()
    df_q = df_q.sort_values('end')
    
    # Create quarter mapping
    df_q['quarter'] = df_q['fp'].str.upper()
    df_q = df_q[df_q['quarter'].isin(['Q1', 'Q2', 'Q3'])].copy()
    
    if df_q.empty:
        logger.warning("No quarterly data available")
        return pd.DataFrame()
    
    # Pivot to get Q1, Q2, Q3 columns
    pivot = df_q.pivot_table(
        index='fy',
        columns='quarter',
        values='Free Cash Flow',
        aggfunc='last'
    )
    
    # Align to annual fiscal years
    cov_df = pd.DataFrame(index=annual_fys)
    
    for q in ['Q1', 'Q2', 'Q3']:
        if q in pivot.columns:
            cov_df[q] = pivot[q].reindex(annual_fys)
    
    # Apply lag
    if lag != 0:
        cov_df = cov_df.shift(-lag)
        logger.info(f"Applied lag of {lag} to quarterly covariates")
    
    # Handle NaN values
    cov_df = cov_df.ffill().bfill()
    
    for col in cov_df.columns:
        if cov_df[col].isna().any():
            col_mean = cov_df[col].mean()
            if not np.isnan(col_mean):
                cov_df[col] = cov_df[col].fillna(col_mean)
            else:
                cov_df[col] = cov_df[col].fillna(0)
    
    # Drop columns that are all NaN or all zeros
    cov_df = cov_df.loc[:, (cov_df != 0).any(axis=0)]
    
    if not cov_df.empty:
        logger.info(f"Created quarterly covariates: {list(cov_df.columns)}")
    else:
        logger.warning("No valid quarterly covariates after cleaning")
    
    return cov_df


def split_data(df_annual: pd.DataFrame, cov_df: pd.DataFrame, 
               min_train_years: int = 3, target_test_years: int = 4) -> dict:
    """
    Intelligent train/test split.
    Fixed: Adapts test size based on available data.
    """
    df = df_annual.sort_values('fy').copy()
    
    total_years = len(df)
    
    # Adaptive test size
    if total_years < min_train_years + 1:
        raise ValueError(f"Not enough data: {total_years} years. Need at least {min_train_years + 1}")
    
    # Calculate test years: min(target, available - min_train)
    max_possible_test = total_years - min_train_years
    test_years = min(target_test_years, max_possible_test)
    train_years = total_years - test_years
    
    # Split
    train_df = df.iloc[:train_years]
    test_df = df.iloc[train_years:]
    
    train_y = train_df['Free Cash Flow'].values
    test_y = test_df['Free Cash Flow'].values
    train_fys = train_df['fy'].values
    test_fys = test_df['fy'].values
    
    logger.info(f"Train FYs: {train_fys.tolist()}")
    logger.info(f"Test FYs: {test_fys.tolist()}")
    logger.info(f"Train samples: {len(train_y)}, Test samples: {len(test_y)}")
    
    # Prepare covariates if available
    train_cov = None
    test_cov = None
    
    if not cov_df.empty:
        # Filter covariates to match train/test years
        train_cov = cov_df.loc[cov_df.index.isin(train_fys)]
        test_cov = cov_df.loc[cov_df.index.isin(test_fys)]
        
        # Ensure proper alignment
        train_cov = train_cov.reindex(train_fys).fillna(0)
        test_cov = test_cov.reindex(test_fys).fillna(0)
    
    return {
        'train_y': train_y,
        'test_y': test_y,
        'train_fys': train_fys,
        'test_fys': test_fys,
        'train_cov': train_cov,
        'test_cov': test_cov,
    }


def train_statsforecast(model, data_split: dict) -> np.ndarray:
    """
    Train StatsForecast ARIMA model.
    """
    from statsforecast import StatsForecast
    
    train_y = data_split['train_y']
    test_y = data_split['test_y']
    train_fys = data_split['train_fys']
    test_fys = data_split['test_fys']
    
    # Create training DataFrame
    train_df = pd.DataFrame({
        'unique_id': ['FCF'] * len(train_y),
        'ds': pd.to_datetime([f'{fy}-12-31' for fy in train_fys]),
        'y': train_y,
    })
    
    horizon = len(test_y)
    
    # Create wrapper and fit
    sf = StatsForecast(models=[model], freq='Y', n_jobs=1)
    sf.fit(df=train_df)
    
    # Forecast
    forecast_df = sf.predict(h=horizon)
    
    model_class_name = model.__class__.__name__
    predictions = forecast_df[model_class_name].values
    
    return predictions


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate forecasting metrics."""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # MAPE
    mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
    
    # sMAPE
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    smape = np.mean(np.abs(actual - predicted) / np.where(denominator != 0, denominator, 1)) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'sMAPE': smape,
    }


def save_predictions(name: str, model_name: str, test_fys: np.ndarray,
                    actual: np.ndarray, predicted: np.ndarray, 
                    result_dir: Path) -> Path:
    """Save predictions to CSV using company name."""
    result_dir.mkdir(parents=True, exist_ok=True)
    
    output_df = pd.DataFrame({
        'fy': test_fys,
        'actual': actual,
        'predicted': predicted,
    })
    
    filename = result_dir / f"{name}_fcf_annual_predictions_{model_name}.csv"
    output_df.to_csv(filename, index=False)
    
    return filename


def train_and_predict(model_name: str, name: str, 
                     test_years: int = 4,
                     data_dir: Path = None,
                     result_dir: Path = None,
                     covariate_lag: int = -1,
                     min_train_years: int = 3) -> tuple:
    """
    Main training and prediction function for ARIMA models.
    Args:
        name: Company name (not ticker)
    """
    from src.config import settings
    from src.model import build_model

    if data_dir is None:
        data_dir = settings.DATA_DIR
    if result_dir is None:
        result_dir = settings.RESULT_DIR

    logger.info(f"Training {model_name.upper()} for {name}")

    try:
        # Load data (only once per company)
        if not hasattr(train_and_predict, 'data_cache') or train_and_predict.data_cache.get('name') != name:
            logger.info(f"Loading data for {name}...")
            df_quarterly, df_annual = load_fcf_data(name, data_dir)

            # Prepare covariates
            annual_fys = df_annual['fy'].values
            cov_df = prepare_quarterly_covariates(df_quarterly, annual_fys, lag=covariate_lag)

            # Split data
            data_split = split_data(df_annual, cov_df, min_train_years, test_years)

            # Cache for reuse
            train_and_predict.data_cache = {
                'name': name,
                'data_split': data_split
            }
        else:
            logger.info(f"Using cached data for {name}")
            data_split = train_and_predict.data_cache['data_split']

        # Build model
        model_obj = build_model(model_name)

        # Train
        logger.info(f"Training {model_name.upper()}...")
        predictions = train_statsforecast(model_obj, data_split)

        # Calculate metrics
        actual = data_split['test_y']
        metrics = calculate_metrics(actual, predictions)

        logger.info(f"  MAE: {metrics['MAE']/1e9:.2f}B | MAPE: {metrics['MAPE']:.1f}%")

        # Save predictions
        output_path = save_predictions(
            name, model_name,
            data_split['test_fys'],
            actual, predictions,
            result_dir
        )

        return predictions, metrics

    except Exception as e:
        logger.error(f"Error training {model_name} for {name}: {str(e)}")
        raise


def create_ensemble(name: str, ensemble_name: str, model_names: list,
                   result_dir: Path = None) -> dict:
    """
    Create ensemble as average of multiple models.
    Reads predictions from CSV files and averages them.
    Args:
        name: Company name (not ticker)
    """
    from src.config import settings

    if result_dir is None:
        result_dir = settings.RESULT_DIR

    logger.info(f"Creating ensemble {ensemble_name.upper()} from {model_names}")

    try:
        # Load predictions from each model
        all_predictions = []
        test_fys = None
        actual = None

        for model_name in model_names:
            pred_file = result_dir / f"{name}_fcf_annual_predictions_{model_name}.csv"

            if not pred_file.exists():
                raise FileNotFoundError(f"Predictions not found for {model_name}: {pred_file}")

            df_pred = pd.read_csv(pred_file)

            if test_fys is None:
                test_fys = df_pred['fy'].values
                actual = df_pred['actual'].values

            all_predictions.append(df_pred['predicted'].values)

        # Average predictions
        ensemble_predictions = np.mean(all_predictions, axis=0)

        # Calculate metrics
        metrics = calculate_metrics(actual, ensemble_predictions)

        logger.info(f"  Ensemble MAE: {metrics['MAE']/1e9:.2f}B | MAPE: {metrics['MAPE']:.1f}%")

        # Save ensemble predictions
        output_path = save_predictions(
            name, ensemble_name,
            test_fys, actual, ensemble_predictions,
            result_dir
        )

        return {'predictions': ensemble_predictions, 'metrics': metrics}

    except Exception as e:
        logger.error(f"Error creating ensemble {ensemble_name} for {name}: {str(e)}")
        raise


def train_all_models_for_company(name: str,
                                  test_years: int = 4,
                                  data_dir: Path = None,
                                  result_dir: Path = None) -> dict:
    """
    Complete workflow for one company:
    1. Fit 4 ARIMA models
    2. Create 4 ensemble averages

    Args:
        name: Company name (not ticker)

    Returns: dict with all results
    """
    from src.config import settings

    if data_dir is None:
        data_dir = settings.DATA_DIR
    if result_dir is None:
        result_dir = settings.RESULT_DIR

    logger.info("\n" + "="*80)
    logger.info(f"TRAINING ALL MODELS FOR {name}")
    logger.info("="*80 + "\n")

    results = {}

    # Clear cache for new company
    if hasattr(train_and_predict, 'data_cache'):
        delattr(train_and_predict, 'data_cache')

    try:
        # STEP 1: Fit 4 ARIMA models
        logger.info("STEP 1: Training 4 ARIMA models...")

        arima_models = list(settings.MODELS.keys())

        for model_name in arima_models:
            try:
                logger.info(f"\n  ▶ Training {model_name.upper()}...")
                predictions, metrics = train_and_predict(
                    model_name, name,
                    test_years=test_years,
                    data_dir=data_dir,
                    result_dir=result_dir,
                )
                results[model_name] = {'predictions': predictions, 'metrics': metrics}
                logger.info(f"    ✓ {model_name.upper()} completed")
            except Exception as e:
                logger.error(f"    ✗ {model_name.upper()} failed: {str(e)}")
                results[model_name] = {'error': str(e)}

        # STEP 2: Create ensembles
        logger.info("\n\nSTEP 2: Creating ensemble models...")

        for ensemble_name, model_list in settings.ENSEMBLES.items():
            try:
                logger.info(f"\n  ▶ Creating {ensemble_name.upper()}...")
                ensemble_result = create_ensemble(
                    name, ensemble_name, model_list, result_dir
                )
                results[ensemble_name] = ensemble_result
                logger.info(f"    ✓ {ensemble_name.upper()} completed")
            except Exception as e:
                logger.error(f"    ✗ {ensemble_name.upper()} failed: {str(e)}")
                results[ensemble_name] = {'error': str(e)}

        logger.info(f"\n✓ Completed all models for {name}")
        return results

    except Exception as e:
        logger.error(f"Error in workflow for {name}: {str(e)}")
        raise


if __name__ == "__main__":
    from src.config import get_company_name

    if len(sys.argv) < 2:
        print("Usage: python -m src.train_predict <ticker>")
        print("Example: python -m src.train_predict AAPL")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    name = get_company_name(ticker)

    results = train_all_models_for_company(name)

    print(f"\n✓ Completed all models for {name}")
    print("\nSummary:")
    for model_name, result in results.items():
        if 'error' in result:
            print(f"  {model_name}: ERROR - {result['error']}")
        else:
            metrics = result['metrics']
            print(f"  {model_name}: MAE={metrics['MAE']/1e9:.2f}B, MAPE={metrics['MAPE']:.1f}%")