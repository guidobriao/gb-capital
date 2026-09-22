"""
Configuration for DCF Valuation with ARIMA Models
Market data fetched from yfinance API
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Settings:
    """Settings for DCF forecasting system"""
    
    # Directories (absolute paths from project root)
    _PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    DATA_DIR = _PROJECT_ROOT / "data"
    RESULT_DIR = _PROJECT_ROOT / "result"

    # File patterns
    ANNUAL_FILE_PATTERN = "{name}_fcf_10K.csv"
    PREDICTION_FILE_PATTERN = "{name}_fcff_forecast.csv"
    DCF_RESULT_FILE_PATTERN = "{name}_dcf_valuation.csv"

    # ARIMA Model for DCF valuation (default)
    ARIMA_ORDER = (1, 1, 0)  # p, d, q for ARIMA(2,1,1)

    # Model Comparison - Base ARIMA models (as dict)
    MODELS = {
        'arima_110': {
            'type': 'statsforecast',
            'params': {'order': (1, 1, 0), 'season_length': 1}
        },
        'arima_111': {
            'type': 'statsforecast',
            'params': {'order': (1, 1, 1), 'season_length': 1}
        },
        'arima_210': {
            'type': 'statsforecast',
            'params': {'order': (2, 1, 0), 'season_length': 1}
        },
        'arima_211': {
            'type': 'statsforecast',
            'params': {'order': (2, 1, 1), 'season_length': 1}
        },
    }

    # Model Comparison - Ensemble combinations (as dict)
    ENSEMBLES = {
        'ensemble_110_111': ['arima_110', 'arima_111'],
        'ensemble_110_210': ['arima_110', 'arima_210'],
        'ensemble_111_211': ['arima_111', 'arima_211'],
        'ensemble_210_211': ['arima_210', 'arima_211'],
    }

    # Forecast horizon
    FORECAST_YEARS = 5

    # Terminal Value parameters
    TERMINAL_GROWTH_MIN = 0.015  # 1.5%
    TERMINAL_GROWTH_MAX = 0.025  # 2.5%
    TERMINAL_GROWTH_DEFAULT = 0.02  # 2.0%

    # Market data tickers for yfinance
    TREASURY_10Y_TICKER = "^TNX"  # US 10-Year Treasury Yield
    MARKET_INDEX_TICKER = "ACWI"  # iShares MSCI ACWI ETF

    # Fallback values (if yfinance fails)
    FALLBACK_RISK_FREE_RATE = 0.04  # 4%
    FALLBACK_MARKET_RETURN = 0.10  # 10%
    FALLBACK_BETA = 1.2  # Market beta

    # Default corporate tax rate
    DEFAULT_TAX_RATE = 0.21

    # Minimum data requirements
    MIN_HISTORICAL_YEARS = 4

    # Market return calculation
    MARKET_RETURN_PERIOD = "20y"  # Long-term period for annualized return
    MARKET_RETURN_FALLBACK_PERIOD = "10y"  # Fallback if 20y not available


settings = Settings()


# ===== TICKER LOADING FUNCTIONS =====

def load_tickers(filename='tickers.txt'):
    """
    Load tickers, names, and CIK from configuration file
    Format: US;INDUSTRY;NAME;TICKER;CIK
    Returns list of dicts with NAME, TICKER, and CIK
    """
    companies = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(';')
                    if len(parts) >= 5:
                        name = parts[2].strip()  # NAME is at index 2
                        ticker = parts[3].strip()
                        cik = parts[4].strip() if parts[4].strip() else None
                        companies.append({"NAME": name, "TICKER": ticker, "CIK": cik})
    except FileNotFoundError:
        logger.error(f"File {filename} not found")
        return []

    return companies


def load_tickers_only(filename='tickers.txt'):
    """
    Load only ticker symbols (for plotting scripts)
    Format: US;INDUSTRY;NAME;TICKER;CIK
    """
    tickers = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(';')
                if len(parts) >= 5:
                    ticker = parts[3].strip()
                    cik = parts[4].strip()
                    if cik:  # Only include tickers with CIK
                        tickers.append(ticker)
    return tickers


def get_company_name(ticker: str, filename='tickers.txt') -> str:
    """
    Get company name for a given ticker
    Format: US;INDUSTRY;NAME;TICKER;CIK
    Returns name if found, otherwise returns ticker
    """
    companies = load_tickers(filename)
    for company in companies:
        if company['TICKER'] == ticker:
            return company['NAME']
    return ticker  # Fallback to ticker if not found