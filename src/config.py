"""
Configuration for DCF Valuation with ARIMA(2,1,1)
Market data fetched from yfinance API
"""

from pathlib import Path
import yfinance as yf
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class Settings:
    """Settings per DCF forecasting system"""
    
    # Directories
    DATA_DIR = Path("data")
    RESULT_DIR = Path("result")
    
    # File patterns
    ANNUAL_FILE_PATTERN = "{ticker}_fcf_10K.csv"  # Updated to match 10-K filename
    PREDICTION_FILE_PATTERN = "{ticker}_fcff_forecast.csv"
    DCF_RESULT_FILE_PATTERN = "{ticker}_dcf_valuation.csv"
    
    # ARIMA Model (for DCF valuation)
    ARIMA_ORDER = (2, 1, 1)  # p, d, q for ARIMA(2,1,1)

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

    # Fallback values (se yfinance fail)
    FALLBACK_RISK_FREE_RATE = 0.04  # 4%
    FALLBACK_MARKET_RETURN = 0.10  # 10%
    FALLBACK_BETA = 1.2  # Market beta

    # Default corporate tax rate (se non disponibile nei dati)
    DEFAULT_TAX_RATE = 0.21

    # Minimum data requirements
    MIN_HISTORICAL_YEARS = 4

    # Market return calculation
    MARKET_RETURN_PERIOD = "20y"  # Long-term period for annualized return
    MARKET_RETURN_FALLBACK_PERIOD = "10y"  # Fallback if 20y not available


settings = Settings()


def load_tickers(filename='tickers.txt'):
    """Carica i ticker dal file di configurazione"""
    companies = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) == 2:
                        cik, ticker = parts
                        companies.append({
                            "CIK": cik.strip(),
                            "TICKER": ticker.strip()
                        })
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return []

    return companies


# ===== YFINANCE MARKET DATA FUNCTIONS =====

def get_risk_free_rate() -> float:
    """
    Fetch current US 10-Year Treasury Yield from yfinance
    Ticker: ^TNX
    Returns: Risk-free rate as decimal (e.g., 0.042 for 4.2%)
    """
    try:
        tnx = yf.Ticker(settings.TREASURY_10Y_TICKER)
        hist = tnx.history(period="5d")  # Last 5 days to handle market closures

        if hist.empty:
            logger.warning(f"No data for {settings.TREASURY_10Y_TICKER}, using fallback")
            return settings.FALLBACK_RISK_FREE_RATE

        # Get most recent close price (yield in %)
        rf_rate = hist['Close'].iloc[-1] / 100  # Convert from % to decimal

        # Sanity check (rf should be between 0% and 10%)
        if not (0 < rf_rate < 0.10):
            logger.warning(f"Risk-free rate {rf_rate:.2%} out of range, using fallback")
            return settings.FALLBACK_RISK_FREE_RATE

        logger.info(f"Risk-free rate (^TNX): {rf_rate:.2%}")
        return rf_rate

    except Exception as e:
        logger.warning(f"Error fetching risk-free rate: {str(e)}, using fallback")
        return settings.FALLBACK_RISK_FREE_RATE


def get_market_return() -> float:
    """
    Calculate annualized market return from MSCI ACWI ETF
    Ticker: ACWI
    Returns: Annualized return as decimal (e.g., 0.095 for 9.5%)
    """
    try:
        acwi = yf.Ticker(settings.MARKET_INDEX_TICKER)

        # Try 20-year period first
        hist = acwi.history(period=settings.MARKET_RETURN_PERIOD)

        # Fallback to 10-year if 20-year not available
        if len(hist) < 252 * 10:  # Less than 10 years of data
            logger.info(f"20-year data not available, using 10-year period")
            hist = acwi.history(period=settings.MARKET_RETURN_FALLBACK_PERIOD)

        if hist.empty or len(hist) < 252:  # Less than 1 year
            logger.warning(f"Insufficient data for {settings.MARKET_INDEX_TICKER}, using fallback")
            return settings.FALLBACK_MARKET_RETURN

        # Calculate annualized return
        years = len(hist) / 252  # Approximate trading days per year
        total_return = hist['Close'].iloc[-1] / hist['Close'].iloc[0]
        annual_return = total_return ** (1 / years) - 1

        # Sanity check (should be between -50% and +50% annually)
        if not (-0.5 < annual_return < 0.5):
            logger.warning(f"Market return {annual_return:.2%} out of range, using fallback")
            return settings.FALLBACK_MARKET_RETURN

        logger.info(f"Market return ({settings.MARKET_INDEX_TICKER}, {years:.1f}y): {annual_return:.2%}")
        return annual_return

    except Exception as e:
        logger.warning(f"Error fetching market return: {str(e)}, using fallback")
        return settings.FALLBACK_MARKET_RETURN


def get_beta(ticker: str) -> float:
    """
    Fetch beta for a ticker from yfinance
    Beta is calculated vs benchmark (typically S&P 500)
    Returns: Beta value (e.g., 1.24)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        beta = info.get('beta')

        if beta is None or not isinstance(beta, (int, float)):
            logger.warning(f"Beta not available for {ticker}, using fallback")
            return settings.FALLBACK_BETA

        # Sanity check (beta should be between -2 and 3 for most stocks)
        if not (-2 < beta < 3):
            logger.warning(f"Beta {beta} for {ticker} out of range, using fallback")
            return settings.FALLBACK_BETA

        logger.info(f"Beta for {ticker}: {beta:.2f}")
        return beta

    except Exception as e:
        logger.warning(f"Error fetching beta for {ticker}: {str(e)}, using fallback")
        return settings.FALLBACK_BETA


def get_market_premium() -> float:
    """
    Calculate market risk premium (rm - rf)
    Returns: Market premium as decimal
    """
    rf = get_risk_free_rate()
    rm = get_market_return()
    premium = rm - rf

    logger.info(f"Market premium: {premium:.2%} (rm={rm:.2%} - rf={rf:.2%})")
    return premium


def load_fcff_from_yfinance(ticker: str, min_years: int = 5) -> pd.DataFrame:
    """
    Carica FCFF storici annuali da yfinance API
    Returns DataFrame con colonne: fy, FCFF, Tax Rate, Total Debt, Equity, Kd, end
    Raises exception se dati insufficienti o non disponibili
    """
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)

        # Get financial statements
        cf = stock.cashflow  # Cash flow statement
        bs = stock.balance_sheet  # Balance sheet
        inc = stock.income_stmt  # Income statement

        if cf.empty or bs.empty or inc.empty:
            raise ValueError(f"Incomplete financial data from yfinance for {ticker}")

        # Transpose per avere anni come righe
        cf = cf.T
        bs = bs.T
        inc = inc.T

        # Calculate FCFF components
        data = []
        for date in cf.index:
            try:
                # Operating Cash Flow
                ocf = cf.loc[date, 'Operating Cash Flow'] if 'Operating Cash Flow' in cf.columns else None

                # CapEx
                capex = cf.loc[date, 'Capital Expenditure'] if 'Capital Expenditure' in cf.columns else 0

                # Interest Expense
                interest = inc.loc[
                    date, 'Interest Expense'] if date in inc.index and 'Interest Expense' in inc.columns else 0

                # Tax Rate
                tax_expense = inc.loc[
                    date, 'Tax Provision'] if date in inc.index and 'Tax Provision' in inc.columns else 0
                pretax_income = inc.loc[
                    date, 'Pretax Income'] if date in inc.index and 'Pretax Income' in inc.columns else 1
                tax_rate = tax_expense / pretax_income if pretax_income != 0 else settings.DEFAULT_TAX_RATE
                tax_rate = max(0, min(0.5, tax_rate))  # Clamp to [0, 0.5]

                # Total Debt
                lt_debt = bs.loc[date, 'Long Term Debt'] if date in bs.index and 'Long Term Debt' in bs.columns else 0
                st_debt = bs.loc[date, 'Current Debt'] if date in bs.index and 'Current Debt' in bs.columns else 0
                total_debt = lt_debt + st_debt

                # Equity
                equity = bs.loc[
                    date, 'Stockholders Equity'] if date in bs.index and 'Stockholders Equity' in bs.columns else 0

                # Calculate FCFF
                if ocf is not None:
                    fcff = ocf + interest * (1 - tax_rate) - abs(capex)
                else:
                    continue  # Skip if no OCF

                # Kd (cost of debt)
                kd = interest / total_debt if total_debt > 0 else 0.05

                # Fiscal year
                fy = date.year

                data.append({
                    'fy': fy,
                    'Free Cash Flow': fcff,
                    'Tax Rate': tax_rate,
                    'Total Debt': total_debt,
                    'Equity': equity,
                    'Kd': kd,
                    'end': date
                })

            except Exception as e:
                logger.warning(f"Error processing year {date}: {e}")
                continue

        if len(data) < min_years:
            raise ValueError(f"Insufficient data from yfinance: {len(data)} years < {min_years} required")

        df = pd.DataFrame(data)
        df = df.sort_values('fy')

        logger.info(f"Loaded {len(df)} years from yfinance for {ticker} (FY {df['fy'].min()}-{df['fy'].max()})")
        return df

    except Exception as e:
        logger.warning(f"Failed to load from yfinance for {ticker}: {str(e)}")
        raise