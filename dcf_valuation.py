#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCF Valuation with ARIMA(2,1,1) Forecasting
NO DEDUPLICATION - CSV already deduplicated by fcf_extractor.py
All data already in BILLIONS ($B) - NO CONVERSION NEEDED
"""

import logging
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from statsforecast import StatsForecast
from statsforecast.models import ARIMA
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from src.config import settings, load_tickers, get_company_name

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== MARKET DATA FUNCTIONS (from utils.py) =====

def get_beta(ticker: str, fallback: float = 1.2) -> float:
    """Fetch beta for a ticker from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        beta = info.get('beta')

        if beta is None or not isinstance(beta, (int, float)):
            logger.warning(f"Beta not available for {ticker}, using fallback")
            return fallback

        if not (-2 < beta < 3):
            logger.warning(f"Beta {beta} for {ticker} out of range, using fallback")
            return fallback

        logger.info(f"Beta for {ticker}: {beta:.2f}")
        return beta

    except Exception as e:
        logger.warning(f"Error fetching beta for {ticker}: {str(e)}, using fallback")
        return fallback


def get_risk_free_rate(treasury_ticker: str = "^TNX", fallback: float = 0.04) -> float:
    """Fetch current US 10-Year Treasury Yield from yfinance"""
    try:
        tnx = yf.Ticker(treasury_ticker)
        hist = tnx.history(period="5d")

        if hist.empty:
            logger.warning(f"No data for {treasury_ticker}, using fallback")
            return fallback

        rf_rate = hist['Close'].iloc[-1] / 100  # Convert from % to decimal

        if not (0 < rf_rate < 0.10):
            logger.warning(f"Risk-free rate {rf_rate:.2%} out of range, using fallback")
            return fallback

        logger.info(f"Risk-free rate: {rf_rate:.2%}")
        return rf_rate

    except Exception as e:
        logger.warning(f"Error fetching risk-free rate: {str(e)}, using fallback")
        return fallback


def get_market_return(market_ticker: str = "ACWI", fallback: float = 0.10) -> float:
    """Calculate annualized market return from MSCI ACWI ETF"""
    try:
        acwi = yf.Ticker(market_ticker)
        hist = acwi.history(period="20y")

        if len(hist) < 252 * 10:
            logger.info("20-year data not available, using 10-year period")
            hist = acwi.history(period="10y")

        if hist.empty or len(hist) < 252:
            logger.warning(f"Insufficient data for {market_ticker}, using fallback")
            return fallback

        years = len(hist) / 252
        total_return = hist['Close'].iloc[-1] / hist['Close'].iloc[0]
        annual_return = total_return ** (1 / years) - 1

        if not (-0.5 < annual_return < 0.5):
            logger.warning(f"Market return {annual_return:.2%} out of range, using fallback")
            return fallback

        logger.info(f"Market return ({years:.1f}y): {annual_return:.2%}")
        return annual_return

    except Exception as e:
        logger.warning(f"Error fetching market return: {str(e)}, using fallback")
        return fallback


def get_market_premium(market_ticker: str = "ACWI", treasury_ticker: str = "^TNX") -> float:
    """Calculate market risk premium (rm - rf)"""
    rf = get_risk_free_rate(treasury_ticker)
    rm = get_market_return(market_ticker)
    premium = rm - rf
    logger.info(f"Market premium: {premium:.2%}")
    return premium


# ===== DCF VALUATION FUNCTIONS =====

def load_fcff_data(name: str) -> pd.DataFrame:
    """
    Load historical FCFF data from CSV (already processed and deduplicated)
    NO DEDUPLICATION - already done in fcf_extractor.py
    Data already in BILLIONS ($B)
    Args:
        name: Company name (not ticker)
    """
    data_file = settings.DATA_DIR / settings.ANNUAL_FILE_PATTERN.format(name=name)

    if not data_file.exists():
        raise FileNotFoundError(f"FCFF file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Validate columns
    required_cols = ['fy', 'Free Cash Flow', 'Tax Rate', 'Total Debt', 'Equity', 'Kd']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Rename for consistency
    df = df.rename(columns={'Free Cash Flow': 'FCFF'})

    # Clean NaN
    df = df.dropna(subset=['fy', 'FCFF'])
    df['fy'] = df['fy'].astype(int)

    # NO DEDUPLICATION - just sort
    df = df.sort_values('fy').reset_index(drop=True)

    if len(df) < settings.MIN_HISTORICAL_YEARS:
        raise ValueError(f"Insufficient data: {len(df)} years < {settings.MIN_HISTORICAL_YEARS} required")

    logger.info(f"Loaded {len(df)} years of FCFF data (FY {df['fy'].min()}-{df['fy'].max()})")
    logger.info(f"Data already in BILLIONS - no conversion needed")

    return df


def load_market_caps() -> dict:
    """Load market caps from CSV"""
    market_cap_file = settings.DATA_DIR / "market_caps.csv"

    if not market_cap_file.exists():
        logger.warning(f"Market cap file not found: {market_cap_file}")
        return {}

    try:
        df = pd.read_csv(market_cap_file)
        # Use 'name' column instead of 'ticker'
        market_caps = dict(zip(df['name'], df['market_cap']))
        logger.info(f"Loaded market caps for {len(market_caps)} companies")
        return market_caps
    except Exception as e:
        logger.warning(f"Error loading market caps: {str(e)}")
        return {}


def estimate_wacc(df: pd.DataFrame, ticker: str) -> float:
    """Estimate WACC for ticker"""
    # Use recent years (last 3 if available)
    recent_years = min(3, len(df))
    df_recent = df.tail(recent_years)

    # Cost of Debt
    kd_values = df_recent['Kd'].dropna()
    kd = kd_values.mean() if len(kd_values) > 0 else 0.05

    # Tax Rate
    tax_values = df_recent['Tax Rate'].dropna()
    tax_rate = tax_values.mean() if len(tax_values) > 0 else settings.DEFAULT_TAX_RATE

    # Debt and Equity
    latest_row = df.iloc[-1]
    debt = latest_row['Total Debt']  # Already in billions
    equity = latest_row['Equity']  # Already in billions

    if pd.isna(debt) or pd.isna(equity) or equity <= 0:
        logger.warning(f"Invalid D/E data, using 50/50 weights")
        weight_e = 0.5
        weight_d = 0.5
    else:
        total_value = debt + equity
        weight_e = equity / total_value
        weight_d = debt / total_value

    # Market data from yfinance
    beta = get_beta(ticker, settings.FALLBACK_BETA)
    rf = get_risk_free_rate(settings.TREASURY_10Y_TICKER, settings.FALLBACK_RISK_FREE_RATE)
    market_premium = get_market_premium(settings.MARKET_INDEX_TICKER, settings.TREASURY_10Y_TICKER)

    # Cost of Equity (CAPM)
    ke = rf + beta * market_premium

    # WACC
    wacc = ke * weight_e + kd * (1 - tax_rate) * weight_d

    logger.info(f"WACC components for {ticker}:")
    logger.info(f"  β = {beta:.2f}, rf = {rf:.2%}, Market premium = {market_premium:.2%}")
    logger.info(f"  Ke = {ke:.2%}, Kd = {kd:.2%}, Tax = {tax_rate:.2%}")
    logger.info(f"  Weight E = {weight_e:.2%}, Weight D = {weight_d:.2%}")
    logger.info(f"  WACC = {wacc:.2%}")

    return wacc


def forecast_fcff_arima(df: pd.DataFrame, horizon: int = 5) -> np.ndarray:
    """
    Forecast FCFF using ARIMA(2,1,1)
    Data already in BILLIONS - returns forecast in BILLIONS
    """
    train_df = pd.DataFrame({
        'unique_id': ['FCFF'] * len(df),
        'ds': pd.to_datetime([f'{fy}-12-31' for fy in df['fy']]),
        'y': df['FCFF'].values,  # Already in billions
    })

    p, d, q = settings.ARIMA_ORDER
    model = ARIMA(order=(p, d, q), season_length=1)

    logger.info(f"Fitting ARIMA{settings.ARIMA_ORDER} on {len(train_df)} observations...")

    sf = StatsForecast(models=[model], freq='Y', n_jobs=1)
    sf.fit(df=train_df)

    forecast_df = sf.predict(h=horizon)
    forecast_values = forecast_df['ARIMA'].values  # Already in billions

    logger.info(f"Forecast completed for {horizon} years (values in billions)")

    return forecast_values


def calculate_terminal_value(fcff_last_forecast: float, wacc: float, growth: float) -> float:
    """Calculate Terminal Value using Gordon Growth Model"""
    if growth >= wacc:
        logger.warning(f"Terminal growth ({growth:.2%}) >= WACC ({wacc:.2%}), adjusting...")
        growth = wacc * 0.8

    fcff_next = fcff_last_forecast * (1 + growth)
    tv = fcff_next / (wacc - growth)

    logger.info(f"Terminal Value: ${tv:.2f}B (growth={growth:.2%}, WACC={wacc:.2%})")

    return tv


def calculate_adaptive_terminal_growth(wacc: float) -> float:
    """Adaptive terminal growth rate based on WACC"""
    if wacc < 0.08:
        g = 0.025
    elif wacc <= 0.12:
        g = 0.020
    else:
        g = 0.015

    if g >= wacc:
        g = wacc * 0.8

    logger.info(f"Adaptive terminal growth: {g:.2%}")
    return g


def calculate_enterprise_value(fcff_forecast: np.ndarray, terminal_value: float, wacc: float) -> dict:
    """Calculate Enterprise Value"""
    n = len(fcff_forecast)

    # PV of forecasted FCFF
    pv_fcff = [fcff / ((1 + wacc) ** t) for t, fcff in enumerate(fcff_forecast, start=1)]
    pv_fcff_total = sum(pv_fcff)

    # PV of Terminal Value
    pv_tv = terminal_value / ((1 + wacc) ** n)

    # Enterprise Value
    ev = pv_fcff_total + pv_tv

    logger.info(f"Enterprise Value: ${ev:.2f}B (PV FCFF: ${pv_fcff_total:.2f}B, PV TV: ${pv_tv:.2f}B)")

    return {
        'enterprise_value': ev,
        'pv_fcff': pv_fcff_total,
        'pv_terminal_value': pv_tv,
        'pv_fcff_by_year': pv_fcff,
    }


def save_results(name: str, df_historical: pd.DataFrame, fcff_forecast: np.ndarray,
                 wacc: float, terminal_growth: float, terminal_value: float, dcf_result: dict):
    """Save forecast and DCF results using company name"""
    settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Forecast DataFrame
    last_fy = df_historical['fy'].max()
    forecast_years = list(range(last_fy + 1, last_fy + 1 + len(fcff_forecast)))

    forecast_df = pd.DataFrame({
        'fy': forecast_years,
        'fcff_forecast': fcff_forecast,
        'pv_fcff': dcf_result['pv_fcff_by_year'],
    })

    forecast_file = settings.RESULT_DIR / settings.PREDICTION_FILE_PATTERN.format(name=name)
    forecast_df.to_csv(forecast_file, index=False)
    logger.info(f"Forecast saved to: {forecast_file}")

    # DCF Summary
    dcf_summary = pd.DataFrame([{
        'name': name,
        'wacc': wacc,
        'terminal_growth': terminal_growth,
        'terminal_value': terminal_value,
        'pv_fcff': dcf_result['pv_fcff'],
        'pv_terminal_value': dcf_result['pv_terminal_value'],
        'enterprise_value': dcf_result['enterprise_value'],
        'latest_historical_fy': df_historical['fy'].max(),
        'forecast_horizon': len(fcff_forecast),
    }])

    dcf_file = settings.RESULT_DIR / settings.DCF_RESULT_FILE_PATTERN.format(name=name)
    dcf_summary.to_csv(dcf_file, index=False)
    logger.info(f"DCF valuation saved to: {dcf_file}")


def plot_fcff_forecast(name: str, df_historical: pd.DataFrame, fcff_forecast: np.ndarray):
    """Create FCFF forecast plot - NO DIVISION, data already in billions"""
    fig, ax = plt.subplots(figsize=(12, 6))

    df_plot = df_historical[['fy', 'FCFF']].dropna()

    # Historical (already in billions)
    fy_historical = df_plot['fy'].values
    fcff_historical = df_plot['FCFF'].values  # Already in billions - NO DIVISION

    if len(fy_historical) > 3:
        fy_smooth = np.linspace(fy_historical.min(), fy_historical.max(), 300)
        spl = make_interp_spline(fy_historical, fcff_historical, k=3)
        fcff_smooth = spl(fy_smooth)
        ax.plot(fy_smooth, fcff_smooth, linewidth=2, color='darkblue', linestyle='-')

    ax.plot(fy_historical, fcff_historical, marker='o', markersize=8,
            color='darkblue', linestyle='', label='Historical FCFF')

    # Forecast (already in billions - NO DIVISION)
    last_fy = df_plot['fy'].max()
    forecast_years = list(range(last_fy + 1, last_fy + 1 + len(fcff_forecast)))
    last_fcff_historical = df_plot['FCFF'].iloc[-1]  # Already in billions - NO DIVISION

    fy_forecast_complete = np.concatenate([[last_fy], forecast_years])
    fcff_forecast_complete = np.concatenate([[last_fcff_historical], fcff_forecast])  # Already in billions

    if len(fy_forecast_complete) > 3:
        fy_forecast_smooth = np.linspace(fy_forecast_complete.min(), fy_forecast_complete.max(), 300)
        spl_forecast = make_interp_spline(fy_forecast_complete, fcff_forecast_complete, k=3)
        fcff_forecast_smooth = spl_forecast(fy_forecast_smooth)
        ax.plot(fy_forecast_smooth, fcff_forecast_smooth, linewidth=2, color='darkred', linestyle='--')

    ax.plot(forecast_years, fcff_forecast, marker='s', markersize=8,  # Already in billions - NO DIVISION
            color='darkred', linestyle='', label='ARIMA(2,1,1) Forecast')

    ax.set_xlabel('Fiscal Year', fontsize=12)
    ax.set_ylabel('FCFF ($B)', fontsize=12)
    ax.set_title(f'{name} - Free Cash Flow to Firm Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = settings.RESULT_DIR / f"{name}_fcff_forecast.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to: {plot_file}")


def run_dcf_valuation(ticker: str, name: str = None) -> dict:
    """
    Complete DCF pipeline for one company
    Args:
        ticker: Stock ticker symbol
        name: Company name (if None, will be fetched from config)
    """
    if name is None:
        name = get_company_name(ticker)

    logger.info("=" * 80)
    logger.info(f"DCF VALUATION - {name}")
    logger.info("=" * 80)

    try:
        df = load_fcff_data(name)
        wacc = estimate_wacc(df, ticker)
        fcff_forecast = forecast_fcff_arima(df, horizon=settings.FORECAST_YEARS)
        terminal_growth = calculate_adaptive_terminal_growth(wacc)
        terminal_value = calculate_terminal_value(fcff_forecast[-1], wacc, terminal_growth)
        dcf_result = calculate_enterprise_value(fcff_forecast, terminal_value, wacc)
        save_results(name, df, fcff_forecast, wacc, terminal_growth, terminal_value, dcf_result)
        plot_fcff_forecast(name, df, fcff_forecast)

        logger.info(f"\n✓ DCF COMPLETED - EV: ${dcf_result['enterprise_value']:.2f}B\n")

        return {
            'name': name,
            'ticker': ticker,
            'success': True,
            'enterprise_value': dcf_result['enterprise_value'],
            'wacc': wacc,
            'terminal_growth': terminal_growth,
        }

    except Exception as e:
        logger.error(f"\n✗ ERROR for {name}: {str(e)}\n")
        return {
            'name': name,
            'ticker': ticker,
            'success': False,
            'error': str(e),
        }


def run_all_companies():
    """Run DCF valuation for all companies"""
    companies = load_tickers()

    logger.info(f"\nProcessing {len(companies)} companies\n")

    market_caps = load_market_caps()
    results = [run_dcf_valuation(c['TICKER'], c['NAME']) for c in companies]

    # Summary
    successful = [r for r in results if r['success']]

    if successful:
        summary_df = pd.DataFrame([{
            'Name': r['name'],
            'EV ($B)': r['enterprise_value'],
            'WACC': f"{r['wacc']:.2%}",
            'Terminal g': f"{r['terminal_growth']:.2%}",
        } for r in successful])

        print("\n" + "=" * 80)
        print("DCF VALUATIONS SUMMARY")
        print("=" * 80)
        print("\n" + summary_df.to_string(index=False))

        # Margin of Safety
        if market_caps:
            safety_data = []
            for r in successful:
                name = r['name']
                ev = r['enterprise_value']
                mc = market_caps.get(name)

                if ev > 0 and mc:
                    margin = (ev - mc / 1e9) / (mc / 1e9) * 100
                    safety_data.append({
                        'Name': name,
                        'EV ($B)': ev,
                        'Market Cap ($B)': mc / 1e9,
                        'Margin of Safety (%)': margin
                    })

            if safety_data:
                safety_df = pd.DataFrame(safety_data).sort_values('Margin of Safety (%)', ascending=False)
                print("\n" + "=" * 80)
                print("MARGIN OF SAFETY")
                print("=" * 80)
                print("\n" + safety_df.to_string(index=False))

                # Save to CSV
                mos_csv_file = settings.RESULT_DIR / "margin_of_safety.csv"
                safety_df.to_csv(mos_csv_file, index=False)
                logger.info(f"\n✓ Margin of Safety table saved to: {mos_csv_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        name = get_company_name(ticker)
        run_dcf_valuation(ticker, name)
    else:
        run_all_companies()