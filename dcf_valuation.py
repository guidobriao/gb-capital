#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCF Valuation with ARIMA(2,1,1) Forecasting
No train/test split - full historical data for model fitting
Calculates Enterprise Value using finite horizon + Terminal Value
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from src.config import settings, load_tickers, get_beta, get_risk_free_rate, get_market_premium

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_fcff_data(ticker: str) -> pd.DataFrame:
    """
    Carica dati FCFF storici da CSV (già processati da fcf_extractor.py)
    Applica solo deduplicazione
    """
    data_file = settings.DATA_DIR / settings.ANNUAL_FILE_PATTERN.format(ticker=ticker)

    if not data_file.exists():
        raise FileNotFoundError(f"FCFF file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Validazione colonne
    required_cols = ['fy', 'Free Cash Flow', 'Tax Rate', 'Total Debt', 'Equity', 'Kd']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Rinomina
    df = df.rename(columns={'Free Cash Flow': 'FCFF'})

    # Pulizia e deduplicazione
    df = df.dropna(subset=['fy', 'FCFF'])
    df['fy'] = df['fy'].astype(int)

    if 'end' in df.columns:
        df['end'] = pd.to_datetime(df['end'])
        df = df.sort_values(['fy', 'end'])
        df = df.drop_duplicates(subset=['fy'], keep='last')
    else:
        df = df.sort_values('fy')
        df = df.drop_duplicates(subset=['fy'], keep='last')

    df = df.reset_index(drop=True)

    if len(df) < settings.MIN_HISTORICAL_YEARS:
        raise ValueError(f"Insufficient data: {len(df)} years < {settings.MIN_HISTORICAL_YEARS} required")

    logger.info(f"Loaded {len(df)} years of FCFF data (FY {df['fy'].min()}-{df['fy'].max()})")

    return df


def estimate_wacc(df: pd.DataFrame, ticker: str) -> float:
    """
    Stima WACC per il ticker
    WACC = Ke × (E/(E+D)) + Kd × (1-T) × (D/(E+D))

    Ke = rf + β × (rm - rf)

    Usa yfinance per ottenere β, rf, e market premium
    """

    # Usa dati più recenti (ultimi 3 anni se disponibili)
    recent_years = min(3, len(df))
    df_recent = df.tail(recent_years)

    # Cost of Debt (Kd) - media degli ultimi anni
    kd_values = df_recent['Kd'].dropna()
    if len(kd_values) > 0:
        kd = kd_values.mean()
    else:
        logger.warning(f"No Kd data available, using 5% default")
        kd = 0.05

    # Tax Rate - media degli ultimi anni
    tax_values = df_recent['Tax Rate'].dropna()
    if len(tax_values) > 0:
        tax_rate = tax_values.mean()
    else:
        tax_rate = settings.DEFAULT_TAX_RATE

    # Debt e Equity - valori più recenti
    latest_row = df.iloc[-1]
    debt = latest_row['Total Debt']
    equity = latest_row['Equity']

    # Controllo validità
    if pd.isna(debt) or pd.isna(equity) or equity <= 0:
        logger.warning(f"Invalid D/E data, using 50/50 weights")
        weight_e = 0.5
        weight_d = 0.5
    else:
        total_value = debt + equity
        weight_e = equity / total_value
        weight_d = debt / total_value

    # ===== FETCH MARKET DATA DA YFINANCE =====

    # Beta da yfinance
    beta = get_beta(ticker)

    # Risk-free rate da ^TNX
    rf = get_risk_free_rate()

    # Market premium (rm - rf) da ACWI
    market_premium = get_market_premium()

    # Cost of Equity (Ke) via CAPM
    ke = rf + beta * market_premium

    # WACC
    wacc = ke * weight_e + kd * (1 - tax_rate) * weight_d

    logger.info(f"WACC components for {ticker}:")
    logger.info(f"  β (yfinance) = {beta:.2f}")
    logger.info(f"  rf (^TNX) = {rf:.2%}")
    logger.info(f"  Market premium = {market_premium:.2%}")
    logger.info(f"  Ke = {ke:.2%}")
    logger.info(f"  Kd = {kd:.2%}")
    logger.info(f"  Tax Rate = {tax_rate:.2%}")
    logger.info(f"  Weight E = {weight_e:.2%}, Weight D = {weight_d:.2%}")
    logger.info(f"  WACC = {wacc:.2%}")

    return wacc


def forecast_fcff_arima(df: pd.DataFrame, horizon: int = 5) -> tuple:
    """
    Forecasta FCFF usando ARIMA(2,1,1) su tutto il dataset storico
    Returns: (forecast_array, fitted_values)
    """

    # Prepara dati per StatsForecast
    train_df = pd.DataFrame({
        'unique_id': ['FCFF'] * len(df),
        'ds': pd.to_datetime([f'{fy}-12-31' for fy in df['fy']]),
        'y': df['FCFF'].values,
    })

    # Crea modello ARIMA(2,1,1)
    p, d, q = settings.ARIMA_ORDER
    model = ARIMA(order=(p, d, q), season_length=1)

    logger.info(f"Fitting ARIMA{settings.ARIMA_ORDER} on {len(train_df)} historical observations...")

    # Fit model
    sf = StatsForecast(models=[model], freq='Y', n_jobs=1)
    sf.fit(df=train_df)

    # Forecast
    forecast_df = sf.predict(h=horizon)

    # Estrai previsioni (StatsForecast usa 'ARIMA' come nome colonna)
    forecast_values = forecast_df['ARIMA'].values

    logger.info(f"Forecast completed for {horizon} years")

    return forecast_values


def calculate_terminal_value(fcff_last_forecast: float, wacc: float, growth: float) -> float:
    """
    Calcola Terminal Value usando Gordon Growth Model
    TV = FCFF_{n+1} / (WACC - g)
    """

    # Verifica vincolo g < WACC
    if growth >= wacc:
        logger.warning(f"Terminal growth ({growth:.2%}) >= WACC ({wacc:.2%}), adjusting...")
        growth = wacc * 0.8  # Usa 80% del WACC come safety margin

    # FCFF al primo anno dopo forecast horizon
    fcff_next = fcff_last_forecast * (1 + growth)

    # Terminal Value
    tv = fcff_next / (wacc - growth)

    logger.info(f"Terminal Value calculation:")
    logger.info(f"  FCFF (Year {settings.FORECAST_YEARS}) = ${fcff_last_forecast / 1e9:.2f}B")
    logger.info(f"  FCFF (Year {settings.FORECAST_YEARS + 1}) = ${fcff_next / 1e9:.2f}B")
    logger.info(f"  Terminal Growth = {growth:.2%}")
    logger.info(f"  WACC = {wacc:.2%}")
    logger.info(f"  Terminal Value = ${tv / 1e9:.2f}B")

    return tv


def calculate_adaptive_terminal_growth(wacc: float) -> float:
    """
    Adaptive terminal growth rate basato su WACC
    Logica euristica conservativa:
    - Se WACC < 8%: usa g = 2.5%
    - Se WACC 8-12%: usa g = 2.0%
    - Se WACC > 12%: usa g = 1.5%
    Vincolo sempre: g < WACC
    """
    if wacc < 0.08:
        g = 0.025  # 2.5%
    elif wacc <= 0.12:
        g = 0.020  # 2.0%
    else:
        g = 0.015  # 1.5%

    # Assicura che g < WACC
    if g >= wacc:
        g = wacc * 0.8
        logger.warning(f"Terminal growth adjusted to {g:.2%} (80% of WACC)")

    logger.info(f"Adaptive terminal growth selected: {g:.2%} (WACC={wacc:.2%})")
    return g


def calculate_enterprise_value(
        fcff_forecast: np.ndarray,
        terminal_value: float,
        wacc: float
) -> dict:
    """
    Calcola Enterprise Value
    EV = Σ(FCFF_t / (1+WACC)^t) + TV / (1+WACC)^n
    """

    n = len(fcff_forecast)

    # Present Value dei FCFF forecasted
    pv_fcff = []
    for t, fcff in enumerate(fcff_forecast, start=1):
        pv = fcff / ((1 + wacc) ** t)
        pv_fcff.append(pv)

    pv_fcff_total = sum(pv_fcff)

    # Present Value del Terminal Value
    pv_tv = terminal_value / ((1 + wacc) ** n)

    # Enterprise Value
    ev = pv_fcff_total + pv_tv

    logger.info(f"Enterprise Value calculation:")
    logger.info(f"  PV(FCFF 1-{n}) = ${pv_fcff_total / 1e9:.2f}B")
    logger.info(f"  PV(Terminal Value) = ${pv_tv / 1e9:.2f}B")
    logger.info(f"  Enterprise Value = ${ev / 1e9:.2f}B")

    return {
        'enterprise_value': ev,
        'pv_fcff': pv_fcff_total,
        'pv_terminal_value': pv_tv,
        'pv_fcff_by_year': pv_fcff,
    }


def save_results(
        ticker: str,
        df_historical: pd.DataFrame,
        fcff_forecast: np.ndarray,
        wacc: float,
        terminal_growth: float,
        terminal_value: float,
        dcf_result: dict
) -> None:
    """Salva risultati forecast e DCF"""

    settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Forecast DataFrame
    last_fy = df_historical['fy'].max()
    forecast_years = list(range(last_fy + 1, last_fy + 1 + len(fcff_forecast)))

    forecast_df = pd.DataFrame({
        'fy': forecast_years,
        'fcff_forecast': fcff_forecast,
        'pv_fcff': dcf_result['pv_fcff_by_year'],
    })

    forecast_file = settings.RESULT_DIR / settings.PREDICTION_FILE_PATTERN.format(ticker=ticker)
    forecast_df.to_csv(forecast_file, index=False)
    logger.info(f"Forecast saved to: {forecast_file}")

    # DCF Valuation Summary
    dcf_summary = pd.DataFrame([{
        'ticker': ticker,
        'wacc': wacc,
        'terminal_growth': terminal_growth,
        'terminal_value': terminal_value,
        'pv_fcff': dcf_result['pv_fcff'],
        'pv_terminal_value': dcf_result['pv_terminal_value'],
        'enterprise_value': dcf_result['enterprise_value'],
        'latest_historical_fy': df_historical['fy'].max(),
        'forecast_horizon': len(fcff_forecast),
    }])

    dcf_file = settings.RESULT_DIR / settings.DCF_RESULT_FILE_PATTERN.format(ticker=ticker)
    dcf_summary.to_csv(dcf_file, index=False)
    logger.info(f"DCF valuation saved to: {dcf_file}")


def plot_fcff_forecast(
        ticker: str,
        df_historical: pd.DataFrame,
        fcff_forecast: np.ndarray
) -> None:
    """Crea plot dei FCFF storici e forecast"""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Remove any potential NaN from historical data for plotting
    df_plot = df_historical[['fy', 'FCFF']].dropna()

    # Historical - smooth interpolation
    fy_historical = df_plot['fy'].values
    fcff_historical = df_plot['FCFF'].values / 1e9

    if len(fy_historical) > 3:  # Need at least 4 points for cubic spline
        fy_smooth = np.linspace(fy_historical.min(), fy_historical.max(), 300)
        spl = make_interp_spline(fy_historical, fcff_historical, k=3)
        fcff_smooth = spl(fy_smooth)
        ax.plot(fy_smooth, fcff_smooth, linewidth=2, color='darkblue', linestyle='-')

    ax.plot(fy_historical, fcff_historical, marker='o', markersize=8,
            color='darkblue', linestyle='', label='Historical FCFF')

    # Forecast - collega ultimo historical con primo forecast
    last_fy = df_plot['fy'].max()
    forecast_years = list(range(last_fy + 1, last_fy + 1 + len(fcff_forecast)))

    last_fcff_historical = df_plot['FCFF'].iloc[-1] / 1e9

    # Crea array completo: ultimo historical + tutti i forecast
    fy_forecast_complete = np.concatenate([[last_fy], forecast_years])
    fcff_forecast_complete = np.concatenate([[last_fcff_historical], fcff_forecast / 1e9])

    # Smooth interpolation per forecast
    if len(fy_forecast_complete) > 3:
        fy_forecast_smooth = np.linspace(fy_forecast_complete.min(), fy_forecast_complete.max(), 300)
        spl_forecast = make_interp_spline(fy_forecast_complete, fcff_forecast_complete, k=3)
        fcff_forecast_smooth = spl_forecast(fy_forecast_smooth)
        ax.plot(fy_forecast_smooth, fcff_forecast_smooth, linewidth=2, color='darkred', linestyle='--')

    ax.plot(forecast_years, fcff_forecast / 1e9, marker='s', markersize=8,
            color='darkred', linestyle='', label='ARIMA(2,1,1) Forecast')

    # Formatting
    ax.set_xlabel('Fiscal Year', fontsize=12)
    ax.set_ylabel('FCFF ($B)', fontsize=12)
    ax.set_title(f'{ticker} - Free Cash Flow to Firm Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    plot_file = settings.RESULT_DIR / f"{ticker}_fcff_forecast.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Plot saved to: {plot_file}")


def run_dcf_valuation(ticker: str) -> dict:
    """
    Pipeline completa DCF per un ticker:
    1. Carica dati FCFF storici
    2. Stima WACC
    3. Forecasta FCFF con ARIMA(2,1,1)
    4. Calcola Terminal Value
    5. Calcola Enterprise Value
    """

    logger.info("=" * 80)
    logger.info(f"DCF VALUATION - {ticker}")
    logger.info("=" * 80)

    try:
        # Step 1: Load data
        logger.info("\nStep 1: Loading FCFF historical data...")
        df = load_fcff_data(ticker)

        # Step 2: Estimate WACC
        logger.info("\nStep 2: Estimating WACC...")
        wacc = estimate_wacc(df, ticker)

        # Step 3: Forecast FCFF
        logger.info(f"\nStep 3: Forecasting FCFF for {settings.FORECAST_YEARS} years...")
        fcff_forecast = forecast_fcff_arima(df, horizon=settings.FORECAST_YEARS)

        # Step 4: Terminal Value
        logger.info("\nStep 4: Calculating Terminal Value...")

        # Usa adaptive terminal growth basato su WACC
        terminal_growth = calculate_adaptive_terminal_growth(wacc)

        terminal_value = calculate_terminal_value(
            fcff_forecast[-1],
            wacc,
            terminal_growth
        )

        # Step 5: Enterprise Value
        logger.info("\nStep 5: Calculating Enterprise Value...")
        dcf_result = calculate_enterprise_value(fcff_forecast, terminal_value, wacc)

        # Step 6: Save results
        logger.info("\nStep 6: Saving results...")
        save_results(
            ticker, df, fcff_forecast,
            wacc, terminal_growth, terminal_value,
            dcf_result
        )

        # Step 7: Plot
        logger.info("\nStep 7: Generating plot...")
        plot_fcff_forecast(ticker, df, fcff_forecast)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"✓ DCF VALUATION COMPLETED FOR {ticker}")
        logger.info(f"  Enterprise Value: ${dcf_result['enterprise_value'] / 1e9:.2f}B")
        logger.info(f"{'=' * 80}\n")

        return {
            'ticker': ticker,
            'success': True,
            'enterprise_value': dcf_result['enterprise_value'],
            'wacc': wacc,
            'terminal_growth': terminal_growth,
        }

    except Exception as e:
        logger.error(f"\n✗ ERROR in DCF valuation for {ticker}: {str(e)}\n")
        return {
            'ticker': ticker,
            'success': False,
            'error': str(e),
        }


def run_all_tickers():
    """Esegue DCF valuation per tutti i ticker"""

    companies = load_tickers()

    if not companies:
        logger.error("No tickers found in tickers.txt")
        return

    tickers = [c['TICKER'] for c in companies]

    logger.info(f"\n{'=' * 80}")
    logger.info(f"DCF VALUATION - ALL TICKERS")
    logger.info(f"{'=' * 80}")
    logger.info(f"Processing {len(tickers)} tickers: {', '.join(tickers)}")
    logger.info(f"{'=' * 80}\n")

    results = []

    for ticker in tickers:
        result = run_dcf_valuation(ticker)
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY - DCF VALUATIONS")
    logger.info("=" * 80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if successful:
        logger.info(f"\n✓ Successful: {len(successful)}/{len(tickers)}")

        summary_data = []
        for r in successful:
            summary_data.append({
                'Ticker': r['ticker'],
                'EV ($B)': r['enterprise_value'] / 1e9,
                'WACC': f"{r['wacc']:.2%}",
                'Terminal g': f"{r['terminal_growth']:.2%}",
            })

        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))

    if failed:
        logger.info(f"\n✗ Failed: {len(failed)}/{len(tickers)}")
        for r in failed:
            logger.info(f"  {r['ticker']}: {r['error']}")

    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single ticker mode
        ticker = sys.argv[1].upper()
        run_dcf_valuation(ticker)
    else:
        # All tickers mode
        run_all_tickers()