#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEC EDGAR Free Cash Flow Historical Data Extractor
FCFF = Operating Cash Flow + Interest × (1-T) - CapEx

All data in BILLIONS ($B)
Deduplication: ONCE in merge function
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import requests
import yfinance as yf

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== SEC EDGAR API =====

def get_company_facts(cik: str, headers: dict):
    """Fetch XBRL data from SEC EDGAR CompanyFacts API"""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logger.warning(f"SEC EDGAR error: {response.status_code}")
        return None


def extract_fcf_data(company_data: dict) -> pd.DataFrame:
    """Extract FCF components from XBRL data"""
    fcf_data = []

    fcf_tags = {
        'NetCashProvidedByUsedInOperatingActivities': 'Operating Cash Flow',
        'PaymentsToAcquirePropertyPlantAndEquipment': 'CapEx',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations': 'Operating Cash Flow (Continuing)',
        'InterestExpense': 'Interest Expense',
        'InterestAndDebtExpense': 'Interest And Debt Expense',
        'IncomeTaxExpenseBenefit': 'Income Tax Expense',
        'OperatingIncomeLoss': 'EBIT',
        'LongTermDebt': 'Long-Term Debt',
        'ShortTermBorrowings': 'Short-Term Debt',
        'DebtCurrent': 'Current Debt',
        'StockholdersEquity': 'Shareholders Equity',
        'CashAndCashEquivalentsAtCarryingValue': 'Cash',
    }

    facts = company_data.get('facts', {})

    if 'us-gaap' in facts:
        for tag, description in fcf_tags.items():
            if tag in facts['us-gaap']:
                units = facts['us-gaap'][tag].get('units', {})
                if 'USD' in units:
                    for item in units['USD']:
                        if 'frame' in item:
                            fcf_data.append({
                                'metric': description,
                                'value': item.get('val', 0),
                                'form': item.get('form', ''),
                                'fy': item.get('fy', ''),
                                'fp': item.get('fp', ''),
                                'end': item.get('end', '')
                            })

    return pd.DataFrame(fcf_data)


# ===== YFINANCE API =====

def load_fcff_from_yfinance(ticker: str) -> pd.DataFrame:
    """
    Load historical annual FCFF from yfinance API
    Returns DataFrame with columns: fy, Free Cash Flow, Tax Rate, Total Debt, Equity, Kd, end
    All financial values in BILLIONS ($B)
    """
    try:
        stock = yf.Ticker(ticker)
        cf = stock.cashflow.T
        bs = stock.balance_sheet.T
        inc = stock.income_stmt.T

        if cf.empty or bs.empty or inc.empty:
            logger.warning(f"Incomplete financial data from yfinance for {ticker}")
            return pd.DataFrame()

        data = []
        for date in cf.index:
            try:
                ocf = cf.loc[date, 'Operating Cash Flow'] if 'Operating Cash Flow' in cf.columns else None
                if ocf is None:
                    continue

                capex = cf.loc[date, 'Capital Expenditure'] if 'Capital Expenditure' in cf.columns else 0
                interest = inc.loc[
                    date, 'Interest Expense'] if date in inc.index and 'Interest Expense' in inc.columns else 0

                tax_expense = inc.loc[
                    date, 'Tax Provision'] if date in inc.index and 'Tax Provision' in inc.columns else 0
                pretax_income = inc.loc[
                    date, 'Pretax Income'] if date in inc.index and 'Pretax Income' in inc.columns else 1
                tax_rate = (tax_expense / pretax_income) if pretax_income != 0 else 0.21
                tax_rate = max(0, min(0.5, tax_rate))

                lt_debt = bs.loc[date, 'Long Term Debt'] if date in bs.index and 'Long Term Debt' in bs.columns else 0
                st_debt = bs.loc[date, 'Current Debt'] if date in bs.index and 'Current Debt' in bs.columns else 0
                total_debt = lt_debt + st_debt

                equity = bs.loc[
                    date, 'Stockholders Equity'] if date in bs.index and 'Stockholders Equity' in bs.columns else 0

                fcff = ocf + interest * (1 - tax_rate) - abs(capex)
                kd = interest / total_debt if total_debt > 0 else 0.05

                data.append({
                    'fy': date.year,
                    'Free Cash Flow': fcff / 1e9,  # Convert to billions
                    'Tax Rate': tax_rate,
                    'Total Debt': total_debt / 1e9,  # Convert to billions
                    'Equity': equity / 1e9,  # Convert to billions
                    'Kd': kd,
                    'end': date,
                    'form': '10-K'
                })

            except Exception:
                continue

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.sort_values('fy')

        logger.info(f"Loaded {len(df)} years from yfinance for {ticker}")
        return df

    except Exception as e:
        logger.warning(f"yfinance failed for {ticker}: {str(e)}")
        return pd.DataFrame()


def get_market_cap_from_yfinance(ticker: str) -> dict:
    """Load market capitalization from yfinance API"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap')

        if market_cap is None or not isinstance(market_cap, (int, float)):
            return None

        return {
            'ticker': ticker,
            'market_cap': market_cap,
            'date': pd.Timestamp.now().strftime('%Y-%m-%d')
        }

    except Exception as e:
        logger.warning(f"Failed to get market cap for {ticker}: {str(e)}")
        return None


# ===== FCF CALCULATION =====

def calculate_fcf(df: pd.DataFrame, form_type: str) -> pd.DataFrame:
    """
    Calculate Free Cash Flow to Firm from components
    FCFF = Operating Cash Flow + Interest × (1-T) - CapEx

    Returns data in BILLIONS ($B)
    NO DEDUPLICATION - done in merge function
    """
    if df.empty:
        return pd.DataFrame()

    df_filtered = df[df['form'] == form_type].copy()
    if df_filtered.empty:
        return pd.DataFrame()

    pivot_df = df_filtered.pivot_table(
        values='value',
        index=['end', 'form', 'fy', 'fp'],
        columns='metric',
        aggfunc='first'
    ).reset_index()

    # Operating Cash Flow
    if 'Operating Cash Flow' in pivot_df.columns:
        ocf = pivot_df['Operating Cash Flow']
    elif 'Operating Cash Flow (Continuing)' in pivot_df.columns:
        ocf = pivot_df['Operating Cash Flow (Continuing)']
    else:
        return pd.DataFrame()

    # CapEx
    capex = pivot_df.get('CapEx', 0)
    if isinstance(capex, pd.Series):
        capex = capex.fillna(0)

    # Interest Expense
    if 'Interest Expense' in pivot_df.columns:
        interest = pivot_df['Interest Expense'].fillna(0)
    elif 'Interest And Debt Expense' in pivot_df.columns:
        interest = pivot_df['Interest And Debt Expense'].fillna(0)
    else:
        interest = 0

    # Tax Rate
    if 'Income Tax Expense' in pivot_df.columns and 'EBIT' in pivot_df.columns:
        pivot_df['Tax Rate'] = (pivot_df['Income Tax Expense'] / pivot_df['EBIT']).clip(0, 0.5).fillna(0.21)
    else:
        pivot_df['Tax Rate'] = 0.21

    # FCFF calculation
    pivot_df['Free Cash Flow'] = ocf + interest * (1 - pivot_df['Tax Rate']) - abs(capex)

    # Debt components
    lt_debt = pivot_df.get('Long-Term Debt', 0)
    st_debt = pivot_df.get('Short-Term Debt', 0)
    if 'Short-Term Debt' not in pivot_df.columns and 'Current Debt' in pivot_df.columns:
        st_debt = pivot_df['Current Debt']

    if isinstance(lt_debt, pd.Series):
        lt_debt = lt_debt.fillna(0)
    if isinstance(st_debt, pd.Series):
        st_debt = st_debt.fillna(0)

    pivot_df['Total Debt'] = lt_debt + st_debt
    pivot_df['Equity'] = pivot_df.get('Shareholders Equity', 0)
    if isinstance(pivot_df['Equity'], pd.Series):
        pivot_df['Equity'] = pivot_df['Equity'].fillna(0)

    pivot_df['Cash'] = pivot_df.get('Cash', 0)
    if isinstance(pivot_df['Cash'], pd.Series):
        pivot_df['Cash'] = pivot_df['Cash'].fillna(0)

    pivot_df['Net Debt'] = pivot_df['Total Debt'] - pivot_df['Cash']

    # Kd (cost of debt)
    pivot_df['Kd'] = pivot_df.apply(
        lambda row: row['Interest Expense'] / row['Total Debt'] if 'Interest Expense' in pivot_df.columns and row[
            'Total Debt'] > 0 else 0,
        axis=1
    )

    # D/E ratio
    pivot_df['D/E'] = pivot_df.apply(
        lambda row: row['Total Debt'] / row['Equity'] if row['Equity'] > 0 else 0,
        axis=1
    )

    # Convert to billions
    pivot_df['Free Cash Flow'] = pivot_df['Free Cash Flow'] / 1e9
    pivot_df['Total Debt'] = pivot_df['Total Debt'] / 1e9
    pivot_df['Equity'] = pivot_df['Equity'] / 1e9
    pivot_df['Cash'] = pivot_df['Cash'] / 1e9
    pivot_df['Net Debt'] = pivot_df['Net Debt'] / 1e9

    # Sort by fiscal year and date
    pivot_df['end'] = pd.to_datetime(pivot_df['end'])
    pivot_df = pivot_df.sort_values(['fy', 'end']).reset_index(drop=True)

    return pivot_df


# ===== DATA MERGING =====

def merge_yfinance_and_sec_data(df_sec: pd.DataFrame, df_yf: pd.DataFrame) -> pd.DataFrame:
    """
    Hybrid merge: SEC EDGAR base + yfinance override
    DEDUPLICATION DONE HERE
    """
    if df_sec.empty:
        return df_yf
    if df_yf.empty:
        return df_sec

    # Clean NaN
    df_yf_clean = df_yf.dropna(subset=['Free Cash Flow']).copy()
    df_sec_clean = df_sec.dropna(subset=['Free Cash Flow']).copy()

    if df_yf_clean.empty:
        return df_sec_clean

    # Get years covered by yfinance
    yf_years = set(df_yf_clean['fy'].unique())

    # Keep SEC for years NOT in yfinance
    df_sec_old = df_sec_clean[~df_sec_clean['fy'].isin(yf_years)].copy()

    # Combine
    df_hybrid = pd.concat([df_sec_old, df_yf_clean], ignore_index=True)

    # DEDUPLICATION
    df_hybrid['end'] = pd.to_datetime(df_hybrid['end'])
    df_hybrid = df_hybrid.sort_values(['fy', 'end'])
    df_hybrid = df_hybrid.drop_duplicates(subset=['fy'], keep='last')
    df_hybrid = df_hybrid.reset_index(drop=True)

    logger.info(f"Merged: {len(df_sec_old)} SEC years + {len(df_yf_clean)} yfinance years = {len(df_hybrid)} total")

    return df_hybrid


# ===== MAIN EXTRACTION =====

if __name__ == "__main__":
    from src.config import load_tickers, get_company_name

    os.makedirs('data', exist_ok=True)

    # SEC EDGAR headers
    HEADERS = {
        'User-Agent': 'Your Company Name your.email@example.com'
    }

    companies = load_tickers('tickers.txt')

    print("=" * 80)
    print("FCF DATA EXTRACTION - MULTIPLE COMPANIES")
    print("=" * 80)
    print(f"Processing {len(companies)} companies...")
    print("=" * 80)

    all_company_data_10q = []
    all_company_data_10k = []
    all_market_caps = []

    for i, company in enumerate(companies, 1):
        CIK = company["CIK"]
        TICKER = company["TICKER"]
        NAME = company["NAME"]

        print(f"\n[{i}/{len(companies)}] Processing {NAME} ({TICKER}, CIK: {CIK if CIK else 'N/A - yfinance only'})...")
        print("-" * 60)

        if i > 1:
            time.sleep(0.11)

        # STEP 1: yfinance FCFF (all data in billions)
        df_yf_annual = load_fcff_from_yfinance(TICKER)

        # STEP 1b: Market cap
        market_cap_data = get_market_cap_from_yfinance(TICKER)
        if market_cap_data:
            market_cap_data['name'] = NAME  # Add name
            all_market_caps.append(market_cap_data)

        # STEP 2: SEC EDGAR (only if CIK available)
        df_sec_10q = pd.DataFrame()
        df_sec_10k = pd.DataFrame()
        company_data = None

        if CIK:
            company_data = get_company_facts(CIK, HEADERS)

            if company_data:
                print(f"✓ SEC EDGAR data retrieved")
                print(f"Company: {company_data.get('entityName', 'N/A')}")

                fcf_df = extract_fcf_data(company_data)

                if not fcf_df.empty:
                    print(f"✓ Found {len(fcf_df)} financial data records")
                    df_sec_10q = calculate_fcf(fcf_df, '10-Q')
                    df_sec_10k = calculate_fcf(fcf_df, '10-K')
                else:
                    print("⚠ No FCF data found")
            else:
                print("⚠ Error retrieving SEC EDGAR data")
        else:
            print("⚠ CIK not available - using yfinance only")

        # STEP 3: MERGE with deduplication
        df_annual_final = merge_yfinance_and_sec_data(df_sec_10k, df_yf_annual)

        # STEP 4: Quarterly (with deduplication for 10-Q)
        if not df_sec_10q.empty:
            df_sec_10q = df_sec_10q.dropna(subset=['Free Cash Flow'])
            df_sec_10q = df_sec_10q.sort_values(['fy', 'end'])
            df_sec_10q = df_sec_10q.drop_duplicates(subset=['fy'], keep='last')
            df_sec_10q = df_sec_10q.reset_index(drop=True)

        df_quarterly_final = df_sec_10q

        # STEP 5: SAVE using NAME
        if not df_quarterly_final.empty and 'Free Cash Flow' in df_quarterly_final.columns:
            print(f"✓ FCF 10-Q: {len(df_quarterly_final)} periods")
            df_quarterly_final['ticker'] = TICKER
            df_quarterly_final['name'] = NAME
            df_quarterly_final['company_name'] = company_data.get('entityName', 'N/A') if company_data else NAME
            output_file = f"data/{NAME}_fcf_10Q.csv"
            df_quarterly_final.to_csv(output_file, index=False)
            print(f"✓ 10-Q data saved: {output_file}")
            all_company_data_10q.append(df_quarterly_final)
        else:
            print("⚠ No 10-Q data available")

        if not df_annual_final.empty and 'Free Cash Flow' in df_annual_final.columns:
            print(
                f"✓ FCF 10-K: {len(df_annual_final)} periods (FY {df_annual_final['fy'].min()}-{df_annual_final['fy'].max()})")
            df_annual_final['ticker'] = TICKER
            df_annual_final['name'] = NAME
            df_annual_final['company_name'] = company_data.get('entityName', 'N/A') if company_data else NAME
            output_file = f"data/{NAME}_fcf_10K.csv"
            df_annual_final.to_csv(output_file, index=False)
            print(f"✓ 10-K data saved: {output_file}")
            all_company_data_10k.append(df_annual_final)
        else:
            print("⚠ No 10-K data available")

    # SAVE MARKET CAPS
    if all_market_caps:
        market_cap_df = pd.DataFrame(all_market_caps)
        market_cap_file = "data/market_caps.csv"
        market_cap_df.to_csv(market_cap_file, index=False)
        print(f"\n✓ Market caps saved to: {market_cap_file}")

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if all_company_data_10q or all_company_data_10k:
        print("\nLATEST FCF PER COMPANY:")
        print("-" * 60)

        for company in companies:
            NAME = company["NAME"]
            print(f"\n{NAME}:")

            if all_company_data_10q:
                combined_10q = pd.concat(all_company_data_10q, ignore_index=True)
                company_10q = combined_10q[combined_10q['name'] == NAME]
                if not company_10q.empty:
                    latest_fcf = company_10q.iloc[-1]['Free Cash Flow']  # Already in billions
                    latest_period = company_10q.iloc[-1]['fp']
                    latest_year = company_10q.iloc[-1]['fy']
                    print(f"  10-Q: ${latest_fcf:>10,.2f}B ({latest_period} FY {int(latest_year)})")

            if all_company_data_10k:
                combined_10k = pd.concat(all_company_data_10k, ignore_index=True)
                company_10k = combined_10k[combined_10k['name'] == NAME]
                if not company_10k.empty:
                    latest_fcf = company_10k.iloc[-1]['Free Cash Flow']  # Already in billions
                    latest_year = company_10k.iloc[-1]['fy']
                    print(f"  10-K: ${latest_fcf:>10,.2f}B (FY {int(latest_year)})")
    else:
        print("⚠ No data retrieved")

    print("\n" + "=" * 80)
    print("FORMULA: FCFF = Operating Cash Flow + Interest × (1-T) - CapEx")
    print("Hybrid data: yfinance (recent) + SEC EDGAR (historical)")
    print("All values in BILLIONS ($B)")
    print("Deduplication: ONCE in merge function")
    print("=" * 80)