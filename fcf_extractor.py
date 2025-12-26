#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEC EDGAR Free Cash Flow Historical Data Extractor
Multi-Company Version - Separate 10-Q and 10-K datasets
FCFF = Operating Cash Flow + Interest × (1-T) - CapEx
"""

import requests
import pandas as pd
import json
from datetime import datetime
import time
import os


def load_fcff_from_yfinance(ticker: str) -> pd.DataFrame:
    """
    Carica FCFF storici annuali da yfinance API
    Returns DataFrame con colonne: fy, Free Cash Flow, Tax Rate, Total Debt, Equity, Kd, end
    Returns empty DataFrame se dati non disponibili
    """
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)

        # Get financial statements
        cf = stock.cashflow  # Cash flow statement
        bs = stock.balance_sheet  # Balance sheet
        inc = stock.income_stmt  # Income statement

        if cf.empty or bs.empty or inc.empty:
            print(f"⚠ Incomplete financial data from yfinance for {ticker}")
            return pd.DataFrame()

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
                tax_rate = tax_expense / pretax_income if pretax_income != 0 else 0.21
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
                    'end': date,
                    'form': '10-K'  # Mark as annual data
                })

            except Exception as e:
                continue

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.sort_values('fy')

        print(f"✓ Loaded {len(df)} years from yfinance for {ticker} (FY {df['fy'].min()}-{df['fy'].max()})")
        return df

    except Exception as e:
        print(f"⚠ yfinance failed for {ticker}: {str(e)}")
        return pd.DataFrame()


def merge_yfinance_and_sec_data(df_yf: pd.DataFrame, df_sec: pd.DataFrame) -> pd.DataFrame:
    """
    Merge yfinance e SEC EDGAR data
    Priority: yfinance sovrascrive SEC EDGAR per gli anni disponibili
    SEC EDGAR riempie gli anni più vecchi non disponibili in yfinance
    """
    if df_yf.empty:
        return df_sec
    if df_sec.empty:
        return df_yf

    # Get years from both sources
    yf_years = set(df_yf['fy'].unique())

    # Keep SEC EDGAR data ONLY for years NOT in yfinance
    df_sec_old = df_sec[~df_sec['fy'].isin(yf_years)].copy()

    # Combine: SEC old years + yfinance recent years
    df_merged = pd.concat([df_sec_old, df_yf], ignore_index=True)
    df_merged = df_merged.sort_values('fy').reset_index(drop=True)

    print(
        f"  Merged: {len(df_sec_old)} years from SEC EDGAR + {len(df_yf)} years from yfinance = {len(df_merged)} total")

    return df_merged

# Leggi la lista dei ticker dal file
def load_tickers(filename='tickers.txt'):
    """Carica i ticker e CIK dal file di configurazione"""
    companies = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) == 2:
                    cik, ticker = parts
                    companies.append({"CIK": cik.strip(), "TICKER": ticker.strip()})
    return companies


COMPANIES = load_tickers('tickers.txt')

HEADERS = {
    'User-Agent': 'Your Company Name your.email@example.com'  # IMPORTANTE: Modificare con i tuoi dati
}


def get_company_facts(cik):
    """Recupera i dati XBRL dall'API CompanyFacts di SEC EDGAR"""
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Errore: {response.status_code}")
        return None


def extract_fcf_data(company_data):
    """Estrae i dati di Free Cash Flow e componenti WACC dai dati XBRL"""
    fcf_data = []

    # Tag XBRL comuni per componenti del FCFF e WACC
    fcf_tags = {
        # Per FCFF
        'NetCashProvidedByUsedInOperatingActivities': 'Operating Cash Flow',
        'PaymentsToAcquirePropertyPlantAndEquipment': 'CapEx',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations': 'Operating Cash Flow (Continuing)',
        'PaymentsForCapitalImprovements': 'Capital Improvements',
        'PaymentsToAcquireProductiveAssets': 'Productive Assets',

        # Per calcolo Tax e Interest
        'InterestExpense': 'Interest Expense',
        'InterestAndDebtExpense': 'Interest And Debt Expense',
        'IncomeTaxExpenseBenefit': 'Income Tax Expense',
        'IncomeTaxesPaid': 'Income Taxes Paid',
        'OperatingIncomeLoss': 'EBIT',

        # Per WACC components
        'LongTermDebt': 'Long-Term Debt',
        'ShortTermBorrowings': 'Short-Term Debt',
        'DebtCurrent': 'Current Debt',
        'StockholdersEquity': 'Shareholders Equity',
        'CashAndCashEquivalentsAtCarryingValue': 'Cash',

        # Depreciation (per info)
        'DepreciationDepletionAndAmortization': 'D&A',
    }

    facts = company_data.get('facts', {})

    # Cerca nei GAAP facts
    if 'us-gaap' in facts:
        for tag, description in fcf_tags.items():
            if tag in facts['us-gaap']:
                units = facts['us-gaap'][tag].get('units', {})

                # Prendi i valori in USD
                if 'USD' in units:
                    for item in units['USD']:
                        if 'frame' in item:  # Dati periodici
                            fcf_data.append({
                                'metric': description,
                                'tag': tag,
                                'value': item.get('val', 0),
                                'period': item.get('frame', ''),
                                'form': item.get('form', ''),
                                'filed': item.get('filed', ''),
                                'fy': item.get('fy', ''),
                                'fp': item.get('fp', ''),
                                'start': item.get('start', ''),
                                'end': item.get('end', '')
                            })

    return pd.DataFrame(fcf_data)


def calculate_fcf(df, form_type):
    """
    Calcola il Free Cash Flow to Firm dai componenti per uno specifico tipo di form
    FCFF = Operating Cash Flow + Interest × (1-T) - CapEx
    """
    if df.empty:
        return pd.DataFrame()

    # Filtra solo i dati del form type specificato
    df_filtered = df[df['form'] == form_type].copy()

    if df_filtered.empty:
        return pd.DataFrame()

    # Pivot per avere metriche come colonne
    pivot_df = df_filtered.pivot_table(
        values='value',
        index=['end', 'form', 'fy', 'fp'],
        columns='metric',
        aggfunc='first'
    ).reset_index()

    # ===== Gestione Operating Cash Flow =====
    if 'Operating Cash Flow' in pivot_df.columns:
        pivot_df['Operating Cash Flow'] = pivot_df['Operating Cash Flow']
    elif 'Operating Cash Flow (Continuing)' in pivot_df.columns:
        pivot_df['Operating Cash Flow'] = pivot_df['Operating Cash Flow (Continuing)']
    else:
        # Nessun Operating Cash Flow disponibile
        return pd.DataFrame()

    # ===== Gestione CapEx =====
    if 'CapEx' in pivot_df.columns:
        pivot_df['CapEx'] = pivot_df['CapEx'].fillna(0)
    else:
        pivot_df['CapEx'] = 0

    # ===== Gestione Interest Expense =====
    if 'Interest Expense' in pivot_df.columns:
        pivot_df['Interest Expense'] = pivot_df['Interest Expense'].fillna(0)
    elif 'Interest And Debt Expense' in pivot_df.columns:
        pivot_df['Interest Expense'] = pivot_df['Interest And Debt Expense'].fillna(0)
    else:
        pivot_df['Interest Expense'] = 0

    # ===== Gestione Tax Rate =====
    # Calcola Tax Rate da Income Tax Expense / EBIT
    if 'Income Tax Expense' in pivot_df.columns and 'EBIT' in pivot_df.columns:
        pivot_df['Tax Rate'] = pivot_df['Income Tax Expense'] / pivot_df['EBIT']
        # Clippa a range ragionevole [0, 0.5]
        pivot_df['Tax Rate'] = pivot_df['Tax Rate'].clip(0, 0.5).fillna(0.21)
    else:
        pivot_df['Tax Rate'] = 0.21  # Default US corporate tax rate

    # ===== Calcola FCFF =====
    # FCFF = Operating Cash Flow + Interest × (1-T) - CapEx
    pivot_df['Free Cash Flow'] = (
            pivot_df['Operating Cash Flow'] +
            pivot_df['Interest Expense'] * (1 - pivot_df['Tax Rate']) -
            abs(pivot_df['CapEx'])
    )

    # ===== Componenti WACC =====

    # Total Debt - usa check esplicito per evitare .fillna() su int
    lt_debt = 0
    if 'Long-Term Debt' in pivot_df.columns:
        lt_debt = pivot_df['Long-Term Debt'].fillna(0)

    st_debt = 0
    if 'Short-Term Debt' in pivot_df.columns:
        st_debt = pivot_df['Short-Term Debt'].fillna(0)
    elif 'Current Debt' in pivot_df.columns:
        st_debt = pivot_df['Current Debt'].fillna(0)

    pivot_df['Total Debt'] = lt_debt + st_debt

    # Equity
    if 'Shareholders Equity' in pivot_df.columns:
        pivot_df['Equity'] = pivot_df['Shareholders Equity'].fillna(0)
    else:
        pivot_df['Equity'] = 0

    # Cash
    if 'Cash' in pivot_df.columns:
        pivot_df['Cash'] = pivot_df['Cash'].fillna(0)
    else:
        pivot_df['Cash'] = 0

    # Net Debt
    pivot_df['Net Debt'] = pivot_df['Total Debt'] - pivot_df['Cash']

    # Cost of Debt (Kd) = Interest Expense / Total Debt
    pivot_df['Kd'] = pivot_df.apply(
        lambda row: row['Interest Expense'] / row['Total Debt'] if row['Total Debt'] > 0 else 0,
        axis=1
    )

    # D/E ratio
    pivot_df['D/E'] = pivot_df.apply(
        lambda row: row['Total Debt'] / row['Equity'] if row['Equity'] > 0 else 0,
        axis=1
    )

    # Ordina per data
    pivot_df['end'] = pd.to_datetime(pivot_df['end'])
    pivot_df = pivot_df.sort_values('end')

    return pivot_df


# Crea la cartella data se non esiste
os.makedirs('data', exist_ok=True)

# Esecuzione principale
print("ESTRAZIONE DATI FCF - MULTIPLE COMPANIES")
print("=" * 80)
print(f"Processando {len(COMPANIES)} aziende...")
print("=" * 80)

all_company_data_10q = []
all_company_data_10k = []

for i, company in enumerate(COMPANIES, 1):
    CIK = company["CIK"]
    TICKER = company["TICKER"]

    print(f"\n[{i}/{len(COMPANIES)}] Recupero dati per {TICKER} (CIK: {CIK})...")
    print("-" * 60)

    # Rate limiting - pausa per rispettare i limiti API
    if i > 1:
        time.sleep(0.11)  # Pausa di 110ms tra le richieste

    # ===== STEP 1: PROVA YFINANCE PRIMA =====
    df_yf_annual = load_fcff_from_yfinance(TICKER)

    # ===== STEP 2: RECUPERA DATI DA SEC EDGAR =====
    company_data = get_company_facts(CIK)

    df_sec_10q = pd.DataFrame()
    df_sec_10k = pd.DataFrame()

    if company_data:
        print(f"✓ Dati recuperati con successo")
        print(f"Azienda: {company_data.get('entityName', 'N/A')}")

        # Estrai dati FCF
        fcf_df = extract_fcf_data(company_data)

        if not fcf_df.empty:
            print(f"✓ Trovati {len(fcf_df)} record di dati finanziari")

            # Calcola FCF per 10-Q (trimestrali) - SEMPRE DA SEC EDGAR
            df_sec_10q = calculate_fcf(fcf_df, '10-Q')

            # Calcola FCF per 10-K (annuali) - DA SEC EDGAR
            df_sec_10k = calculate_fcf(fcf_df, '10-K')
        else:
            print("⚠ Nessun dato FCF trovato")
    else:
        print("⚠ Errore nel recupero dei dati SEC EDGAR")

    # ===== STEP 3: MERGE ANNUAL DATA (yfinance + SEC EDGAR) =====
    df_annual_final = merge_yfinance_and_sec_data(df_yf_annual, df_sec_10k)

    # ===== STEP 4: QUARTERLY SEMPRE DA SEC EDGAR (yfinance non li ha) =====
    df_quarterly_final = df_sec_10q

    # ===== STEP 5: SALVA CSV =====
    # Processa dati 10-Q
    if not df_quarterly_final.empty and 'Free Cash Flow' in df_quarterly_final.columns:
        print(f"✓ FCF 10-Q: {len(df_quarterly_final)} periodi")
        df_quarterly_final['ticker'] = TICKER
        df_quarterly_final['company_name'] = company_data.get('entityName', 'N/A') if company_data else TICKER
        output_file = f"data/{TICKER}_fcf_10Q.csv"
        df_quarterly_final.to_csv(output_file, index=False)
        print(f"✓ Dati 10-Q esportati in: {output_file}")
    else:
        print("⚠ Nessun dato 10-Q disponibile")

    # Processa dati 10-K
    if not df_annual_final.empty and 'Free Cash Flow' in df_annual_final.columns:
        print(f"✓ FCF 10-K: {len(df_annual_final)} periodi (FY {df_annual_final['fy'].min()}-{df_annual_final['fy'].max()})")
        df_annual_final['ticker'] = TICKER
        df_annual_final['company_name'] = company_data.get('entityName', 'N/A') if company_data else TICKER
        output_file = f"data/{TICKER}_fcf_10K.csv"
        df_annual_final.to_csv(output_file, index=False)
        print(f"✓ Dati 10-K esportati in: {output_file}")
    else:
        print("⚠ Nessun dato 10-K disponibile")

# Consolidamento finale
print("\n" + "=" * 80)
print("RIEPILOGO FINALE")
print("=" * 80)

# Statistiche riassuntive
if all_company_data_10q or all_company_data_10k:
    print("\nULTIMO FCF PER AZIENDA:")
    print("-" * 60)

    for ticker in [comp["TICKER"] for comp in COMPANIES]:
        print(f"\n{ticker}:")

        # 10-Q
        if all_company_data_10q:
            combined_10q = pd.concat(all_company_data_10q, ignore_index=True)
            company_10q = combined_10q[combined_10q['ticker'] == ticker]
            if not company_10q.empty:
                latest_fcf = company_10q.iloc[-1]['Free Cash Flow'] / 1_000_000
                latest_period = company_10q.iloc[-1]['fp']
                latest_year = company_10q.iloc[-1]['fy']
                print(f"  10-Q: ${latest_fcf:>10,.0f}M ({latest_period} FY {int(latest_year)})")
            else:
                print(f"  10-Q: Dati non disponibili")

        # 10-K
        if all_company_data_10k:
            combined_10k = pd.concat(all_company_data_10k, ignore_index=True)
            company_10k = combined_10k[combined_10k['ticker'] == ticker]
            if not company_10k.empty:
                latest_fcf = company_10k.iloc[-1]['Free Cash Flow'] / 1_000_000
                latest_year = company_10k.iloc[-1]['fy']
                print(f"  10-K: ${latest_fcf:>10,.0f}M (FY {int(latest_year)})")
            else:
                print(f"  10-K: Dati non disponibili")
else:
    print("⚠ Nessun dato è stato recuperato con successo")

print("\n" + "=" * 80)
print("FORMULA: FCFF = Operating Cash Flow + Interest × (1-T) - CapEx")
print("Both 10-K (annual) and 10-Q (quarterly) extracted")
print("Additional WACC components included: Total Debt, Equity, Kd, D/E")
print("\nIMPORTANTE: Ricorda di modificare l'User-Agent nell'header con i tuoi dati!")
print("=" * 80)