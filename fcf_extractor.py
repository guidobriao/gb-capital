#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEC EDGAR Free Cash Flow Historical Data Extractor
Multi-Company Version
"""

import requests
import pandas as pd
import json
from datetime import datetime
import time
import os

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
    """Estrae i dati di Free Cash Flow dai dati XBRL"""
    fcf_data = []

    # Tag XBRL comuni per componenti del FCF
    fcf_tags = {
        'NetCashProvidedByUsedInOperatingActivities': 'Operating Cash Flow',
        'PaymentsToAcquirePropertyPlantAndEquipment': 'CapEx',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations': 'Operating Cash Flow (Continuing)',
        'PaymentsForCapitalImprovements': 'Capital Improvements',
        'PaymentsToAcquireProductiveAssets': 'Productive Assets'
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

def calculate_fcf(df):
    """Calcola il Free Cash Flow dai componenti"""
    if df.empty:
        return pd.DataFrame()

    # Filtra solo i dati dai 10-K e 10-Q
    df_filtered = df[df['form'].isin(['10-K', '10-Q'])].copy()

    # Pivot per avere metriche come colonne
    pivot_df = df_filtered.pivot_table(
        values='value',
        index=['end', 'form', 'fy', 'fp'],
        columns='metric',
        aggfunc='first'
    ).reset_index()

    # Calcola FCF (Operating Cash Flow - CapEx)
    if 'Operating Cash Flow' in pivot_df.columns and 'CapEx' in pivot_df.columns:
        # CapEx è riportato come valore positivo ma rappresenta un'uscita
        pivot_df['Free Cash Flow'] = pivot_df['Operating Cash Flow'] - abs(pivot_df['CapEx'])
    elif 'Operating Cash Flow (Continuing)' in pivot_df.columns and 'CapEx' in pivot_df.columns:
        pivot_df['Free Cash Flow'] = pivot_df['Operating Cash Flow (Continuing)'] - abs(pivot_df['CapEx'])

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

all_company_data = []

for i, company in enumerate(COMPANIES, 1):
    CIK = company["CIK"]
    TICKER = company["TICKER"]

    print(f"\n[{i}/{len(COMPANIES)}] Recupero dati per {TICKER} (CIK: {CIK})...")
    print("-" * 60)

    # Rate limiting - pausa per rispettare i limiti API
    if i > 1:
        time.sleep(0.11)  # Pausa di 110ms tra le richieste

    # Recupera i dati
    company_data = get_company_facts(CIK)

    if company_data:
        print(f"✓ Dati recuperati con successo")
        print(f"Azienda: {company_data.get('entityName', 'N/A')}")

        # Estrai dati FCF
        fcf_df = extract_fcf_data(company_data)

        if not fcf_df.empty:
            print(f"✓ Trovati {len(fcf_df)} record di dati finanziari")

            # Calcola FCF
            fcf_summary = calculate_fcf(fcf_df)

            if not fcf_summary.empty and 'Free Cash Flow' in fcf_summary.columns:
                print("✓ FCF calcolato con successo")

                # Aggiungi ticker alla tabella
                fcf_summary['ticker'] = TICKER
                fcf_summary['company_name'] = company_data.get('entityName', 'N/A')

                # Mostra gli ultimi 5 periodi per questa azienda
                print("\nUltimi 5 periodi:")
                display_df = fcf_summary[['end', 'form', 'fy', 'fp', 'Operating Cash Flow', 'CapEx', 'Free Cash Flow']].tail(5).copy()

                # Formatta i valori in milioni
                for col in ['Operating Cash Flow', 'CapEx', 'Free Cash Flow']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col] / 1_000_000

                display_df.columns = ['Data', 'Form', 'Anno Fiscale', 'Periodo', 'Op. CF ($M)', 'CapEx ($M)', 'FCF ($M)']

                pd.set_option('display.float_format', '{:,.0f}'.format)
                print(display_df.to_string(index=False))

                all_company_data.append(fcf_summary)

                # Esporta in CSV individuale nella cartella data
                output_file = f"data/{TICKER}_fcf_history.csv"
                fcf_summary.to_csv(output_file, index=False)
                print(f"✓ Dati esportati in: {output_file}")

            else:
                print("⚠ Impossibile calcolare FCF - dati mancanti")
        else:
            print("⚠ Nessun dato FCF trovato")
    else:
        print("✗ Errore nel recupero dei dati")

# Consolidamento finale
print("\n" + "=" * 80)
print("RIEPILOGO FINALE")
print("=" * 80)

if all_company_data:
    # Combina tutti i dati
    combined_df = pd.concat(all_company_data, ignore_index=True)

    # Esporta file consolidato nella cartella data
    consolidated_file = "data/ALL_COMPANIES_fcf_history.csv"
    combined_df.to_csv(consolidated_file, index=False)
    print(f"✓ File consolidato esportato: {consolidated_file}")

    # Statistiche riassuntive per tutte le aziende
    print(f"\nAziende processate con successo: {len(all_company_data)}")
    print(f"Record totali: {len(combined_df)}")

    print("\nULTIMO FCF ANNUALE PER AZIENDA (valori in $M):")
    print("-" * 60)

    for ticker in [comp["TICKER"] for comp in COMPANIES]:
        company_data = combined_df[combined_df['ticker'] == ticker]
        if not company_data.empty:
            annual_data = company_data[company_data['form'] == '10-K']
            if not annual_data.empty:
                latest_fcf = annual_data.iloc[-1]['Free Cash Flow'] / 1_000_000
                latest_year = annual_data.iloc[-1]['fy']
                print(f"{ticker:>6}: ${latest_fcf:>10,.0f}M (FY {latest_year})")
            else:
                print(f"{ticker:>6}: Dati annuali non disponibili")
        else:
            print(f"{ticker:>6}: Nessun dato disponibile")
else:
    print("⚠ Nessun dato è stato recuperato con successo")

print("\n" + "=" * 80)
print("NOTA: I valori di CapEx sono mostrati come positivi per chiarezza,")
print("ma rappresentano uscite di cassa nel calcolo del FCF.")
print("\nIMPORTANTE: Ricorda di modificare l'User-Agent nell'header con i tuoi dati!")