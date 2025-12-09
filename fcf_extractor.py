#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEC EDGAR Free Cash Flow Historical Data Extractor
Multi-Company Version - Separate 10-Q and 10-K datasets
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


def calculate_fcf(df, form_type):
    """Calcola il Free Cash Flow dai componenti per uno specifico tipo di form"""
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

    # Sostituisci NaN in CapEx con 0
    if 'CapEx' in pivot_df.columns:
        pivot_df['CapEx'] = pivot_df['CapEx'].fillna(0)

    # Calcola FCF (Operating Cash Flow - CapEx)
    if 'Operating Cash Flow' in pivot_df.columns:
        pivot_df['Free Cash Flow'] = pivot_df['Operating Cash Flow'] - abs(pivot_df['CapEx'])
    elif 'Operating Cash Flow (Continuing)' in pivot_df.columns:
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

    # Recupera i dati
    company_data = get_company_facts(CIK)

    if company_data:
        print(f"✓ Dati recuperati con successo")
        print(f"Azienda: {company_data.get('entityName', 'N/A')}")

        # Estrai dati FCF
        fcf_df = extract_fcf_data(company_data)

        if not fcf_df.empty:
            print(f"✓ Trovati {len(fcf_df)} record di dati finanziari")

            # Calcola FCF per 10-Q (trimestrali)
            fcf_10q = calculate_fcf(fcf_df, '10-Q')

            # Calcola FCF per 10-K (annuali)
            fcf_10k = calculate_fcf(fcf_df, '10-K')

            # Processa dati 10-Q
            if not fcf_10q.empty and 'Free Cash Flow' in fcf_10q.columns:
                print(f"✓ FCF 10-Q calcolato con successo ({len(fcf_10q)} periodi)")

                # Aggiungi ticker alla tabella
                fcf_10q['ticker'] = TICKER
                fcf_10q['company_name'] = company_data.get('entityName', 'N/A')

                all_company_data_10q.append(fcf_10q)

                # Esporta in CSV individuale nella cartella data
                output_file = f"data/{TICKER}_fcf_10Q.csv"
                fcf_10q.to_csv(output_file, index=False)
                print(f"✓ Dati 10-Q esportati in: {output_file}")
            else:
                print("⚠ Nessun dato 10-Q disponibile")

            # Processa dati 10-K
            if not fcf_10k.empty and 'Free Cash Flow' in fcf_10k.columns:
                print(f"✓ FCF 10-K calcolato con successo ({len(fcf_10k)} periodi)")

                # Aggiungi ticker alla tabella
                fcf_10k['ticker'] = TICKER
                fcf_10k['company_name'] = company_data.get('entityName', 'N/A')

                all_company_data_10k.append(fcf_10k)

                # Esporta in CSV individuale nella cartella data
                output_file = f"data/{TICKER}_fcf_10K.csv"
                fcf_10k.to_csv(output_file, index=False)
                print(f"✓ Dati 10-K esportati in: {output_file}")
            else:
                print("⚠ Nessun dato 10-K disponibile")

            # Mostra gli ultimi 5 periodi combinati
            if (not fcf_10q.empty and 'Free Cash Flow' in fcf_10q.columns) or \
                    (not fcf_10k.empty and 'Free Cash Flow' in fcf_10k.columns):
                print("\nUltimi 5 periodi (10-Q e 10-K combinati):")

                combined_display = []
                if not fcf_10q.empty and 'Free Cash Flow' in fcf_10q.columns:
                    combined_display.append(
                        fcf_10q[['end', 'form', 'fy', 'fp', 'Operating Cash Flow', 'CapEx', 'Free Cash Flow']])
                if not fcf_10k.empty and 'Free Cash Flow' in fcf_10k.columns:
                    combined_display.append(
                        fcf_10k[['end', 'form', 'fy', 'fp', 'Operating Cash Flow', 'CapEx', 'Free Cash Flow']])

                if combined_display:
                    display_df = pd.concat(combined_display).sort_values('end').tail(5).copy()

                    # Formatta i valori in milioni
                    for col in ['Operating Cash Flow', 'CapEx', 'Free Cash Flow']:
                        if col in display_df.columns:
                            display_df[col] = display_df[col] / 1_000_000

                    display_df.columns = ['Data', 'Form', 'Anno Fiscale', 'Periodo', 'Op. CF ($M)', 'CapEx ($M)',
                                          'FCF ($M)']

                    pd.set_option('display.float_format', '{:,.0f}'.format)
                    print(display_df.to_string(index=False))

        else:
            print("⚠ Nessun dato FCF trovato")
    else:
        print("✗ Errore nel recupero dei dati")

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
print("NOTA: I valori di CapEx sono mostrati come positivi per chiarezza,")
print("ma rappresentano uscite di cassa nel calcolo del FCF.")
print("\nIMPORTANTE: Ricorda di modificare l'User-Agent nell'header con i tuoi dati!")