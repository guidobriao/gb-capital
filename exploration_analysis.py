# -*- coding: utf-8 -*-
"""
Exploration Analysis - Historical FCFF Plots
NO DEDUPLICATION - CSV already deduplicated by fcf_extractor.py
All data already in BILLIONS ($B)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
import os
from src.config import settings, load_tickers

# Load companies
companies = load_tickers('tickers.txt')

# Create result directory
settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8')


def plot_fcf_data(name: str, form_type: str, color: str, marker: str):
    """
    Plot FCF data for a company (10Q or 10K)
    Args:
        name: Company name (not ticker)
    """
    filename = f"data/{name}_fcf_{form_type}.csv"

    if not os.path.exists(filename):
        print(f"✗ {name}: File {filename} not found")
        return False

    df = pd.read_csv(filename)

    if df.empty:
        print(f"⚠ {name}: File {form_type} empty")
        return False

    # Convert dates and sort
    df['end'] = pd.to_datetime(df['end'])
    df = df.sort_values('end')

    # Data already in billions - NO conversion needed
    df['FCF_B'] = df['Free Cash Flow']
    df = df.dropna(subset=['end', 'FCF_B'])

    if df.empty:
        print(f"⚠ {name}: No valid data after cleaning ({form_type})")
        return False

    # Create plot
    plt.figure(figsize=(12, 6))

    # Smooth interpolation
    fy_vals = df['end'].values
    fcf_vals = df['FCF_B'].values

    if len(fy_vals) > 3:
        x_smooth = np.linspace(0, len(fy_vals) - 1, 300)
        spl = make_interp_spline(range(len(fy_vals)), fcf_vals, k=3)
        fcf_smooth = spl(x_smooth)
        x_dates_smooth = np.interp(x_smooth, range(len(fy_vals)),
                                   [pd.Timestamp(d).timestamp() for d in fy_vals])
        dates_smooth = pd.to_datetime(x_dates_smooth, unit='s')
        plt.plot(dates_smooth, fcf_smooth, linewidth=2, color=color)

    plt.plot(df['end'], df['FCF_B'], marker=marker, markersize=6 if form_type == '10-Q' else 8,
             color=color, linestyle='')

    form_label = "Quarterly" if form_type == "10Q" else "Annual"
    plt.title(f'{name} - {form_label} Free Cash Flow ({form_type})',
              fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Free Cash Flow ($B)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Add latest value
    last_value = df['FCF_B'].iloc[-1]
    last_year = int(df['fy'].iloc[-1])

    if 'fp' in df.columns and form_type == '10Q':
        last_period = df['fp'].iloc[-1]
        label_text = f'Latest: ${last_value:.1f}B ({last_period} FY{last_year})'
    else:
        label_text = f'Latest: ${last_value:.1f}B (FY{last_year})'

    box_color = 'lightblue' if form_type == '10Q' else 'lightgreen'
    plt.text(0.02, 0.98, label_text,
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round',
                                                facecolor=box_color, alpha=0.8))

    plt.tight_layout()
    plt.savefig(settings.RESULT_DIR / f'{name}_fcf_{form_type}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ {form_type} plot saved: result/{name}_fcf_{form_type}.png")
    return True


# PLOT QUARTERLY DATA (10-Q)
print("=" * 80)
print("GENERATING QUARTERLY PLOTS (10-Q)")
print("=" * 80)

for company in companies:
    name = company['NAME']
    plot_fcf_data(name, '10Q', 'steelblue', 'o')

print("\n✓ All quarterly plots (10-Q) completed")

# PLOT ANNUAL DATA (10-K)
print("\n" + "=" * 80)
print("GENERATING ANNUAL PLOTS (10-K)")
print("=" * 80)

for company in companies:
    name = company['NAME']
    plot_fcf_data(name, '10K', 'darkgreen', 's')

print("\n✓ All annual plots (10-K) completed")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETED")
print("Data source: CSV files (already in billions, already deduplicated)")
print("=" * 80)