"""
Filters clusters by season to get separate medoid profiles per season for each RES.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

# === PARAMETERS ===
n_clusters = 1
csv_filename = "TimeSeries_2021-2024.csv"

# === LOAD CSV ===
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, csv_filename)

df = pd.read_csv(csv_path, sep=';', decimal=',', parse_dates=['HourUTC'])

# === PREPROCESS ===
df['CF_Offshore'] = df['OffshoreWind_MWh'] / df['OffshoreWindCapacity']
df['CF_Onshore'] = df['OnshoreWind_MWh'] / df['OnshoreWindCapacity']
df['CF_Solar'] = df['SolarPower_MWh'] / df['SolarPowerCapacity']

# Check for invalid rows
df[['CF_Offshore', 'CF_Onshore', 'CF_Solar']] = df[['CF_Offshore', 'CF_Onshore', 'CF_Solar']].replace([np.inf, -np.inf], np.nan)
invalid_rows = df[df[['CF_Offshore', 'CF_Onshore', 'CF_Solar']].isna().any(axis=1)]
if not invalid_rows.empty:
    raise ValueError(f"Found {len(invalid_rows)} rows with NaN or Inf in capacity factors. First few:\n{invalid_rows.head()}")

# Add time features
df['Date'] = df['HourUTC'].dt.date
df['Hour'] = df['HourUTC'].dt.hour
df['Month'] = df['HourUTC'].dt.month

# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df['Season'] = df['Month'].apply(get_season)

# === FUNCTION: Create daily matrix ===
def create_daily_matrix(df, cf_col):
    pivot = df.pivot(index='Date', columns='Hour', values=cf_col)
    return pivot.dropna()

# === FUNCTION: Run clustering and plot ===
def cluster_by_season(daily_matrix, label, res_tag):
    medoid_summary = []
    
    plt.figure(figsize=(15,8))
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        season_dates = df[df['Season'] == season]['Date'].unique()
        seasonal_matrix = daily_matrix.loc[daily_matrix.index.isin(season_dates)]

        if len(seasonal_matrix) < n_clusters:
            print(f"Skipping {label} {season} (not enough days)")
            continue
        
        model = KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=0)
        model.fit(seasonal_matrix)

        medoids = seasonal_matrix.iloc[model.medoid_indices_]
        cluster_labels = pd.Series(model.labels_, index=seasonal_matrix.index, name='Cluster')

        # Plot
        ind = ['Winter', 'Spring', 'Summer', 'Autumn'].index(season)
        plt.subplot(2,2,ind+1)
        for i, (_, row) in enumerate(medoids.iterrows()):
            plt.plot(row.values, label=f"Scenario {i+1}")
        plt.title(f"{label} – {season} – {n_clusters} Typical Days")
        plt.xlabel("Hour of Day")
        plt.ylabel("Capacity Factor")
        plt.grid(True)
        plt.xticks(np.arange(0, 24))
        plt.legend()
        plt.tight_layout()

        # Save results
        # medoids.to_csv(f'medoid_profiles_{res_tag}_{season}.csv')
        # cluster_labels.to_csv(f'cluster_assignments_{res_tag}_{season}.csv')
        print(f"\n{label} – {season} cluster counts:\n{cluster_labels.value_counts().sort_index()}")
        medoid_summary.append((season, medoids))

    return medoid_summary

# === CREATE DAILY MATRICES ===
daily_offshore = create_daily_matrix(df, 'CF_Offshore')
daily_onshore = create_daily_matrix(df, 'CF_Onshore')
daily_solar = create_daily_matrix(df, 'CF_Solar')

# === CLUSTER PER RES AND SEASON ===
offshore_medoids = cluster_by_season(daily_offshore, "Offshore Wind", "offshore")
onshore_medoids = cluster_by_season(daily_onshore, "Onshore Wind", "onshore")
solar_medoids = cluster_by_season(daily_solar, "Solar", "solar")

plt.show()
