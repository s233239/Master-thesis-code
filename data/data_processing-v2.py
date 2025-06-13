"""
Created to observe several medoids for each RES capacity factor - but without
any seasonal or weather context assigned.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

# === PARAMETERS ===
n_clusters = 6  # Number of typical daily scenarios per RES
csv_filename = "TimeSeries_2021-2024.csv"

# === LOAD CSV ===
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, csv_filename)

df = pd.read_csv(csv_path, sep=';', decimal=',', parse_dates=['HourUTC'])

# === PREPROCESSING ===
df['CF_Offshore'] = df['OffshoreWind_MWh'] / df['OffshoreWindCapacity']
df['CF_Onshore'] = df['OnshoreWind_MWh'] / df['OnshoreWindCapacity']
df['CF_Solar'] = df['SolarPower_MWh'] / df['SolarPowerCapacity']

# Check for invalid capacity factor rows
df[['CF_Offshore', 'CF_Onshore', 'CF_Solar']] = df[['CF_Offshore', 'CF_Onshore', 'CF_Solar']].replace([np.inf, -np.inf], np.nan)
invalid_rows = df[df[['CF_Offshore', 'CF_Onshore', 'CF_Solar']].isna().any(axis=1)]
if not invalid_rows.empty:
    raise ValueError(f"Found {len(invalid_rows)} rows with NaN or Inf in capacity factors. First few:\n{invalid_rows.head()}")

# Add date/hour for pivoting
df['Date'] = df['HourUTC'].dt.date
df['Hour'] = df['HourUTC'].dt.hour

# === FUNCTIONS ===
def create_daily_matrix(df, cf_col):
    pivot = df.pivot(index='Date', columns='Hour', values=cf_col)
    return pivot.dropna()  # Drop days with missing hours

def compute_k_medoids_profiles(daily_cf_matrix, label):
    model = KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=0)
    model.fit(daily_cf_matrix)

    # Extract medoids and cluster assignments
    medoids = daily_cf_matrix.iloc[model.medoid_indices_]
    labels = pd.Series(model.labels_, index=daily_cf_matrix.index, name='Cluster')

    # Plotting
    plt.figure(figsize=(10, 6))
    for i, (_, row) in enumerate(medoids.iterrows()):
        plt.plot(row.values, label=f"{label} - Scenario {i+1}")
    plt.title(f"{label} Capacity Factor – {n_clusters} Typical Daily Profiles")
    plt.xlabel("Hour of Day")
    plt.ylabel("Capacity Factor")
    plt.grid(True)
    plt.xticks(ticks=np.arange(0, 24))
    plt.legend()
    plt.tight_layout()

    # Optional: show cluster sizes
    print(f"\n{label} – Daily Profile Cluster Counts:")
    print(labels.value_counts().sort_index())

    return medoids, labels

# === DAILY MATRICES ===
daily_offshore = create_daily_matrix(df, 'CF_Offshore')
daily_onshore = create_daily_matrix(df, 'CF_Onshore')
daily_solar = create_daily_matrix(df, 'CF_Solar')

# === CLUSTERING ===
medoids_offshore, labels_offshore = compute_k_medoids_profiles(daily_offshore, "Offshore Wind")
medoids_onshore, labels_onshore = compute_k_medoids_profiles(daily_onshore, "Onshore Wind")
medoids_solar, labels_solar = compute_k_medoids_profiles(daily_solar, "Solar")

# === Optional: Export results ===
# Save medoid profiles
# medoids_offshore.to_csv('medoid_profiles_offshore.csv')
# medoids_onshore.to_csv('medoid_profiles_onshore.csv')
# medoids_solar.to_csv('medoid_profiles_solar.csv')

# Save day-to-cluster assignments
labels_df = pd.DataFrame({
    'OffshoreCluster': labels_offshore,
    'OnshoreCluster': labels_onshore,
    'SolarCluster': labels_solar,
})
labels_df.index.name = 'Date'
# labels_df.to_csv('daily_cluster_assignments.csv')

print("\nSaved medoid profiles and daily cluster assignments.")
plt.show()
