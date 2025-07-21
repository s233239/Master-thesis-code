"""
Filters clusters by season to get separate k-medoid (k>=1) profiles per season
for each RES.
Season = ['Winter', 'Summer']

Plots these season profiles for each RES capacity factor.
Exports results as csv (optional): 
    - k medoids profiles for each RES, season
    - day-to-cluster assignments: index is the date, column values are the
    assigned cluster for each RES
    - cluster diagnostics for each: probability, distance max, avg, standard deviation
This enables to observe and analyse the cluster mapping over time. 
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import cdist


# === PARAMETERS ===
n_clusters = 3  # Number of typical daily scenarios per RES per season
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
    if month in [11, 12, 1, 2]:
        return "Winter"
    elif month in [5, 6, 7, 8]:
        return "Summer"
    else:
        return "NA"

df['Season'] = df['Month'].apply(get_season)

# === FUNCTION: Create daily matrix ===
def create_daily_matrix(df, cf_col):
    pivot = df.pivot(index='Date', columns='Hour', values=cf_col)
    return pivot.dropna()

# === FUNCTION: DIAGNOSTIC OF CLUSTERS ===
def cluster_diagnostic(daily_matrix, model, labels):
    cluster_diagnostics = []

    # Loop over clusters
    for cluster_id in np.unique(model.labels_):
        # Select all days in this cluster
        cluster_days = daily_matrix[labels == cluster_id]
        
        # Get the medoid profile (as a 1D array)
        medoid_row = daily_matrix.iloc[model.medoid_indices_[cluster_id]].values.reshape(1, -1)
        
        # Compute distances from all cluster members to the medoid
        distances = cdist(cluster_days.values, medoid_row, metric='euclidean').flatten()
        
        # Save diagnostics
        cluster_diagnostics.append({
            'Cluster': cluster_id,
            'Medoid Date': daily_matrix.index[model.medoid_indices_[cluster_id]],
            'Count': len(distances),
            'Proportion': round(len(distances)/len(daily_matrix),2), 
            'Max Distance': distances.max(),
            'Avg Distance': distances.mean(),
            'Standard Deviation': distances.std(),
        })

    # Turn into a DataFrame
    diagnostics_df = pd.DataFrame(cluster_diagnostics).set_index('Cluster')
    
    return diagnostics_df

# === FUNCTION: Run clustering and plot ===
def cluster_by_season(daily_matrix, label, res_tag):
    medoids_profile_summary = pd.DataFrame()
    cluster_labels_summary = pd.DataFrame()
    cluster_diagnostics_summary = pd.DataFrame()

    for season in ['Winter', 'Summer']:
        index_labels = [f"{label} - {season} {i}" for i in range(n_clusters)]
        season_dates = df[df['Season'] == season]['Date'].unique()
        seasonal_matrix = daily_matrix.loc[daily_matrix.index.isin(season_dates)]

        model = KMedoids(n_clusters=n_clusters, metric='euclidean', random_state=0)
        model.fit(seasonal_matrix)

        medoids_profile = seasonal_matrix.iloc[model.medoid_indices_]
        cluster_labels = pd.Series(model.labels_, index=seasonal_matrix.index, name='Cluster')

        # Plot
        linestyle = "solid" if season=="Winter" else "dashed"
        for i, (idx, row) in enumerate(medoids_profile.iterrows()):
            plt.plot(row.values, label=f"{season} {i+1} ({idx})", linestyle=linestyle)

        # Diagnostic of KMedoids model
        seasonal_diagnostic_df = cluster_diagnostic(seasonal_matrix, model, cluster_labels)
        seasonal_diagnostic_df.index = index_labels
        print(f"\n{label} - {season} Daily Profile Cluster Characteristics:\n{seasonal_diagnostic_df}")
        if n_clusters > 1:
            print(f"Silhouette Score: {silhouette_score(seasonal_matrix.values, model.labels_, metric='euclidean')}")

        # Export results
        medoids_profile.reset_index(inplace=True)
        medoids_profile.insert(0, 'Cluster', [i for i in range(n_clusters)])
        medoids_profile.insert(0, 'Season', season)
        medoids_profile.insert(0, 'RES', label)
        medoids_profile_summary = pd.concat([medoids_profile_summary, medoids_profile], axis=0)
        
        cluster_labels = pd.DataFrame(cluster_labels)
        cluster_labels.insert(0, 'RES', label)
        cluster_labels.insert(0, 'Season', season)
        cluster_labels_summary = pd.concat([cluster_labels_summary, cluster_labels], axis=0)

        cluster_diagnostics_summary = pd.concat([cluster_diagnostics_summary, seasonal_diagnostic_df], axis=0)


    # Plotting for each RES
    plt.title(f"{label} Capacity Factor - {n_clusters} Typical Daily Profiles (2021-2024)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Capacity Factor")
    plt.grid(False)
    plt.xticks(np.arange(0, 24, 2))
    plt.ylim(top=1)
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Export result
    cluster_labels_summary.index = pd.to_datetime(cluster_labels_summary.index)
    cluster_labels_summary.sort_index(inplace=True)

    return medoids_profile_summary, cluster_labels_summary, cluster_diagnostics_summary


# === CREATE DAILY MATRICES ===
daily_offshore = create_daily_matrix(df, 'CF_Offshore')
daily_onshore = create_daily_matrix(df, 'CF_Onshore')
daily_solar = create_daily_matrix(df, 'CF_Solar')

# === CLUSTER PER RES AND SEASON ===
plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
offshore_medoids, offshore_clusters, offshore_cluster_diagnostic = cluster_by_season(daily_offshore, "Offshore Wind", "offshore")
plt.subplot(2,2,2)
onshore_medoids, onshore_clusters, onshore_cluster_diagnostic = cluster_by_season(daily_onshore, "Onshore Wind", "onshore")
plt.subplot(2,2,3)
solar_medoids, solar_clusters, solar_cluster_diagnostic = cluster_by_season(daily_solar, "Solar", "solar")

# Save results
medoids_profile_summary = pd.concat([offshore_medoids, onshore_medoids, solar_medoids], axis=0)
medoids_profile_summary.reset_index(drop=True)
medoids_profile_summary.to_csv(f'medoids_profile_summary--{n_clusters}cluster.csv', index=False)

cluster_labels_summary = pd.concat([offshore_clusters, 
                                    onshore_clusters.drop(columns=['Season']), 
                                    solar_clusters.drop(columns=['Season'])], 
                                    axis=1)
cluster_labels_summary.to_csv(f'clusters_assignment_summary--{n_clusters}cluster.csv')

clusters_diagnostic_summary = pd.concat([offshore_cluster_diagnostic, onshore_cluster_diagnostic, solar_cluster_diagnostic], axis=0)
clusters_diagnostic_summary.index.name = 'Cluster'
clusters_diagnostic_summary.to_csv(f'clusters_diagnostic_summary--{n_clusters}cluster.csv')

plt.savefig(fname=os.path.join(script_dir, "plots", f"vf--{n_clusters}cluster.png"), format="png")
plt.show()
