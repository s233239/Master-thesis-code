"""
Most basic version of RES data processing. 
Creates one daily medoid for each RES capacity factor (hourly evolution).
"""

import os
import pandas as pd
import numpy as np
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# === PARAMETERS ===
n_clusters = 1  # Number of typical daily scenarios per RES
csv_filename = "TimeSeries_2021-2024.csv"


# === LOAD CSV ===
# Get directory where the script is located
script_dir = os.path.dirname(__file__)

# Build relative path to CSV file in the same folder
csv_path = os.path.join(script_dir, csv_filename)

df = pd.read_csv(csv_path, sep=';', decimal=',', parse_dates=['HourUTC'])


# === PREPROCESSING ===
# Ensure sorted datetime
df = df.sort_values('HourUTC')

# Calculate capacity factors
df['CF_Offshore'] = df['OffshoreWind_MWh'] / df['OffshoreWindCapacity']
df['CF_Onshore'] = df['OnshoreWind_MWh'] / df['OnshoreWindCapacity']
df['CF_Solar'] = df['SolarPower_MWh'] / df['SolarPowerCapacity']

# Drop any rows with NaNs or Infs
# df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['CF_Offshore', 'CF_Onshore', 'CF_Solar'])

# Add date and hour columns for pivoting
df['Date'] = df['HourUTC'].dt.date
df['Hour'] = df['HourUTC'].dt.hour


# === FUNCTIONS ===

def create_daily_matrix(df, cf_col):
    """
    Pivot to daily capacity factor profiles (24-hour vectors): reshape data (produce a "pivot" table) 
    based on column values. Uses unique values from specified index / columns to form axes of the resulting 
    DataFrame.
    """
    pivot = df.pivot(index='Date', columns='Hour', values=cf_col)

    # Drop days with missing hours
    # pivot = pivot.dropna()

    return pivot

daily_cf_offshore = create_daily_matrix(df, 'CF_Offshore')
daily_cf_onshore = create_daily_matrix(df, 'CF_Onshore')
daily_cf_solar = create_daily_matrix(df, 'CF_Solar')

# === Compute true medoids (k-medoids with k=1) ===
def compute_medoid(daily_cf_matrix):
    model = KMedoids(n_clusters=1, metric='euclidean', method='alternate', init='heuristic')
    model.fit(daily_cf_matrix)                          # Fitting the k-medoids model onto our dataset
    medoid_index = model.medoid_indices_[0]             # Attribute medoid_indices_ contains the indices (row numbers) of the selected medoids (one for each cluster)
    medoid_profile = daily_cf_matrix.iloc[medoid_index] # Retrieve the daily cf profile for that medoid day (corresponding to medoid_indices_)
    medoid_date = daily_cf_matrix.index[medoid_index]   # Get the corresponding date (index label)
    return medoid_profile, medoid_date

medoid_offshore, medoid_offshore_date = compute_medoid(daily_cf_offshore)
medoid_onshore, medoid_onshore_date = compute_medoid(daily_cf_onshore)
medoid_solar, medoid_solar_date = compute_medoid(daily_cf_solar)

# === Plot results ===
plt.figure(figsize=(10, 6))
plt.plot(medoid_offshore.values, label=f'Offshore Wind ({medoid_offshore_date})')
plt.plot(medoid_onshore.values, label=f'Onshore Wind ({medoid_onshore_date})')
plt.plot(medoid_solar.values, label=f'Solar ({medoid_solar_date})')
plt.xlabel("Hour of Day")
plt.ylabel("Capacity Factor")
plt.title("Medoid Daily Capacity Factor Profiles (2021-2024)")
plt.legend()
plt.grid(False)
plt.xticks(ticks=np.arange(0, 24, 1))
plt.tight_layout()
plt.show()
