"""
Functions:
process_hourly_demand_curve, load_demand_curve, load_merged_demand_curve, load_hourly_demand_curves

Main:
Loads the demand curves from one scenario and plots the output.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# == FUNCTIONS ==
def process_hourly_demand_curve(file_path, hour):
    """
    Processes the demand curve data for a given file and hour.
    
    Parameters:
    - file_path: str, path to CSV - assumed to contain data for a single day and bidding zone
    - hour: int, hour from 0 to 23
    
    Returns:
    - pd.DataFrame with ['Price', 'Volume'] for the given hour
    """
    # Load and clean
    bids = pd.read_csv(file_path, skiprows=1)
    bids['Hour'] = pd.to_numeric(bids['Hour'], errors='coerce').ffill().astype(int)     # Convert the values in the 'Hour' column to integers. If not possible, becomes Nan then equal to previous row values (by forward-fills).
    
    # Filter desired hour (note: bids['Hour'] uses 1-24, so +1)
    hourly_bids = bids[bids['Hour'] == hour + 1]
    
    # Filter for demand (Purchase) and sort by price descending (economic merit order)
    demand_bids = hourly_bids[hourly_bids['Sale/Purchase'] == 'Purchase'].sort_values(by='Price', ascending=False)
    
    # Aggregate: keep cumulative demand as price increases
    demand_bids = demand_bids[['Price', 'Volume']].reset_index(drop=True)

    return demand_bids


def load_demand_curve(file_path, hour):
    """
    Processes the CSV spot market file and returns the aggregated demand curve
    for a given hour (0-23).

    Parameters:
    - file_path: str, path to the CSV file - assumed to contain data for a single day and bidding zone
    - hour: int, hour of the day (0-23)

    Returns:
    - pd.DataFrame with columns ['Price', 'Volume'], representing the aggregated demand curve
    """
    # Process file
    demand_bids = process_hourly_demand_curve(file_path, hour)

    # Build final aggregated demand curve
    demand_bids['CumulativeVolume'] = demand_bids['Volume'].cumsum()
    aggregated_curve = demand_bids[['Price', 'CumulativeVolume']].rename(columns={'CumulativeVolume': 'Volume'})
    
    return aggregated_curve


def load_merged_demand_curve(file_dk1, file_dk2, hour):
    """
    Loads and merges demand curves for DK1 and DK2 for a given hour.
    
    Parameters:
    - file_dk1: str, path to DK1 CSV file
    - file_dk2: str, path to DK2 CSV file
    - hour: int, hour of the day (0-23)
    
    Returns:
    - pd.DataFrame with columns ['Price', 'Volume'], representing the merged aggregated demand curve
    """
    # Process both zones
    demand_dk1 = process_hourly_demand_curve(file_dk1, hour)
    demand_dk2 = process_hourly_demand_curve(file_dk2, hour)

    # Combine and group by price (to handle duplicate prices across zones)
    combined = pd.concat([demand_dk1, demand_dk2], ignore_index=True)
    combined_grouped = combined.groupby('Price', as_index=False).sum()
    combined_sorted = combined_grouped.sort_values(by='Price', ascending=False).reset_index(drop=True)

    # Build final aggregated demand curve
    combined_sorted['CumulativeVolume'] = combined_sorted['Volume'].cumsum()
    aggregated_curve = combined_sorted[['Price', 'CumulativeVolume']].rename(columns={'CumulativeVolume': 'Volume'})
    
    return aggregated_curve


def load_hourly_demand_curves(file_path):
    """
    Processes a full day's demand bids into a dictionary of hourly demand curves.
    
    Parameters:
    - file_path: str, path to the CSV file
    - zone_label: optional str, used as a label in the dictionary (for identification)

    Returns:
    - dict: keys = hour (int from 0 to 23), values = pd.DataFrame with ['Price', 'Volume']
    """
    hourly_curves = {}

    for h in range(24):
        demand_bids = process_hourly_demand_curve(file_path, hour=h)
        demand_bids['CumulativeVolume'] = demand_bids['Volume'].cumsum()
        curve = demand_bids[['Price', 'CumulativeVolume']].rename(columns={'CumulativeVolume': 'Volume'})
        hourly_curves[h] = curve

    return hourly_curves


# == MAIN == 
# Get directory path where the script is located
script_dir = os.path.dirname(__file__)

# Set data file paths
dk1_summer_file = 'auction_aggregated_curves_dk1_20240710.csv'
dk1_winter_file = 'auction_aggregated_curves_dk1_20241127.csv'
dk1_lowload_file = 'auction_aggregated_curves_dk1_20240519.csv'
dk2_summer_file = 'auction_aggregated_curves_dk2_20240710.csv'
dk2_winter_file = 'auction_aggregated_curves_dk2_20241127.csv'
dk2_lowload_file = 'auction_aggregated_curves_dk2_20240519.csv'

# Choose scenario
# bidding_zone  \in {'dk1', 'dk2'}
# scenario      \in {'winter', 'summer', 'lowload'}
csv_filename = dk2_winter_file
label = "DK2 Winter"

# Build relative path to CSV file in the same folder
csv_path = os.path.join(script_dir, csv_filename)

# If merging bidding zones DK1 and DK2
# csv_filename2 = dk1_winter_file
# label = "DK Winter"
# csv_path2 = os.path.join(script_dir, csv_filename2)

# Get aggregated demand curve for all hours
plt.figure(figsize=(10, 6))
for h in range(24):
    # Load for one bidding zone, one scenario, one hour
    demand_curve = load_demand_curve(csv_path, hour=h)

    # Load for two merged bidding zones, one scenario, one hour
    # demand_curve = load_merged_demand_curve(csv_path, csv_path2, hour=h)

    # Plot
    plt.plot(demand_curve['Volume'], demand_curve['Price'], label=f'{label} - {h:02}:00')
    
plt.xlabel('Cumulative Volume (MW)')
plt.ylabel('Price (€/MWh)')
plt.title('Demand Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
