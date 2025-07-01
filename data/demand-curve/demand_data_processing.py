"""
Functions:
process_hourly_demand_curve, process_merged_demand_curves, load_hourly_demand_curves

Main:
Loads the hourly demand curves from one scenario and plots the output.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# == FUNCTIONS ==
def process_hourly_demand_curve(file_path:str, hour:int):
    """
    Processes the CSV spot market file and returns the aggregated demand curve
    for a given hour.
    
    Parameters:
        file_path (str): path to CSV file - assumed to contain data for a single day and bidding zone
        hour (int): hour of the day (0-23)
    
    Returns:
        pd.DataFrame with columns ['Price', 'Volume'] for the given hour, representing the aggregated demand curve
    """
    # Load and clean
    bids = pd.read_csv(file_path, skiprows=1)
    bids['Hour'] = pd.to_numeric(bids['Hour'], errors='coerce').ffill().astype(int)     # Convert the values in the 'Hour' column to integers. If not possible, becomes Nan then equal to previous row values (by forward-fills).
    
    # Filter desired hour (note: bids['Hour'] uses 1-24, so +1)
    hourly_bids = bids[bids['Hour'] == hour + 1]
    
    # Filter for demand (Purchase) and sort by price descending (economic merit order)
    demand_bids = hourly_bids[hourly_bids['Sale/Purchase'] == 'Purchase'].sort_values(by='Price', ascending=False)

    # Final aggregated demand curve
    aggregated_curve = demand_bids[['Price', 'Volume']].reset_index(drop=True)

    return aggregated_curve


def cumulative_to_marginal(df):
    """
    Converts a cumulative volume curve to a marginal (stepwise) volume curve.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Price', 'Volume'], where 'Volume' is cumulative.

    Returns:
        df (pd.DataFrame): DataFrame with columns ['Price', 'Marginal'], containing marginal volumes per price step.
    """
    # Calculate marginal volumes from cumulative
    df = df.copy()
    df['Marginal'] = df['Volume'].diff(1)
    df = df[df['Marginal'].notna() & (df['Marginal'] != 0)]

    return df[['Price', 'Marginal']]


def process_merged_demand_curves(file_dk1:str, file_dk2:str, hour:int):
    """
    Processes DK1 and DK2 CSV spot market files for a given hour. Loads and merges demand curves. 
    Returns the aggregated demand curve.
    
    Parameters:
        file_dk1 (str): path to DK1 CSV file
        file_dk2 (str): path to DK2 CSV file
        hour (int): hour of the day (0-23)
    
    Returns:
        pd.DataFrame with columns ['Price', 'Volume'] for the given hour, representing the merged aggregated demand curve
    """

    # Load and convert the demand curve for each bidding zone
    demand_dk1 = process_hourly_demand_curve(file_dk1, hour)
    demand_dk2 = process_hourly_demand_curve(file_dk2, hour)
    demand_dk1_marg = cumulative_to_marginal(demand_dk1)
    demand_dk2_marg = cumulative_to_marginal(demand_dk2)

    # Combine **marginal** volumes and group by marginal price (to handle duplicate prices across zones)
    combined = pd.concat([demand_dk1_marg, demand_dk2_marg], ignore_index=True)
    combined_grouped = combined.groupby('Price', as_index=False).sum()
    combined_sorted = combined_grouped.sort_values(by='Price', ascending=False).reset_index(drop=True)

    # Build final aggregated demand curve by recomputing cumulative volume
    combined_sorted['CumulativeVolume'] = combined_sorted['Marginal'].cumsum()
    aggregated_curve = combined_sorted[['Price', 'CumulativeVolume']].rename(columns={'CumulativeVolume': 'Volume'})

    # Add the initial row for the stepwise demand curve to begin at volume 0
    new_row = pd.DataFrame({'Price': [4000], 'Volume': [0]})
    aggregated_curve = pd.concat([new_row, aggregated_curve], ignore_index=True)
    
    return aggregated_curve


def load_hourly_demand_curves(file_path:str):
    """
    Processes a full day's demand bids into a dictionary of hourly demand curves.
    
    Parameters:
        file_path (str): path to the CSV file

    Returns:
        dict: keys = hour (int from 0 to 23), values = pd.DataFrame with ['Price', 'Volume']
    """
    hourly_curves = {}

    for h in range(24):
        curve = process_hourly_demand_curve(file_path, hour=h)
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
csv_filename2 = dk1_winter_file
label = "DK Winter"
csv_path2 = os.path.join(script_dir, csv_filename2)

# Get aggregated demand curve for all hours
plt.figure(figsize=(10, 6))
plt.subplot(1,2,1)

inflexible_demand = []
total_demand = []
for h in range(24):
    # Load for one bidding zone, one scenario, one hour
    # demand_curve = process_hourly_demand_curve(csv_path, hour=h)

    # Load for two merged bidding zones, one scenario, one hour
    demand_curve = process_merged_demand_curves(csv_path, csv_path2, hour=h)

    inflexible_demand.append(demand_curve['Volume'][1])     # First row (Price == 4000)
    total_demand.append(demand_curve.iloc[-1]['Volume'])    # Last row (Accumulated volume)

    # Plot
    plt.step(demand_curve['Volume'], demand_curve['Price'], label=f'{h:02}:00')
    
plt.xlabel('Cumulative Volume (MW)')
plt.ylabel('Price (€/MWh)')
plt.title(f'Demand Curve of {label} scenario')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Plot the demand over time in the day
plt.subplot(1,2,2)
plt.plot(range(24), inflexible_demand, label="Inflexible Demand")
plt.plot(range(24), total_demand, label="Total Demand")
plt.xlabel('Time (h)')
plt.ylabel('Volume (MW)')
plt.title(f'Demand Over Time for {label} scenario')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()