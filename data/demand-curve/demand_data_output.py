"""
Main script to load, process, and reduce hourly demand curves for multiple scenarios and bidding zones.

Processes:
- Loads auction aggregated demand curves from CSV files for DK1 and DK2 zones in Summer, Winter, and Low-load scenarios.
- Converts raw hourly demand bids into dictionaries of demand curves per hour.
- Reduces each hourly demand curve to N representative price steps using clustering.
- Saves the reduced demand price and volume data as CSV files for each scenario and zone.

## Parameters:
    N (int):
    Number of representative price steps for demand curve reduction.

## Files:
    Input CSV files must be placed in the script directory with predefined names.

## Outputs:
    CSV files 'demand_price-\<Zone>\<Scenario>-\<N>steps.csv' and 'demand_volume-\<Zone>\<Scenario>-\<N>steps.csv'
    for each bidding zone and scenario, containing approximated demand price and volume data.
"""

import os
from demand_data_processing import load_hourly_demand_curves
from demand_data_processing_step2 import reduce_demand_curves

# == MAIN == 
# **Parameter to change if needed**
N = 20

# Get directory path where the script is located
script_dir = os.path.dirname(__file__)

# Set data file paths
dk1_summer_file = 'auction_aggregated_curves_dk1_20240710.csv'
dk1_winter_file = 'auction_aggregated_curves_dk1_20241127.csv'
dk1_lowload_file = 'auction_aggregated_curves_dk1_20240519.csv'
dk2_summer_file = 'auction_aggregated_curves_dk2_20240710.csv'
dk2_winter_file = 'auction_aggregated_curves_dk2_20241127.csv'
dk2_lowload_file = 'auction_aggregated_curves_dk2_20240519.csv'

## === Load all demand curves and store in a dictionnary ===
# demand_data = {}

# Characterize scenarios by bidding zone (DK1, DK2) and time period (summer, winter, low-load)
data_scenario_files = {
    "Summer": (dk1_summer_file, dk2_summer_file),
    "Winter": (dk1_winter_file, dk2_winter_file),
    "Low load": (dk1_lowload_file, dk2_lowload_file)
    }
bidding_zones = ["DK1", "DK2"]

for scenario, scenario_files in data_scenario_files.items():
    for zone in bidding_zones:
        # Loop on each scenario
        label = zone + scenario                                 # scenario label
        filename = scenario_files[bidding_zones.index(zone)]    # csv filename corresponding to scenario
        
        # Build relative path to CSV file in the same folder
        csv_path = os.path.join(script_dir, filename)

        # Processes a full day's demand bids into a dictionary of hourly demand curves
        price_demand_curves = load_hourly_demand_curves(csv_path)

        # Processes each demand curve into a reduced stepwise demand curve with N steps
        # Establishes the model inputs
        demand_price, demand_volume = reduce_demand_curves(price_demand_curves, N)

        # Increment the final dictionnary
        # demand_data[label] = (demand_price, demand_volume)

        # Save demand data as csv files
        demand_price.to_csv(f'demand_price-{label}-{N}steps.csv')
        demand_volume.to_csv(f'demand_volume-{label}-{N}steps.csv')










