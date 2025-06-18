import os

from demand_data_processing import process_hourly_demand_curve, load_demand_curve, load_merged_demand_curve, load_hourly_demand_curves

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

# Load all demand curves and store in a dictionnary
# ...
# for file in [...]:
# 
