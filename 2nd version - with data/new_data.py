import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# === INITIALIZATION ===
# Parameters
# bidding_zone = input('Choose the bidding zone: DK1 or DK2 \n')
# year_period = input('Choose the representative period in the year: Winter, Summer or LowLoad \n')
bidding_zone = 'DK2'
year_period = 'Winter'
scenario_label = bidding_zone + year_period

demand_step_numbers = 20        # number of steps in the stepwise price demand curve

# Path to this script
base_dir = Path(__file__).resolve().parent

# === PRICE DEMAND CURVE DATA ===
# Path for price demand curve data
demand_files_path = base_dir.parent / 'data' / 'demand_curve' / 'extracted' / f'N={demand_step_numbers}'

# Import the csv data files relevant for our scenario
csv_files = list(demand_files_path.glob('*.csv'))
demand_files = {file.name: pd.read_csv(file, header=0, index_col=0) for file in csv_files}
print(demand_files.keys(), '\n')

# Load the files
demand_price = {}
demand_volume = {}
for filename, dataframe in demand_files.items():
    parts = filename.split('-')
    data_type = parts[0]
    scenario = parts[1]

    if data_type == 'demand_price':
        demand_price[scenario] = dataframe

    elif data_type == 'demand_volume':
        demand_volume[scenario] = dataframe

    else:
        raise("Error in the data files nomenclature.")
    
print(demand_price.keys(), '\nIt is not missing any data: ', demand_price.keys()==demand_volume.keys(), '\n')
    
# === DATA PLOTS ===
scenario_colors = {"Winter": "blue", "Summer": "orange", "LowLoad": "purple"}

# Plot the price demand curve for some hours
plt.subplots(2, 2, figsize=(12, 6))
hours_to_plot = [0,6,12,18]

for hour in hours_to_plot:
    plt.subplot(2, 2, hours_to_plot.index(hour)+1)
    for scenario_to_plot in ["Winter", "Summer", "LowLoad"]:
        scenario_label_to_plot = bidding_zone + scenario_to_plot
        x = demand_volume[scenario_label_to_plot][f'{hour}'].to_numpy()
        y = demand_price[scenario_label_to_plot][f'{hour}'].to_numpy()

        # Plot
        plt.step(np.insert(x, 0, 0), np.insert(y, 0, 4000), label=f'{scenario_to_plot}', color=scenario_colors[scenario_to_plot])

    plt.xlabel('Cumulative Volume (MW)')
    plt.ylabel('Price (€/MWh)')
    plt.axhline(y=0, color="black", linewidth=0.8)
    plt.title(f'Price Demand Curve of Scenarios at {hour:02}:00')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# Plot the demand over time in the day
plt.figure(figsize=(12, 6))

for scenario_to_plot in ["Winter", "Summer", "LowLoad"]:
    scenario_label_to_plot = bidding_zone + scenario_to_plot
    inflexible_demand = demand_volume[scenario_label_to_plot].loc[0].to_numpy()                 # First row (Price == 4000)
    total_demand = demand_volume[scenario_label_to_plot].loc[demand_step_numbers-1].to_numpy()  # Last row (Accumulated volume)

    # Plot
    plt.plot(range(24), inflexible_demand, label=f"{scenario_to_plot} Inflexible Demand", linestyle='--', color=scenario_colors[scenario_to_plot])
    plt.plot(range(24), total_demand, label=f"{scenario_to_plot} Total Demand", color=scenario_colors[scenario_to_plot])

plt.xlabel('Time (h)')
plt.ylabel('Volume (MW)')
plt.ylim(bottom=0)
plt.title(f'Demand Over Time for Scenarios')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

# === INITIALIZATION OF OUR MODEL ===
# Extract the corresponding data for our desired scenario
Demand_Price = demand_price[scenario_label]
Demand_Volume = demand_volume[scenario_label]

# === RES PRODUCTION DATA ===
#  