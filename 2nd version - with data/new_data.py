"""
Functions to load renewable energy production and price-demand curve data for one selected scenario.

Includes:
- load_res_production_data: computes hourly RES production profiles.
- load_price_demand_curve_data: loads demand price and volume curves.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# === INITIALIZATION FUNCTIONS FOR OUR MODEL ===
def load_price_demand_curve_data(bidding_zone:str, time_period:str, demand_step_numbers=20, plots=False, plot_hours = [0,6,12,18]):
    """
    Load demand price and volume data for a specific bidding zone and time period.

    Parameters:
        bidding_zone (str):
            Identifier for the bidding zone, e.g. 'DK1' or 'DK2'.
        time_period (str):
            Identifier for the time period: 'Winter', 'Summer', or 'LowLoad'.
        demand_step_numbers (int, optional):
            Total number of steps desired in the discretized price demand curve.
            Default is 20.
        plots (bool, optional):
            If True, generates plots for the price-demand curves and demand over time.
            Default is False.
        plot_hours (list of int, optional):
            Hours of the day (0-23) to plot price-demand curves for.
            Default is [0, 6, 12, 18].

    Returns:
        (Demand_Price, Demand_Volume) (pd.DataFrame, pd.DataFrame):
            - Demand_Price: DataFrame of shape (N, 24) containing demand price steps for the scenario.
              Columns are hours (0-23), rows are price steps (sorted high to low).
            - Demand_Volume: DataFrame of shape (N, 24) with corresponding cumulative volumes per price step,
              same structure as Demand_Price.
    """
    # === PARAMETERS ===
    # bidding_zone = input('Choose the bidding zone: DK1 or DK2 \n')
    # year_period = input('Choose the representative period in the year: Winter, Summer or LowLoad \n')
    # scenario_label = bidding_zone + year_period

    N = demand_step_numbers

    # Path to this script
    base_dir = Path(__file__).resolve().parent

    # === PRICE DEMAND CURVE DATA ===
    # Path for price demand curve data
    demand_files_path = base_dir.parent / 'data' / 'demand_curve' / 'extracted' / f'N={N}'

    # Import the csv data files relevant for our scenario
    csv_files = list(demand_files_path.glob('*.csv'))
    demand_files = {file.name: pd.read_csv(file, header=0, index_col=0) for file in csv_files}
    # print(demand_files.keys(), '\n')

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
        
    # print(demand_price.keys(), '\nIt is not missing any data: ', demand_price.keys()==demand_volume.keys(), '\n')
    
    # === OUTPUT ===
    # Extract the demand data for the modelled scenario
    scenario_label = bidding_zone + time_period

    Demand_Price = demand_price[scenario_label]
    Demand_Volume = demand_volume[scenario_label]


    if plots:
        # === DATA PLOTS ===
        scenario_colors = {"Winter": "blue", "Summer": "orange", "LowLoad": "purple"}
        bidding_zone_to_plot = bidding_zone

        # Plot the price demand curve for some hours
        plt.subplots(2, 2, figsize=(12, 6))
        hours_to_plot = plot_hours

        for hour in hours_to_plot:
            plt.subplot(2, 2, hours_to_plot.index(hour)+1)
            for scenario_to_plot in ["Winter", "Summer", "LowLoad"]:
                scenario_label_to_plot = bidding_zone_to_plot + scenario_to_plot
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
        plt.savefig("price_demand_curves_fulldata.png")

        # Plot the demand over time in the day
        plt.figure(figsize=(12, 6))

        for scenario_to_plot in ["Winter", "Summer", "LowLoad"]:
            scenario_label_to_plot = bidding_zone_to_plot + scenario_to_plot
            inflexible_demand = demand_volume[scenario_label_to_plot].loc[0].to_numpy()     # First row (Price == 4000)
            total_demand = demand_volume[scenario_label_to_plot].loc[N-1].to_numpy()        # Last row (Accumulated volume)

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
        # plt.show()
        plt.savefig("demand_over_time_fulldata.png")


    return Demand_Price, Demand_Volume


def load_res_production_data(season:str, plots=False):
    """
    Load and compute renewable energy source (RES) hourly production profiles for a given season.

    The function reads cluster medoid profiles from CSV files, computes capacity factors for
    offshore wind, onshore wind, solar, and bioenergy, and calculates the total hourly RES production
    based on 2030 capacity installation plans. Optionally, it can generate plots showing the
    hourly production mix.

    Parameters:
        season (str):
            Season label for scenario selection. Supported values: "Winter" or "Summer".
        plots (bool, optional):
            If True, displays plots of RES production profiles and mix for each season.
            Default is False.

    Returns:
        RES (np.ndarray):
            Array of length 24 with the total hourly RES production (in MW) for the selected season.
    """
    # Path to this script
    base_dir = Path(__file__).resolve().parent

    # === RES PRODUCTION DATA ===
    # Path for RES production cluster data - 1, 2 or 3 clusters: assumed to be fixed
    files_path = base_dir.parent / 'data' / 'RES' / 'csv--data_processing-v4'

    # Import the csv data files relevant for our model
    csv_files = list(files_path.glob('medoids_profile*.csv'))
    cluster_files = {file.name: pd.read_csv(file, header=0) for file in csv_files}

    # Cluster data
    medoids_profile = cluster_files["medoids_profile_summary--2cluster.csv"][[f"{h}" for h in range(24)]]
    medoids_profile_average = cluster_files["medoids_profile_summary--1cluster.csv"][[f"{h}" for h in range(24)]]

    # Compute RES hourly capacity factors - higher probability cluster is chosen (or in some cases of similar probability, the more relevant) - see clusters diagnostic
    offshore_profile_winter = medoids_profile.iloc[1].to_numpy() * 0.9
    # offshore_profile_summer = medoids_profile.iloc[2].to_numpy()
    # offshore_profile_winter = medoids_profile_average.iloc[0].to_numpy()
    offshore_profile_summer = medoids_profile_average.iloc[1].to_numpy()

    # onshore_profile_winter = medoids_profile.iloc[5].to_numpy()
    # onshore_profile_summer = medoids_profile.iloc[7].to_numpy()
    onshore_profile_winter = medoids_profile_average.iloc[2].to_numpy()
    onshore_profile_summer = medoids_profile_average.iloc[3].to_numpy()

    # solar_profile_winter = medoids_profile.iloc[9].to_numpy()
    solar_profile_summer = medoids_profile.iloc[10].to_numpy() * 0.9
    solar_profile_winter = medoids_profile_average.iloc[4].to_numpy()
    # solar_profile_summer = medoids_profile_average.iloc[5].to_numpy()

    # Energy mix: capacity installation plans for 2030 (IEA)
    offshore_capacity = 7200
    onshore_capacity = 5500
    solar_capacity = 5265
    bioenergy_capacity = 557


    # Compute RES hourly production = RES cf * installed capacity
    RES_winter = offshore_profile_winter*offshore_capacity + onshore_profile_winter*onshore_capacity + solar_profile_winter*solar_capacity + bioenergy_capacity*1
    RES_summer = offshore_profile_summer*offshore_capacity + onshore_profile_summer*onshore_capacity + solar_profile_summer*solar_capacity + bioenergy_capacity*1
    

    # === OUTPUT ===
    # Choose data corresponding to the chosen scenario
    if season == "Winter":
        RES = RES_winter
    else:
        RES = RES_summer


    if plots:
        # === DATA PLOTS ===
        plt.figure(figsize=(10, 6))
        temps = range(24)

        # plt.plot(temps, RES_winter, label="RES in winter")
        # plt.plot(temps, RES_summer, label="RES in summer")
        # plt.xlabel("Hour (h)")
        # plt.ylabel("Power (MW)")
        # plt.title("Renewable Hourly Production Scenarios (Winter vs Summer)")
        # plt.legend(loc="upper right")
        # plt.grid()

        plt.subplot(1,2,1)
        plt.bar(x=temps, height=bioenergy_capacity, color='gray', align='edge', label="Bioenergy")
        plt.bar(x=temps, height=offshore_profile_winter*offshore_capacity, bottom=bioenergy_capacity, color='darkblue', align='edge', label="Offshore wind")
        plt.bar(x=temps, height=onshore_profile_winter*onshore_capacity, bottom=bioenergy_capacity+offshore_profile_winter*offshore_capacity, color='lightskyblue', align='edge', label="Onshore wind")
        plt.bar(x=temps, height=solar_profile_winter*solar_capacity, bottom=bioenergy_capacity+offshore_profile_winter*offshore_capacity+onshore_profile_winter*onshore_capacity, color='orange', align='edge', label="Solar")
        plt.plot(temps, RES_winter, label="RES in winter", linestyle='--', color='black')
        plt.plot(temps, RES_summer, label="RES in summer", linestyle='--', color='gray')
        plt.xlabel("Hour (h)")
        plt.ylabel("Power (MW)")
        plt.title("Renewable Hourly Production Mix in Winter")
        plt.legend(loc="upper right")

        plt.subplot(1,2,2)
        plt.bar(x=temps, height=bioenergy_capacity, color='gray', align='edge', label="Bioenergy")
        plt.bar(x=temps, height=offshore_profile_summer*offshore_capacity, bottom=bioenergy_capacity, color='darkblue', align='edge', label="Offshore wind")
        plt.bar(x=temps, height=onshore_profile_summer*onshore_capacity, bottom=bioenergy_capacity+offshore_profile_summer*offshore_capacity, color='lightskyblue', align='edge', label="Onshore wind")
        plt.bar(x=temps, height=solar_profile_summer*solar_capacity, bottom=bioenergy_capacity+offshore_profile_summer*offshore_capacity+onshore_profile_summer*onshore_capacity, color='orange', align='edge', label="Solar")
        plt.plot(temps, RES_winter, label="RES in winter", linestyle='--', color='black')
        plt.plot(temps, RES_summer, label="RES in summer", linestyle='--', color='gray')
        plt.xlabel("Hour (h)")
        plt.ylabel("Power (MW)")
        plt.title("Renewable Hourly Production Mix in Summer")
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig("RES_production_mix-fulldata.png")
        # plt.show()


    return RES


