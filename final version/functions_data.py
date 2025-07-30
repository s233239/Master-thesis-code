"""
Functions to load renewable energy production and price-demand curve data for one selected scenario.

Includes:
- load_res_production_data: computes hourly RES production profiles.
- load_price_demand_curve_data: loads demand price and volume curves.
- load_storage_data: computes storage sizing for multiple players.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# === INITIALIZATION FUNCTIONS FOR OUR MODEL ===
def load_price_demand_curve_data(bidding_zone:str, time_period:str, demand_step_numbers=20, plots=False, hour_to_plot = 14):
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
        (Demand_Price, Demand_Volume) (np.array, np.array):
            - Demand_Price: Array of shape (N, 24) containing demand price steps for the scenario.
              Columns are hours (0-23), rows are price steps (sorted high to low).
            - Demand_Volume: Array of shape (N, 24) with corresponding cumulative volumes per price step,
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

    Demand_Price = demand_price[scenario_label].to_numpy()
    Demand_Volume = demand_volume[scenario_label].to_numpy()

    # Scaled up to 2030 projections
    scaling_factor = 1.85
    Demand_Volume = scaling_factor * Demand_Volume


    if plots:
        # === DATA PLOTS ===
        scenario_names = ["Winter", "Summer", "LowLoad"]
        scenario_colors = {"Winter": "blue", "Summer": "orange", "LowLoad": "purple"}
        bidding_zone_to_plot = bidding_zone

        # Plot the price demand curve for some hours
        fig, axs = plt.subplots(1,2,figsize=(7, 3))

        # Plot the complete price demand curve
        ax = axs[0]
        for scenario_to_plot in scenario_names:
            scenario_label_to_plot = bidding_zone_to_plot + scenario_to_plot
            x = demand_volume[scenario_label_to_plot][f'{hour_to_plot}'].to_numpy() * scaling_factor
            y = demand_price[scenario_label_to_plot][f'{hour_to_plot}'].to_numpy()

            # Plot
            ax.step(np.insert(x,0,0), np.insert(y,0,4000), where='post', label=f'{scenario_to_plot}', color=scenario_colors[scenario_to_plot])

        ax.set_xlabel('Cumulative Volume (MW)')
        ax.set_ylabel('Price (€/MWh)')
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_title(f'DPQC at {hour_to_plot:02}:00')
        ax.grid(True)
        ax.legend(loc="upper right")

        # Zoomed in without considering inflexible demand
        ax = axs[1]
        for scenario_to_plot in scenario_names:
            scenario_label_to_plot = bidding_zone_to_plot + scenario_to_plot
            x = demand_volume[scenario_label_to_plot][f'{hour_to_plot}'].to_numpy() * scaling_factor
            y = demand_price[scenario_label_to_plot][f'{hour_to_plot}'].to_numpy()

            # Plot
            ax.step(np.insert(x,0,0), np.insert(y,0,4000), where='post', label=f'{scenario_to_plot}', color=scenario_colors[scenario_to_plot])

        ax.set_xlabel('Cumulative Volume (MW)')
        # ax.set_ylabel('Price (€/MWh)')
        bottom, top = ax.set_ylim()
        ax.set_ylim(bottom=0.1*bottom, top=350)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_title(f'Zoomed-in at hour {hour_to_plot:02}:00')
        ax.grid(True)
        # ax.legend()
        
        plt.tight_layout()
        plt.savefig("price_demand_curves_fulldata.pdf")

        # Plot the demand over time in the day
        fig, axs = plt.subplots(1,3,figsize=(9, 4), sharey=True)

        for scenario_to_plot in scenario_names:
            scenario_label_to_plot = bidding_zone_to_plot + scenario_to_plot
            inflexible_demand = demand_volume[scenario_label_to_plot].loc[0].to_numpy()  * scaling_factor     # First row (Price == 4000)
            total_demand = demand_volume[scenario_label_to_plot].loc[N-1].to_numpy()  * scaling_factor        # Last row (Accumulated volume)
            ax = axs[scenario_names.index(scenario_to_plot)]

            # Plot
            ax.plot(range(24), inflexible_demand/1000, label=f"Inflexible Demand", linestyle='--', color=scenario_colors[scenario_to_plot])
            ax.plot(range(24), total_demand/1000, label=f"Total Demand", color=scenario_colors[scenario_to_plot])
            ax.set_xlabel('Time (h)')
            if scenario_names.index(scenario_to_plot) == 0:
                ax.set_ylabel('Volume (GW)')
            ax.set_ylim(bottom=0)
            ax.set_title(f'{scenario_to_plot}')
            ax.grid(True)
            ax.legend(loc='lower left')

        plt.tight_layout()
        plt.savefig("demand_over_time_fulldata.pdf")


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
    files_path = base_dir.parent / 'data' / 'RES' / 'extracted_csv-clusters'

    # Import the csv data files relevant for our model
    csv_files = list(files_path.glob('medoids_profile*.csv'))
    cluster_files = {file.name: pd.read_csv(file, header=0) for file in csv_files}

    # Cluster data
    medoids_profile = cluster_files["medoids_profile_summary--2cluster.csv"][[f"{h}" for h in range(24)]]
    medoids_profile_average = cluster_files["medoids_profile_summary--1cluster.csv"][[f"{h}" for h in range(24)]]

    # Compute RES hourly capacity factors - higher probability cluster is chosen (or in some cases of similar probability, the more relevant) - see clusters diagnostic
    # offshore_profile_winter = medoids_profile.iloc[1].to_numpy()
    # offshore_profile_summer = medoids_profile.iloc[2].to_numpy()
    offshore_profile_winter = medoids_profile_average.iloc[0].to_numpy()
    offshore_profile_summer = medoids_profile_average.iloc[1].to_numpy()

    # onshore_profile_winter = medoids_profile.iloc[5].to_numpy()
    # onshore_profile_summer = medoids_profile.iloc[7].to_numpy()
    onshore_profile_winter = medoids_profile_average.iloc[2].to_numpy()
    onshore_profile_summer = medoids_profile_average.iloc[3].to_numpy()

    # solar_profile_winter = medoids_profile.iloc[9].to_numpy()
    # solar_profile_summer = medoids_profile.iloc[10].to_numpy()
    solar_profile_winter = medoids_profile_average.iloc[4].to_numpy()
    solar_profile_summer = medoids_profile_average.iloc[5].to_numpy()

    # Energy mix: capacity installation plans for 2030 (IEA)
    offshore_capacity = 18e3
    onshore_capacity = 9e3
    solar_capacity = 12e3
    bioenergy_capacity = 0


    # Compute RES hourly production = RES cf * installed capacity
    RES_winter = offshore_profile_winter*offshore_capacity + onshore_profile_winter*onshore_capacity + solar_profile_winter*solar_capacity + bioenergy_capacity*1
    RES_summer = offshore_profile_summer*offshore_capacity + onshore_profile_summer*onshore_capacity + solar_profile_summer*solar_capacity + bioenergy_capacity*1
    

    # === OUTPUT ===
    # Choose data corresponding to the chosen scenario
    if season == "Winter":
        RES = RES_winter
    else:   # Summer or LowLoad
        RES = RES_summer


    if plots:
        # === DATA PLOTS ===
        plt.subplots(1,2,figsize=(8, 4), sharey=True)
        temps = range(24)

        # plt.plot(temps, RES_winter, label="RES in winter")
        # plt.plot(temps, RES_summer, label="RES in summer")
        # plt.xlabel("Hour (h)")
        # plt.ylabel("Power (MW)")
        # plt.title("Renewable Hourly Production Scenarios (Winter vs Summer)")
        # plt.legend(loc="upper right")
        # plt.grid()

        plt.subplot(1,2,1)
        # plt.bar(x=temps, height=bioenergy_capacity, color='gray', align='edge', label="Bioenergy")
        plt.bar(x=temps, height=offshore_profile_winter*offshore_capacity/1000, bottom=bioenergy_capacity/1000, color='darkblue', align='edge', label="Offshore wind")
        plt.bar(x=temps, height=onshore_profile_winter*onshore_capacity/1000, bottom=(bioenergy_capacity+offshore_profile_winter*offshore_capacity)/1000, color='lightskyblue', align='edge', label="Onshore wind")
        plt.bar(x=temps, height=solar_profile_winter*solar_capacity/1000, bottom=(bioenergy_capacity+offshore_profile_winter*offshore_capacity+onshore_profile_winter*onshore_capacity)/1000, color='orange', align='edge', label="Solar")
        # plt.plot(temps, RES_winter, label="RES in winter", linestyle='--', color='black')
        # plt.plot(temps, RES_summer, label="RES in summer", linestyle='--', color='gray')
        plt.xlabel("Hour (h)")
        plt.ylabel("Power (GW)")
        plt.title("Mix in Winter")
        # plt.legend(loc="lower left")

        plt.subplot(1,2,2)
        # plt.bar(x=temps, height=bioenergy_capacity, color='gray', align='edge', label="Bioenergy")
        plt.bar(x=temps, height=offshore_profile_summer*offshore_capacity/1000, bottom=bioenergy_capacity/1000, color='darkblue', align='edge', label="Offshore wind")
        plt.bar(x=temps, height=onshore_profile_summer*onshore_capacity/1000, bottom=(bioenergy_capacity+offshore_profile_summer*offshore_capacity)/1000, color='lightskyblue', align='edge', label="Onshore wind")
        plt.bar(x=temps, height=solar_profile_summer*solar_capacity/1000, bottom=(bioenergy_capacity+offshore_profile_summer*offshore_capacity+onshore_profile_summer*onshore_capacity)/1000, color='orange', align='edge', label="Solar")
        # plt.plot(temps, RES_winter, label="RES in winter", linestyle='--', color='black')
        # plt.plot(temps, RES_summer, label="RES in summer", linestyle='--', color='gray')
        plt.xlabel("Hour (h)")
        plt.title("Mix in Summer")
        plt.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig("RES_production_mix-fulldata.pdf")
        # plt.show()


    return RES

def load_storage_data(Residual, n_players, min_eta, storage_Crate_default, OC_default, N,
                      storage_capacity=None,
                      plots=False, bidding_zone=None, season=None):
    """
    Compute storage sizing for multiple players based on residual demand.

    Parameters:
        Residual (array): Hourly residual demand (MW)
        n_players (int): Number of storage players
        min_eta (float): Round-trip efficiency (e.g. 0.85)
        storage_Crate_default (float): C-rate to derive power rating (1/h)
        OC_default (float): Default marginal cost for all players
        N (int): Number of bidding steps per player
        plots (bool): If True, generate diagnostic plots
        bidding_zone (str): For plot filename (required if plots=True)
        season (str): For plot filename (required if plots=True)

    Returns:
        storage_summary_data (dict): Summary dictionary with storage parameters per player
    """
    # Battery parameters for all players
    Q_max_all = np.zeros(n_players)     # Maximum available power (MW)
    Q_all = np.zeros((n_players, N))      # List of possible power bids
    OC_all = np.zeros(n_players)        # Marginal cost (€/MW)
    E_max_all = np.zeros(n_players)     # Maximum battery level (MWh)
    Eta_all = np.zeros(n_players)       # Storage round-trip efficiency
    
    # Offset with 0 to effectively compute battery requirements 
    # (if first residual demand value x is positive, Local_cumul=0 though we would need x MWh to satisfy the demand - it is due to the min being updated over time but not initialized)
    Residual = np.insert(Residual, 0, 0, 0)

    # Storage requirement computation
    Residual_corrected = np.where(Residual > 0, Residual/min_eta, Residual)   # Taking into account the round-trip efficiency when the battery discharge on the grid (=> need to discharge 100% + eta% energy to satisfy the corresponding demand)

    Cummul_res_corr = np.cumsum(Residual_corrected)

    Local_cumul = Cummul_res_corr - np.minimum.accumulate(Cummul_res_corr)
                                    # minimum of sliced list[:t]

    # Minimum total amount of energy the battery must store to satisfy the demand at any time step > informs E_max
    Capacity_req = int(np.ceil(np.max(Local_cumul)))  # MWh total required
    if storage_capacity:
        Capacity_req = storage_capacity*Capacity_req

    # Minimum instantaneous power the storage system need to have to satisfy the demand at any time step > informs Q_max
    # PowerRating_req = np.max(np.abs(Residual_corrected))    
    # PowerRating_req = int(np.round(PowerRating_req/10)*10)
    PowerRating_available = int(np.ceil(Capacity_req * storage_Crate_default))  # MW total available

    # Offset back
    Residual = Residual[1:]
    Residual_corrected = Residual_corrected[1:]

    # Define player size shares depending on n_players
    if n_players == 1:
        size_stor = [1]
    elif n_players == 2:
        size_stor = [1/3, 2/3]
    elif n_players == 4:
        size_stor = [0.1, 0.2, 0.3, 0.4]
    elif n_players == 6:
        size_stor = [0.05, 0.1, 0.1, 0.15, 0.25, 0.35]
    elif n_players == 8:
        size_stor = [0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25]
    elif n_players == 0:
        print(f"Distribution of storage capacity is not defined.")
    else:
        raise ValueError(f"Distribution of storage capacity is not defined for n_players={n_players}")

    for player in range(n_players):
        OC_all[player] = OC_default
        Eta_all[player] = min_eta
        E_max_all[player] = int(np.floor(Capacity_req * size_stor[player] / 10) * 10)
        Q_max_all[player] = int(np.floor(PowerRating_available * size_stor[player]))
        for i in range(N):
            Q_all[player,i] = round(Q_max_all[player] * (i / N), 2)
  
    # DataFrame of storage characteristics
    storage_summary_data = pd.DataFrame({
        "Player": [chr(65 + p) for p in range(n_players)],
        "OC": OC_all,
        "Eta": Eta_all,
        "E_max": E_max_all,
        "Q_max": Q_max_all,
        "Q_steps": [Q_all[p, :] for p in range(n_players)]
        })
    print(storage_summary_data)

    if plots:
        if bidding_zone is None or season is None:
            raise ValueError("bidding_zone and season must be provided if plots=True")

        # fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        x = range(len(Residual))

        # Plot 1: Residual and Residual Corrected
        fig0, ax0 = plt.subplots(figsize=(8, 4))
        ax0.bar(x, Residual_corrected, color='tab:orange',
                   label='Residual Corrected (inefficiency)', align='edge')
        ax0.bar(x, Residual, color='tab:blue',
                   label='Residual (Demand - RES)', align='edge')
        ax0.axhline(PowerRating_available, color='tab:red', linestyle='--',
                       label='[max] Available Storage Power (MW)')
        ax0.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax0.set_title('Residual Demand vs Corrected Residual')
        ax0.set_ylabel('Power [MW]')
        ax0.legend()
        ax0.grid(True)
        fig0.tight_layout()

        # Plot 2: Cumulative and Local Cumulative
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(x, Residual_corrected/1000, color='blue',
                   label='Residual Corrected (efficiency-adjusted)', align='edge')
        ax1.bar(x, Residual/1000, color='lightskyblue',
                   label='Residual (Demand - RES)', align='edge')
        ax1.plot(Local_cumul/1000, label='Local Cumulative (Storage Level)',
                    color='green', marker='.')
        ax1.plot(Cummul_res_corr/1000, label='Cumulative Residual Corrected',
                    color='blue', marker='.')
        ax1.axhline(Capacity_req/1000, color='red', linestyle='--', linewidth=2,
                       label='Required Energy Capacity')
        ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
        # ax1.set_title('Cumulative Imbalance and Virtual Storage Level')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Energy [GWh]')

        # Get handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Create a dictionary and sort it as desired
        label_order = ['Residual (Demand - RES)', 
                       'Residual Corrected (efficiency-adjusted)',
                       'Cumulative Residual Corrected',
                       'Local Cumulative (Storage Level)',
                       'Required Energy Capacity']
        sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: label_order.index(x[1]))

        # Unzip and pass to legend
        sorted_handles, sorted_labels = zip(*sorted_handles_labels)
        ax1.legend(sorted_handles, sorted_labels)

        ax1.grid(True)
        fig1.tight_layout()
        fig1.savefig(f"{season}-storage_capacity.pdf")


    return OC_all, Eta_all, E_max_all, Q_max_all, Q_all

