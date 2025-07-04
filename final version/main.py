# Import relevant packages
import gurobipy as gp                 # Gurobi Python API
from gurobipy import GRB              # Gurobi constants (e.g., GRB.MAXIMIZE)

import pandas as pd                   # DataFrames
import matplotlib.pyplot as plt       # Plotting
import numpy as np                    # Numerical operations (similar to Julia base)

import warnings
warnings.filterwarnings("ignore", message=".*All values for SymLogScale are below linthresh.*")

from functions_data import *
from functions_plots import *
from functions_model import *

## === Initialization of the problem ===
# Set changing parameters
season = "Winter"           # Modelled season \in {"Winter", "Summer", "LowLoad"}
plots = False
data_plots = True
bidding_zone = "DK2"        # Modelled Denmark bidding zone \in {"DK1", "DK2"} for price demand curve
n_players = 4               # Number of storage players in the Cournot game \in {1, 2, 4, 6, 8}
alpha_batt = 0.5            # Initial storage level (%)
min_eta = 0.85              # Minimal storage round-trip efficiency
OC_default = 5              # Default storage operating cost
storage_Crate_default = 0.5 # Charge/discharge rate relative to energy capacity. A 1C battery can discharge fully in 1 hour.
N = 10                      # Discretization number for power outputs
D = 20                      # Discretization number for price demand curve steps
tol = 1e-5                  # Nash equilibrium tolerance parameter
max_iter = 50               # Nash equilibrium maximum iteration number

# Diverse parameters
diff_table_initial = []     # Store the difference between model outputs for each iteration

# Set time horizon parameters
T = 24              # number of time periods
TIME = range(T)    # time periods iterable


def main():
    ## === DATA LOADING ===
    ## Load price demand curves
    Demand_price, Demand_volume = load_price_demand_curve_data(bidding_zone=bidding_zone, time_period=season, demand_step_numbers=D, plots=data_plots)
    Demand_volume_total = Demand_volume[D-1,:]   # Last row (Total accumulated volume)

    ## Load RES profile
    RES = load_res_production_data(season, plots=data_plots)

    # Compute hourly residual demand (<0 if residual production)
    Residual = -RES + Demand_volume_total

    ## Load Battery Parameters
    OC_all, Eta_all, E_max_all, Q_max_all, Q_all = load_storage_data(Residual, n_players, min_eta, storage_Crate_default, OC_default, N, 
                                                                    plots=True, bidding_zone=bidding_zone, season=season)

    ## === PLOTTING: loaded data for the modelled scenario ===
    fig, axs = plt.subplots(2, 2, figsize=(14, 7))
    plot_price_demand_curve(axs[0,0], Demand_price, Demand_volume)
    plot_demand_over_time(axs[0, 1], Demand_volume_total)
    plot_renewable_over_time(axs[1, 0], RES, Demand_volume_total)
    plot_residual_over_time(axs[1, 1], Residual)
    fig.tight_layout()
    if plots:
        fig.savefig(f"{bidding_zone+season}-market_data.png")

    ## === MODEL ===
    # Setting values to initialize the run
    q_ch_assumed_ini = [0 for _ in TIME]
    q_dis_assumed_ini = [0 for _ in TIME]

    model_parameters = [max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table_initial]
    storage_parameters = [alpha_batt, OC_all, Eta_all, E_max_all, Q_max_all, Q_all]

    # Cournot iteration of the profit optimization model
    output, ne, iter, u, profits, diff_table = nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, n_players, model_parameters, storage_parameters, tol)
    plot_results(output, profits, diff_table, n_players, model_parameters, storage_parameters)
    plt.show()


if __name__ == '__main__':
    main()
