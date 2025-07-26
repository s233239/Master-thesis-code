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


def main(season="Winter", bidding_zone="DK2", n_players=4, 
         storage_capacity=None,
         alpha_batt=0.5, min_eta=0.85, OC_default=0.5, storage_Crate_default=0.5, N=10, 
         D=20, tol=1e-5, max_iter=200,
         plots=True, data_plots=False, scenario_plots=False, save_plots=False,
         policy_type="none", policy_parameters=None, reserve_policy=False):
    
    # Diverse parameters
    diff_table_initial = []     # Store the difference between model outputs for each iteration

    # Set time horizon parameters
    T = 24              # number of time periods
    TIME = range(T)    # time periods iterable

    ## === DATA LOADING ===
    ## Load price demand curves
    Demand_price, Demand_volume = load_price_demand_curve_data(bidding_zone=bidding_zone, time_period=season, demand_step_numbers=D, plots=data_plots)
    Demand_volume_total = Demand_volume[D-1,:]   # Last row (Total accumulated volume)

    ## Load RES profile
    RES = load_res_production_data(season, plots=data_plots)

    # Compute hourly residual demand (<0 if residual production)
    Residual = -RES + Demand_volume_total

    # Final initialization for model
    model_parameters = [max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table_initial]

    ## === NO STORAGE ===
    if n_players == 0:
        output = market_clearing_no_storage(model_parameters, plots=plots)
        return output

    ## Load Battery Parameters
    OC_all, Eta_all, E_max_all, Q_max_all, Q_all = load_storage_data(Residual, n_players, min_eta, storage_Crate_default, OC_default, N,
                                                                     storage_capacity=storage_capacity, 
                                                                     plots=scenario_plots, bidding_zone=bidding_zone, season=season)

    ## === PLOTTING: loaded data for the modelled scenario ===
    if scenario_plots:
        fig, axs = plt.subplots(2, 2, figsize=(14, 7))
        plot_price_demand_curve(axs[0,0], Demand_price, Demand_volume)
        plot_demand_over_time(axs[0, 1], Demand_volume_total)
        plot_renewable_over_time(axs[1, 0], RES, Demand_volume_total)
        plot_residual_over_time(axs[1, 1], Residual)
        fig.tight_layout()
        if save_plots:
            fig.savefig(f"{bidding_zone+season}-market_data.png")

    ## === MODEL ===
    # Setting values to initialize the run
    q_ch_assumed_ini = [0 for _ in TIME]
    q_dis_assumed_ini = [0 for _ in TIME]
    storage_parameters = [alpha_batt, OC_all, Eta_all, E_max_all, Q_max_all, Q_all]

    # Cournot iteration of the profit optimization model
    output, ne, iter, u, profits, diff_table = nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, n_players, 
                                                       model_parameters, storage_parameters, 
                                                       policy_type, policy_parameters, reserve_policy,
                                                       tol)
    if plots:
        plot_results(output, profits, diff_table, n_players, model_parameters, storage_parameters, save_plots, bidding_zone, season)
        plt.show()

    return output


if __name__ == '__main__':
    main()
