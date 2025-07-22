"""
Plotting utilities for electricity market data analysis.

This module provides functions to generate common visualizations such as
demand curves, renewable production, and residual demand over time.
Each function is designed to take preprocessed data and a matplotlib axes
object for flexible subplot integration.

Functions follow a consistent interface: 
    plot_x(ax, data1, data2, ...)

Currently available:
    - plot_price_demand_curve
    - plot_demand_over_time
    - plot_renewable_over_time
    - plot_residual_over_time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

from functions_policy import *


## Result Plots
def plot_results(output, profits, diff_table, n_players, model_parameters, storage_parameters,
                 plots=False, bidding_zone=None, season=None):
    """
    Generate and display plots for Cournot simulation results, including market dynamics,
    battery operations, and welfare metrics.

    Parameters:
        output (list): 
            Nested output list from the storage optimization model for each player.
        profits (list): 
            List of lists showing player profits over Cournot iterations.
        n_players (int): 
            Number of storage players in the simulation.
        model_parameters (list): 
            Model-wide variables including max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table_initial.
        storage_parameters (list): 
            Storage-related parameters including alpha_batt, OC_all, Eta_all, E_max_all, Q_max_all, Q_all.
        plots (bool, optional): 
            Whether to save generated plots as PNG files. Default is False.
        bidding_zone (str, optional): 
            Name of the bidding zone used for filename saving. Required if plots=True.
        season (str, optional): 
            Name of the season used for filename saving. Required if plots=True.

    Returns:
        None:
            Plots are shown (and optionally saved), but no data is returned.
    """


    
    ## === EXPORT RESULTS ===
    [max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table_initial] = model_parameters
    [alpha_batt, OC_all, Eta_all, E_max_all, Q_max_all, Q_all] = storage_parameters

    # 1. Proad = Discharge - Charge for each player and time
    proad = [
        [output[player][1][t] - output[player][0][t] for t in TIME]
        for player in range(n_players)
    ]

    # 2. Battery storage level per player
    batt = [
        [E_max_all[player] * alpha_batt] + [output[player][2][t] for t in TIME]
        for player in range(n_players)
    ]

    # 3. Market price over time
    # for p in range(1,n_players):
    #     if output[p][3] != output[0][3]:
    #         raise "Error in convergence"

    # Now we can assume each player outputs the same market price, CS, PS, SW etc.
    market_price = [output[0][3][t] for t in TIME]

    # 4. Revenue per player and time
    revenue = [
        [output[player][4][t] for t in TIME]
        for player in range(n_players)
    ]

    # 5. Total profit per player
    profit_tot = [sum(revenue[player]) for player in range(n_players)]
    profit_tot_by_cap = [profit_tot[p]/E_max_all[p] if E_max_all[p]!=0 else 0 for p in range(n_players)]

    # 6. Total quantity offered to the market
    supply_total = [sum(proad[player][t] for player in range(n_players) if proad[player][t] >= 0) for t in TIME]   # positive for supply
    demand_total = [sum(proad[player][t] for player in range(n_players) if proad[player][t] < 0) for t in TIME]    # negative for demand
    proad_total = [supply_total[t] + demand_total[t] for t in TIME]
    q_total = [RES[t] + proad_total[t] for t in TIME]

    # 7. Unmet demand
    unmet_demand = [max(Demand_volume[-1, t] - q_total[t], 0) for t in TIME]

    # 8. Curtailed production
    curtailed_prod = [max(-Demand_volume[-1, t] + q_total[t], 0) for t in TIME]

    # 9. Consumer Surplus
    CS = np.array([output[0][7][t] for t in TIME])

    # 10. Producer Surplus
    PS = np.array([
        sum(revenue[player][t] for player in range(n_players)) + 
        (RES[t] - curtailed_prod[t]) * market_price[t]
        for t in TIME
    ])

    # 11. Social Welfare
    SW = CS + PS


    ## === PLOTTING ===
    plt.figure(figsize=(14,7))
    temps_np = np.array(TIME)
    temps_with_zero_np = np.array([t for t in TIME] + [T])

    # 1. Market Price Plot
    plt.subplot(2,2,1)

    values_to_show = [round(p,2) for p in market_price if p > 0]
    values_to_show.sort()
    values_to_show_filtered = [x for i, x in enumerate(values_to_show) if i == 0 or abs(x - values_to_show[i-1]) >= 2]
    index=1
    while len(values_to_show_filtered) > 4:
        values_to_show_filtered.remove(values_to_show_filtered[index])
        index += 1
        if index >= len(values_to_show_filtered):
            index = index // 2

    for player in range(n_players):
        plt.step(temps_with_zero_np, np.append(output[player][3], output[player][3][-1]), where='post')
    for p in values_to_show_filtered:
        plt.axhline(y=p, linestyle='--', color='gray', linewidth=1)
        plt.text(x=temps_with_zero_np[-1]+1.5, y=p, s=f'y={round(p)}', color='black', ha='left', va='bottom')
    plt.xlabel("Time (h)")
    plt.ylabel("Market Price (€/MWh)")
    plt.title("Market Price Over Time")
    plt.grid(True)


    # 2. Market Clearing View
    plt.subplot(2,2,2)

    plt.step(temps_with_zero_np, np.append(Demand_volume[-1, :], Demand_volume[-1, -1]), label="Demand", where='post', color='red', linestyle='--') 
    plt.bar(temps_np+0.5, RES, label="RES Production", color='green')
    plt.bar(temps_np+0.5, supply_total, label="Storage Discharge", color='blue', bottom=RES)
    plt.bar(temps_np+0.5, demand_total, label="Storage Charge", color='deepskyblue', bottom=0)
    plt.xlabel("Time (h)")
    plt.ylabel("Power (MW)")
    bottom, top = plt.ylim()
    plt.ylim(top=top*1.2)
    plt.legend(loc='upper left')
    plt.title("Market Clearing: Supply vs Demand Over Time")

    # 3. Summary Bars for Unmet Demand, Curtailment and Market Metrics
    ax1 = plt.subplot(2,2,3)

    def engineering_notation(x, precision=3):
        if x == 0:
            return f"0"
        exponent = int(np.floor(np.log10(abs(x)) // 3 * 3))
        mantissa = x / (10 ** exponent)
        return f"{mantissa:.{precision}g}e{exponent}"

    # === Ax1: Energy metrics ===
    ax1_labels = ["Unmet Demand", "Curtailed Production"]
    ax1_heights = [sum(unmet_demand), sum(curtailed_prod)]
    x1 = np.arange(len(ax1_labels))

    bars1 = ax1.bar(x1, ax1_heights, width=0.5, color='tab:red', label="Energy Metrics")
    ax1.bar_label(bars1, [f"{engineering_notation(x)} MWh" for x in ax1_heights])
    ax1.set_ylabel("Energy (MWh)")
    ax1.set_ylim(0, max(ax1_heights) * 10)
    ax1.set_yscale('symlog', linthresh=1e2)
    ax1.tick_params(axis='y', colors='tab:red')

    # === Ax2: Economic metrics ===
    ax2_labels = ["Consumer Surplus", "Producer Surplus", "Social Welfare"]
    ax2_heights = [np.average(CS), np.average(PS), np.average(SW)]                # Can be improved by stacking each CS[t], PS[t]
    x2 = np.arange(len(ax2_labels)) + len(x1) + 0.5     # offset to avoid overlap

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x2, ax2_heights, width=0.5, color='tab:purple', label="Welfare Metrics")
    ax2.bar_label(bars2, [f"{engineering_notation(x)} €/h" for x in ax2_heights])
    ax2.set_ylabel("Average Amount per Hour (€/h)")
    ax2.set_ylim(0, max(ax2_heights) * 10)
    ax2.set_yscale('symlog', linthresh=10)
    ax2.tick_params(axis='y', colors='tab:purple')

    xticks = np.concatenate([x1, x2])
    xlabels = ax1_labels + ax2_labels
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels, rotation=20)

    ax1.set_title("Market Metrics")

    # 4. Optimized Profits
    ax1 = plt.subplot(2,2,4)

    player_labels = [f"{chr(65 + p)}" for p in range(n_players)]
    width = 0.3
    x = np.arange(1,n_players+1,1)

    container = ax1.bar(x=x-width/2, height=profit_tot, width=width, color="tab:blue")
    ax1.bar_label(container, [f"{round(p)} €" for p in profit_tot])
    ax1.set_ylim(top=1.1*max(profit_tot))
    ax1.set_ylabel("Total profit (€)")
    ax1.tick_params(axis='y', colors='tab:blue')

    ax2 = ax1.twinx()
    container = ax2.bar(x=x+width/2, height=profit_tot_by_cap, width=width, color="tab:orange")
    ax2.bar_label(container, [f"{round(p,2)} €/MWh" for p in profit_tot_by_cap])
    ax2.set_ylim(top=1.2*max(profit_tot_by_cap))
    ax2.set_ylabel("Profit by Installed Capacity Unit (€/MWh)")
    ax2.tick_params(axis='y', colors='tab:orange')

    ax1.set_title("Player Optimal Profits over the Period")
    ax1.set_xticks(x)
    ax1.set_xticklabels(player_labels)

    plt.tight_layout()
    if plots:
        plt.savefig(f"{bidding_zone+season}-{n_players}players-main_market_results.pdf")


    # 5. Production and SoC per Player
    # Ax 1 for energy storage levels, ax 2 for energy storage discharging/charging power
    fig, ax1 = plt.subplots(figsize=(14,7))

    for player in range(n_players):
        ax1.plot(temps_with_zero_np, batt[player], label=f"SoC for Player {player + 1}")
    ax1.set_ylim(bottom=0)  # top=max(E_max_all)
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Battery State of Charge (MWh)")
    ax1.legend(loc="upper left")
    ax1.set_title("Battery Cycle")

    ax2 = ax1.twinx()
    for player in range(n_players):
        ax2.step(temps_with_zero_np, proad[player] + [proad[player][-1]], where="post", label=f"Supply from Player {player + 1}", linestyle='--', linewidth=0.9)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_ylabel('Power [MW]')
    ax2.legend(loc='upper right')
    ax2.grid()

    fig.tight_layout()
    if plots:
        plt.savefig(f"{bidding_zone+season}-{n_players}players-storage_soc.pdf")


    # 6. Nash Equilibrium Result
    plt.figure(figsize=(14,7))
    x = range(1, len(profits[0]) + 1)
    if len(x) <= 20:
        xticks = np.array([1]+[2+2*i for i in range(int(np.floor(len(x)/2)))])
    elif len(x) <= 50:
        xticks = np.array([1]+[5+5*i for i in range(int(np.floor(len(x)/5)))])
    elif len(x) <= 100:
        xticks = np.array([1]+[10+10*i for i in range(int(np.floor(len(x)/10)))])
    else:
        xticks = np.array([1]+[20+20*i for i in range(int(np.floor(len(x)/20)))])


    plt.subplot(2,2,1)
    plt.plot(x[1:], diff_table, label="Max Change per Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Number of Computed Difference")
    plt.xticks(xticks)   
    plt.title("Cournot Iteration Convergence Plot")
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,2)
    for player in range(n_players):
        plt.plot(x, profits[player], label=f"Player {player+1} Profit")
    plt.xlabel("Iteration")
    plt.ylabel("Profit (€)")
    xticks[0]=2
    plt.xticks(xticks)       
    plt.title("Profit Evolution over Cournot Iteration")
    plt.ylim(bottom = 0)
    plt.grid(True)
    plt.legend()

    plt.subplot(2,2,4)
    for player in range(n_players):
        plt.plot(x[2:], [profits[player][-1] - profits[player][i] for i in range(2,len(profits[player]))], label=f"Player {player+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Profit (€)")
    # xticks[0]=2
    # plt.xticks(xticks)       
    plt.title("Zoom on Profit Evolution")
    # plt.ylim(bottom = 0)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    if plots:
        plt.savefig(f"{bidding_zone+season}-{n_players}players-cournot_metrics.pdf")

    return


def plot_scenarios_analysis(outputs, model_parameters=None):
    """
    Generate comparative plots of market and player metrics across multiple policy scenarios.

    Parameters:
        outputs (dict):
            Dictionary mapping scenario names (str) to output data dictionaries.
            Each output dict is indexed by player and contains lists of time series data.
    """
    if model_parameters:
        [max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table_initial] = model_parameters

    scenario_names = list(outputs.keys())
    n_scenarios = len(scenario_names)
    scenario_colors = mpl.color_sequences['tab10'][:n_scenarios]
    scenario_linestyles = ['-', ':', '--', '-.', ''][:n_scenarios]
    n_players = len(outputs[scenario_names[0]])
    T = len(outputs[scenario_names[0]][0][3])
    TIME = range(T)
    
    # --- 1) Market price over time ---
    plt.figure(figsize=(14, 7))
    plt.subplot(2,2,1)
    for j, scen in enumerate(scenario_names):
        # Price is same across players, take player 0 for reference
        price = outputs[scen][0][3]  
        plt.step(TIME, price, label=scen, color=scenario_colors[j], linestyle=scenario_linestyles[j])
    plt.title("Market Price Over Time")
    plt.xlabel("Time")
    plt.ylabel("Price (€/MWh)")
    plt.legend()
    plt.grid()


    # --- 2) Unmet demand over time ---
    plt.subplot(2,2,2)
    for j, scen in enumerate(scenario_names):
        unmet = outputs[scen][0][9]
        plt.step(TIME, unmet, label=scen, color=scenario_colors[j], linestyle=scenario_linestyles[j])
    plt.title("Unmet Demand Over Time")
    plt.xlabel("Time")
    plt.ylabel("Unmet Demand (MW)")
    plt.legend()
    plt.grid()


    # # --- 3) Curtailed production over time ---
    # plt.subplot(2,2,3)
    # for j, scen in enumerate(scenario_names):
    #     curtailed = outputs[scen][0][10]
    #     plt.step(TIME, curtailed, label=scen, color=scenario_colors[j])
    # plt.title("Curtailed Production Over Time")
    # plt.xlabel("Time")
    # plt.ylabel("Curtailed Production (MW)")
    # plt.legend()
    # plt.grid()

    # --- 3) Charging behavior over time ---
    plt.subplot(2,2,3)
    for i, scen in enumerate(scenario_names):
        output = outputs[scen]
        total_q_ch = [sum(output[p][0][t] for p in range(n_players)) for t in TIME]
        plt.step(TIME, total_q_ch, where='post', label=scen, color=scenario_colors[i], linestyle=scenario_linestyles[i])

    plt.title("Total Charging Behavior Across Scenarios")
    plt.xlabel("Hour")
    plt.ylabel("Charging Power [MW]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    # --- 4) Profits per player grouped bar chart ---
    # Build matrix of profits: shape (n_players, n_scenarios)
    profits_matrix = np.zeros((n_players, n_scenarios))
    policy_addon = np.zeros((n_players, n_scenarios))
    for j, scen in enumerate(scenario_names):
        for p in range(n_players):
            # sum revenue as profit proxy, or have dedicated profit in outputs if available
            profits_matrix[p, j] = sum(outputs[scen][p][4])  # sum over time of revenue
            policy_addon[p, j] = sum(outputs[scen][p][5])  # sum over time of revenue

    # Plot grouped bar chart
    bar_width = 0.15
    x = np.arange(n_players)

    plt.subplot(2,2,4)
    for i, scen in enumerate(scenario_names):
        plt.bar(x + i*bar_width, profits_matrix[:, i], width=bar_width, label=f'{scen} Base Revenue', color=scenario_colors[i], edgecolor='black')
        plt.bar(x + i*bar_width, policy_addon[:, i], width=bar_width, bottom=profits_matrix[:, i], label=f'{scen} Tariff Cost', hatch='//', color='gray', edgecolor='black')

    # Create custom legend for scenarios
    scenario_handles = [Patch(color=scenario_colors[i], label=scenario_names[i]) for i in range(n_scenarios)]

    # Create custom legend for revenue types
    type_handles = [
        # Patch(facecolor='gray', edgecolor='black', label='Market revenue'),
        Patch(facecolor='gray', edgecolor='black', hatch='//', label='tariff cost')
    ]

    # # First legend
    # leg1 = plt.legend(handles=scenario_handles, loc='upper left')
    # # Second legend, added manually
    # leg2 = plt.legend(handles=type_handles, loc='upper right')
    # ax = plt.gca()
    # ax.add_artist(leg1)
    plt.legend(handles=scenario_handles + type_handles, loc="upper left", ncol=2)


    plt.xticks(x + bar_width * (n_scenarios - 1) / 2, [f'Player {p+1}' for p in range(n_players)])
    plt.ylabel("Total Profit (€)")
    plt.ylim(top=1.1*np.max(profits_matrix))
    plt.title("Player Profits by Scenario")
    plt.grid(axis='y')

    plt.tight_layout()


    # --- Other subplot: Revenue over time (aggregated over players) ---
    bar_width = 0.8 / n_scenarios
    x = np.arange(len(TIME))

    plt.figure(figsize=(14, 7))

    for j, scen in enumerate(scenario_names):
        # Aggregate over players
        base_revenue = np.sum([outputs[scen][p][4] for p in range(n_players)], axis=0)
        tariff_addon = np.sum([outputs[scen][p][5] for p in range(n_players)], axis=0)

        x_offset = x - 0.4 + j * bar_width

        plt.bar(x_offset, base_revenue, width=bar_width, label=f"{scen} Base", color=scenario_colors[j], edgecolor='black')
        plt.bar(x_offset, tariff_addon, width=bar_width, bottom=base_revenue, 
                label=f"{scen} Tariff", hatch='//', color='gray', edgecolor='black')

    plt.xlabel("Time (Hour)")
    plt.ylabel("Revenue (€)")
    plt.title("Scenario Comparison: Revenue and Tariff Component per Hour")
    plt.xticks(x, [str(t) for t in TIME])
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Custom legend: one per scenario and one for tariff layer
    scenario_legend = [Patch(color=scenario_colors[i], label=scenario_names[i]) for i in range(n_scenarios)]
    tariff_legend = [Patch(facecolor='gray', hatch='//', label="tariff cost", edgecolor='black')]
    plt.legend(handles=scenario_legend + tariff_legend, loc="upper right", ncol=2)
    plt.tight_layout()


    # --- Tariffs Plots ---

    # # Plot tariffs
    # plt.figure(figsize=(10, 3))
    # plt.step(range(24), tau_ch, where='mid')
    # plt.step(range(24), tau_dis, where='mid')
    # plt.step(range(24), residual_series, where='mid', color='red', linestyle='--')
    # plt.title("Dynamic Tariffs (€/MWh)")
    # plt.xlabel("Time (h)")
    # plt.ylabel("Tariff")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.xticks(range(24))
    # plt.tight_layout()


## Price Demand Curve
def plot_price_demand_curve(ax, Demand_price, Demand_volume):
    """
    Plot hourly price-demand step curves.

    Parameters:
        ax (matplotlib.axes.Axes): Plotting axes
        Demand_price (DataFrame Nx24): Price per demand step per hour
        Demand_volume (DataFrame Nx24): Volume per demand step per hour
    """
    temps = Demand_price.shape[1]
    for t in range(temps):
        ax.step(
            np.insert(Demand_volume[:, t], 0, 0),
            np.insert(Demand_price[:, t], 0, Demand_price[0, t]),
            where='post',
            label=f"Hour {t+1}"
        )
    ax.set_xlabel("Volume (MWh)")
    ax.set_ylabel("Price (€/MWh)")
    ax.set_title("Price Demand Curve")
    ax.grid()


## Demand Over Time
def plot_demand_over_time(ax, Demand_volume_total):
    """
    Plot total demand per hour.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes
        Demand_volume_total (np.ndarray): Total hourly demand
    """
    temps = Demand_volume_total.shape[0]
    ax.plot(Demand_volume_total, color="red", marker='.')
    ax.bar(x=range(temps), height=Demand_volume_total, color='red', alpha=0.5, align='edge')
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Hour (h)")
    ax.set_ylabel("Cumulated demand (MWh)")
    ax.set_title("Demand Over Time")
    ax.grid()


## Renewable Production Over Time
def plot_renewable_over_time(ax, RES, Demand_volume_total):
    """
    Plot RES production and demand comparison.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes
        RES (Series or array): Renewable hourly production
        Demand_volume_total (np.ndarray): Hourly demand
    """
    temps = Demand_volume_total.shape[0]
    ax.plot(RES, color="green", marker='.')
    ax.bar(x=range(temps), height=RES, color='green', alpha=0.5, align='edge')
    ax.plot(Demand_volume_total, color='red', linestyle='--', linewidth=1, label="Total Demand")
    ax.set_xlabel("Hour (h)")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Renewable Production Over Time")
    ax.legend()
    ax.grid()


## Residual Demand Over Time
def plot_residual_over_time(ax, Residual):
    """
    Plot hourly residual demand.

    Parameters:
        ax (matplotlib.axes.Axes): Target axes
        Residual (np.ndarray): Residual hourly demand
    """
    temps = Residual.shape[0]
    ax.plot(Residual, color="red", marker='.')
    ax.bar(x=range(temps), height=Residual, color='red', alpha=0.5, align='edge')
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Power (MW)")
    ax.set_title("Residual Demand Over Time")
    ax.grid()
