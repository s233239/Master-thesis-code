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
diff_table = []     # Store the difference between model outputs for each iteration

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

model_variables = [max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table]
storage_variables = [alpha_batt, OC_all, Eta_all, E_max_all, Q_max_all, Q_all]

# Cournot iteration of the profit optimization model
output, ne, iter, u, profits = nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, n_players, model_variables, storage_variables, tol)


## === EXPORT RESULTS ===

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
for p in range(1,n_players):
    if output[p][3] != output[0][3]:
        raise "Error in convergence"

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
unmet_demand = sum(max(Demand_volume[-1, t] - q_total[t], 0) for t in TIME)

# 8. Curtailed production
curtailed_prod = sum(max(-Demand_volume[-1, t] + q_total[t], 0) for t in TIME)

# 9. Consumer Surplus
CS = np.array([output[0][5][t] for t in TIME])

# 10. Producer Surplus
PS = np.array([
    sum(revenue[player][t] for player in range(n_players)) + 
    RES[t] * market_price[t]
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
ax1_heights = [unmet_demand, curtailed_prod]
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
width = 0.4
x = np.arange(1,n_players+1,1)

container = ax1.bar(x=x-width/2, height=profit_tot, width=width, tick_label=player_labels, color="tab:blue")
ax1.bar_label(container, [f"{round(p)} €" for p in profit_tot])
ax1.set_ylim(top=1.1*max(profit_tot))
ax1.set_ylabel("Total profit (€)")
ax1.tick_params(axis='y', colors='tab:blue')

ax2 = ax1.twinx()
container = ax2.bar(x=x+width/2, height=profit_tot_by_cap, width=width, label=player_labels, color="tab:orange")
ax2.bar_label(container, [f"{round(p,2)} €/MWh" for p in profit_tot_by_cap])
ax2.set_ylim(top=1.2*max(profit_tot_by_cap))
ax2.set_ylabel("Profit by Installed Capacity Unit (€/MWh)")
ax2.tick_params(axis='y', colors='tab:orange')

ax1.set_title("Player Optimal Profits over the Period")

plt.tight_layout()
if plots:
    plt.savefig(f"{bidding_zone+season}-{n_players}players-main_market_results.png")


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
    plt.savefig(f"{bidding_zone+season}-{n_players}players-storage_soc.png")


# 6. Nash Equilibrium Result
plt.figure(figsize=(14,7))
x = range(1, len(profits[0]) + 1)
if len(x) <= 20:
    xticks = np.array([1]+[2+2*i for i in range(int(np.floor(len(x)/2)))])
elif len(x) <= 50:
    xticks = np.array([1]+[5+5*i for i in range(int(np.floor(len(x)/5)))])
elif len(x) <= 100:
    xticks = np.array([1]+[10+10*i for i in range(int(np.floor(len(x)/10)))])
elif len(x) <= 200:
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
    plt.plot(x, profits[player], label=f"Player {player+1} Profit Over Iteration")
plt.xlabel("Iteration")
plt.ylabel("Profit (€)")
xticks[0]=2
plt.xticks(xticks)       
plt.title("Profit Evolution over Cournot Iteration")
plt.ylim(bottom = 0)
plt.grid(True)
plt.legend()

plt.tight_layout()
if plots:
    plt.savefig(f"{bidding_zone+season}-{n_players}players-cournot_metrics.png")


# Adjust layout and show the plot
plt.show()


