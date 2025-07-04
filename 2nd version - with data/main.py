# Import relevant packages
import gurobipy as gp                 # Gurobi Python API
from gurobipy import GRB              # Gurobi constants (e.g., GRB.MAXIMIZE)

import pandas as pd                   # DataFrames
import matplotlib.pyplot as plt       # Plotting
import numpy as np                    # Numerical operations (similar to Julia base)
from operator import itemgetter

from functions_data import *
from functions_plots import *
import os

## === Initialization of the problem ===

# Set changing parameters
season = "Summer"           # Modelled season \in {"Winter", "Summer", "LowLoad"}
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
max_iter = 100              # Nash equilibrium maximum iteration number


# Diverse parameters
diff_table = []     # Store the difference between model outputs for each iteration

# Set time horizon parameters
T = 24              # number of time periods
TIME = range(T)    # time periods iterable


## === DATA LOADING ===
## Load price demand curves
Demand_price, Demand_volume = load_price_demand_curve_data(bidding_zone=bidding_zone, time_period=season, demand_step_numbers=D, plots=data_plots)
Demand_volume_total = Demand_volume.loc[D-1].to_numpy()   # Last row (Total accumulated volume)


## Load RES profile
RES = load_res_production_data(season, plots=data_plots)

# Compute hourly residual demand (<0 if residual production)
Residual = -RES + Demand_volume_total


## Load Battery Parameters
OC_all, Eta_all, E_max_all, Q_max_all, Q_all = load_storage_data(Residual, n_players, min_eta, storage_Crate_default, OC_default, N, 
                                                                 plots=True, bidding_zone=bidding_zone, season=season)


## === PLOTTING: loaded data for the modelled scenario ===
fig, axs = plt.subplots(2, 2, figsize=(15, 7))
plot_price_demand_curve(axs[0,0], Demand_price, Demand_volume)
plot_demand_over_time(axs[0, 1], Demand_volume_total)
plot_renewable_over_time(axs[1, 0], RES, Demand_volume_total)
plot_residual_over_time(axs[1, 1], Residual)
fig.tight_layout()
if plots:
    fig.savefig(f"{bidding_zone+season}-market_data.png")



# Function containing the optimisation model
def model_run(q_ch_assumed, q_dis_assumed, player, state_ini=([],[])):

    # Initialization of model parameters
    # Data: RES[t], Demand_price[j,t], Demand_volume[j,t] - t in TIME, j in range(D)
    forecasted_production = np.array([RES[t] + q_dis_assumed[t] - q_ch_assumed[t] for t in TIME])

    if any(forecasted_production < 0):
        raise "Negative production: charging when no RES available"

    # Adjust demand curve as a price quota curve (based on residual demand) characterized by residual_demand_volume[j,t], residual_demand_price[j,t]
    residual_demand_volume = np.array([[Demand_volume[j,t] - forecasted_production[t] for t in TIME] for j in range(D)])
    residual_demand_volume = np.where(residual_demand_volume < 0, 0, residual_demand_volume)

    residual_demand_price = np.zeros((D,T))
    for j in range(D):
        for t in TIME:
            if residual_demand_volume[j,t] > 0:
                residual_demand_price[j,t] = Demand_price[j,t]
            else:
                residual_demand_price[j,t] = 0


    # Initialize price quota curve parameters
    step_min_demand = np.insert(residual_demand_volume[:-1,:], 0, 0, axis=0)
    step_max_additional_demand = np.array([[residual_demand_volume[j,t] - step_min_demand[j,t] for t in TIME] for j in range(D)])

    # Used to constrain charging variables
    residual_production = np.array([forecasted_production[t] - Demand_volume[-1,t] for t in TIME])
    residual_production = np.where(residual_production < 0, 0, residual_production)


    # Initiate optimization process
    model = gp.Model()
    model.Params.OutputFlag = 0
    # model.Params.NonConvex = 2  # Allow for quadratic constraints

    z_ch = model.addVars(TIME, range(N), vtype=GRB.BINARY, name="z_ch")
    z_dis = model.addVars(TIME, range(N), vtype=GRB.BINARY, name="z_dis")
    e = model.addVars(TIME, lb=0, ub=E_max_all[player], name="e")
    u = model.addVars(TIME, range(D), vtype=GRB.BINARY, name="u")
    b = model.addVars(TIME, range(D), lb=0, name="b")
    q = model.addVars(TIME, lb=0, ub=Q_max_all[player], name="q")

    # Define expression of variables
    q_ch = {t: gp.quicksum(Q_all[player][i] * z_ch[t, i] for i in range(N)) for t in TIME}
    q_dis = {t: gp.quicksum(Q_all[player][i] * z_dis[t, i] for i in range(N)) for t in TIME}
    revenue = {
        t:  gp.quicksum(residual_demand_price[j,t] * (b[t,j] + u[t,j] * step_min_demand[j,t]) for j in range(D)) - 
        OC_all[player] * (q_dis[t] + q_ch[t])
        for t in TIME
    }

    # Linear objective function
    model.setObjective(gp.quicksum(revenue[t] for t in TIME), GRB.MAXIMIZE)

    ## Storage feasible operating region / technical constraints
    # Energy storage intertemporal constraints: SOC update
    model.addConstr(e[0] == E_max_all[player] * alpha_batt + Eta_all[player] * q_ch[0] - q_dis[0])
    model.addConstrs(e[t] == e[t-1] + Eta_all[player] * q_ch[t] - q_dis[t] for t in range(1, T))
    model.addConstr(e[T-1] >= E_max_all[player] * alpha_batt)

    # Only one storage action possible: idle, charge or discharge at a fixed power rate
    model.addConstrs(gp.quicksum(z_ch[t, i] + z_dis[t, i] for i in range(N)) <= 1 for t in TIME)

    ## Identification of the market price > linearization
    # Only a single active price level (corresponding to the market clearing price)
    model.addConstrs(gp.quicksum(u[t, j] for j in range(D)) == 1 for t in TIME)

    # Identify amount of demand volume satisfied (for last demand step)
    model.addConstrs(b[t,j] <= u[t,j] * step_max_additional_demand[j,t]
                     for t in TIME for j in range(D))
    
    # Constrain the discharging power to be equal to the satisfied demand ("balance equation")
    model.addConstrs(q[t] == q_dis[t] for t in TIME)
    model.addConstrs(q[t] == gp.quicksum(b[t,j] + u[t,j] * step_min_demand[j,t] for j in range(D)) for t in TIME)

    # In the case of residual production:
    # Market price = 0 because of data characteristics 
    # Charging is possible from residual RES
    model.addConstrs(q_ch[t] <= residual_production[t] for t in TIME)
    
    if isinstance(state_ini[0], np.ndarray):
        # Force convergence
        model.addConstrs(z_ch[t,i] == state_ini[0][t,i] for t in TIME for i in range(N))
        model.addConstrs(z_dis[t,i] == state_ini[1][t,i] for t in TIME for i in range(N))


    # Optimization of the model
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"Model status: {model.status}")

        if model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            model.computeIIS()
            model.write("infeasible.ilp")

        model.Params.OutputFlag = 1
        model.Params.LogFile = "gurobi_log.txt"
        model.write("myLP_model.lp")

        return None

    # Return outputs
    state = [[z_ch[t, i].X for i in range(N)] for t in TIME], \
            [[z_dis[t, i].X for i in range(N)] for t in TIME]
    
    u = [[u[t, j].X for j in range(D)] for t in TIME]

    y = [[1 - sum(u[t][k] for k in range(j+1)) for j in range(D)] for t in TIME]

    CS = [sum((Demand_price[j, t] - Demand_price[j+1, t]) * Demand_volume[j, t] * y[t][j]
                for j in range(D-1)) for t in TIME]    # not necessary to include computation for last satisfied load bc it sets the price hence does not increase CS

    price = [sum(residual_demand_price[j,t] * u[t][j] for j in range(D)) for t in TIME]

    output = [[q_ch[t].getValue() for t in TIME],
              [q_dis[t].getValue() for t in TIME],
              [e[t].X for t in TIME],
              price,
              [revenue[t].getValue() for t in TIME],
              CS]
    

    return state, output, u


def arrays_are_equal(a1, a2, n_players, tol=1e-7):
    if not a2:
        return False 

    diff = 0
    for p in range(n_players):
        a_new, a_old = np.array(a1[p]).flatten(), np.array(a2[p]).flatten()
        diff_p = np.sum(np.abs(a_new - a_old))
        diff = max(diff, diff_p)
    
    # Other method to compute convergence
    # a1, a2 = np.array(a1), np.array(a2)
    # diff = np.linalg.norm((a1 - a2)/a2)

    diff_table.append(diff)

    return diff < tol


def nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, n_players, tol=1e-7):

    ne = [[], []]
    state = {}
    output = {}
    u = {}
    profits = {p: [] for p in range(n_players)}
    
    iter = 0
    for player in range(n_players):
        # Initialize optimization model
        state[player], output[player], u[player] = model_run(q_ch_assumed_ini, q_dis_assumed_ini, player)

        # Store profits for later plots
        profits[player].append(sum(output[player][4][t] for t in TIME))


    # Store outputs of initialization models
    state_sys = [state[player] for player in range(n_players)]
    ne.append(state_sys.copy())
    iter += 1

    if n_players == 1:
        return output, ne, iter, u, profits

    while not arrays_are_equal(state_sys, ne[-2], n_players, tol) and iter < max_iter:

        # Profit maximisation for each player
        for player in range(n_players):
            q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in TIME]
            q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in TIME]
            state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player)
            
            # Store profits for later plots
            profits[player].append(sum(output[player][4][t] for t in TIME))

        state_sys = [state[player] for player in range(n_players)]
        ne.append(state_sys.copy())
        iter += 1

    if iter == max_iter:
        convergence = False
        # Iterate the profit maximisation for all players again WHILE fixing one or more players' decision variables
        for p in range(n_players-1):
            print(f"Convergence has not been reached. Let's try again by fixing player {chr(65 + p)} outputs.")

            for it in range(max_iter//10):

                for player in range(p+1):
                    q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in TIME]
                    q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in TIME]
                    state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player, 
                                                                         state_ini=(np.array(state[player][0]), np.array(state[player][1])))  
                    profits[player].append(sum(output[player][4][t] for t in TIME))      

                for player in range(p+1, n_players):
                    q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in TIME]
                    q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in TIME]
                    state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player)
                    profits[player].append(sum(output[player][4][t] for t in TIME))

                state_sys = [state[player] for player in range(n_players)]
                ne.append(state_sys.copy())
                iter += 1

                if arrays_are_equal(state_sys, ne[-2], n_players, tol):
                    print(f"Optimization was successful. It converged in {iter} iterations.")
                    convergence = True
                    break

                # Otherwise, continue iterating

            # If convergence has been reached on a nested loop, break as well the current iteration
            if convergence:
                break
            
            # Otherwise, loop back by fixing another player decision variables

        # All but one player's decision variables have been fixed, but it has not converged
        if not convergence:
            print("Optimization was unsuccessful.")

    else:
        print(f"Optimization was successful. It converged in {iter} iterations.")
        
    return output, ne, iter, u, profits


## === Setting values to initialize the run ===
q_ch_assumed_ini = [0 for _ in TIME]
q_dis_assumed_ini = [0 for _ in TIME]

output, ne, iter, u, profits = nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, n_players, tol)



## === Export results ===

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

# 3. Market price over time (assumed same for all players)
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
for p in range(1,n_players):
    if output[p][5] != output[0][5]:
        raise "Error in convergence"
    
CS = np.array([output[0][5][t] for t in TIME])    # Now we can assume each player outputs the same CS

# 10. Producer Surplus
PS = np.array([
    sum(revenue[player][t] for player in range(n_players)) + 
    RES[t] * market_price[t]
    for t in TIME
])

# 11. Social Welfare
SW = CS + PS


## === Plots ===
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
plt.tight_layout()


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
plt.tight_layout()

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
plt.tight_layout()

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
fig, ax1 = plt.subplots(figsize=(15,7))

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
if plots:
    plt.savefig(f"{bidding_zone+season}-{n_players}players-storage_soc.png")


# 6. Nash Equilibrium Result
plt.figure(figsize=(15,7))
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
if plots:
    plt.savefig(f"{bidding_zone+season}-{n_players}players-cournot_metrics.png")


# Adjust layout and show the plot
plt.show()


