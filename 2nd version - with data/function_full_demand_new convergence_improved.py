# Import relevant packages
import gurobipy as gp                 # Gurobi Python API
from gurobipy import GRB              # Gurobi constants (e.g., GRB.MAXIMIZE)

import pandas as pd                   # DataFrames
import matplotlib.pyplot as plt       # Plotting
import numpy as np                    # Numerical operations (similar to Julia base)

from joblib import Parallel, delayed  # For parallel computing (optional alternative: multiprocessing)

import data
import os

## --- Initialization of the problem ---

# Set changing parameters
season = "Summer"           # Modelled season \in {"Winter", "Summer"}
n_players = 4               # Number of storage players in the Cournot game \in {1, 2, 4, 6, 8}
factor = 1.4                # Scaling factor for RES production
alpha_batt = 0.5            # Initial storage level (%)
min_eta = 0.85              # Minimal storage round-trip efficiency
OC_default = 5              # Default storage operating cost
storage_Crate_default = 0.5 # Charge/discharge rate relative to energy capacity. A 1C battery can discharge fully in 1 hour.
N = 10                      # Discretization number for power outputs
tol = 1e-5                  # Nash equilibrium tolerance parameter
max_iter = 100              # Nash equilibrium maximum iteration number


# Diverse parameters
np.random.seed(101)
epsilon = 1e-5

# Set time horizon parameters_
T = 24              # number of time periods
temps = range(T)    # time periods iterable

# Load demand curves
D = len(data.LOADS) # number of loads (10)

# Demand_price = pd.DataFrame({d: [round(data.load_bids[d],2)] * T for d in data.LOADS}).T
Demand_price_array = np.sort(np.random.rand(D))[::-1] * data.load_cost
Demand_price = pd.DataFrame({d: [round(Demand_price_array[d],2)] * T for d in range(D)}).T
Demand_price.columns = range(T)
Demand_price.index = data.LOADS

Demand_volume = pd.DataFrame({d: [round(data.load_profile[d,t],2) for t in data.TIMES] for d in data.LOADS}).T
Demand_volume_cumulative = Demand_volume.cumsum(axis=0)
Demand_volume = Demand_volume_cumulative

Demand_volume_total = Demand_volume.iloc[-1, :].values

max_dem = data.load_capacity
min_dem = 0

# Load RES profile
csv_filename = "medoids_profile_summary--1cluster.csv"
data_dir = r'C:\Users\ppers\OneDrive\Documents\Cours DTU\MASTER THESIS\Codes\Master-thesis-code\data\csv--data_processing-v4'
csv_path = os.path.join(data_dir, csv_filename)

RES_profiles = pd.read_csv(csv_path)
print(RES_profiles)

# RES hourly capacity factors
offshore_profile_winter = RES_profiles.iloc[0].loc[[f"{i}" for i in temps]].to_numpy()
onshore_profile_winter = RES_profiles.iloc[2].loc[[f"{i}" for i in temps]].to_numpy()
solar_profile_winter = RES_profiles.iloc[4].loc[[f"{i}" for i in temps]].to_numpy()
offshore_profile_summer = RES_profiles.iloc[1].loc[[f"{i}" for i in temps]].to_numpy()
onshore_profile_summer = RES_profiles.iloc[3].loc[[f"{i}" for i in temps]].to_numpy()
solar_profile_summer = RES_profiles.iloc[5].loc[[f"{i}" for i in temps]].to_numpy()

# Energy mix: capacity installation plans for 2030 (DEA)
offshore_capacity = 4900
onshore_capacity = 4800
solar_capacity = 5265
bioenergy_capacity = 557

# Compute RES hourly production = RES cf * cap
RES_winter = offshore_profile_winter*offshore_capacity + onshore_profile_winter*onshore_capacity + solar_profile_winter*solar_capacity + bioenergy_capacity*1
RES_summer = offshore_profile_summer*offshore_capacity + onshore_profile_summer*onshore_capacity + solar_profile_summer*solar_capacity + bioenergy_capacity*1
# As a comparison with previous data:
# data.generator_availability['W3'][t] = total RES capacity factors
# * max_dem = maximal demand scaling cf => here corresponding to total RES capacity installed
# * RES_factor = scaling factor to play with different RES production scenarios => here represented by scenarios

# Choose data corresponding to the chosen scenario
if season == "Winter":
    RES = RES_winter
else:
    RES = RES_summer

# Compute hourly residual demand (<0 if residual production)
Residual = -RES + Demand_volume_total

# Plotting
# == RES scenarios ==
plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
plt.plot(temps, RES_winter, label="RES in winter")
plt.plot(temps, RES_summer, label="RES in summer")
plt.xlabel("Hour (h)")
plt.ylabel("Power (MW)")
plt.title("Renewable Hourly Production Scenarios (Winter vs Summer)")
plt.legend(loc="upper right")
plt.grid()

plt.subplot(2,2,2)


plt.subplot(2,2,3)
plt.bar(x=temps, height=bioenergy_capacity, color='gray', align='edge', label="Bioenergy")
plt.bar(x=temps, height=offshore_profile_winter*offshore_capacity, bottom=bioenergy_capacity, color='darkblue', align='edge', label="Offshore wind")
plt.bar(x=temps, height=onshore_profile_winter*onshore_capacity, bottom=bioenergy_capacity+offshore_profile_winter*offshore_capacity, color='lightskyblue', align='edge', label="Onshore wind")
plt.bar(x=temps, height=solar_profile_winter*solar_capacity, bottom=bioenergy_capacity+offshore_profile_winter*offshore_capacity+onshore_profile_winter*onshore_capacity, color='orange', align='edge', label="Solar")
plt.xlabel("Hour (h)")
plt.ylabel("Power (MW)")
plt.title("Renewable Hourly Production Mix in Winter")
plt.legend(loc="upper right")

plt.subplot(2,2,4)
plt.bar(x=temps, height=bioenergy_capacity, color='gray', align='edge', label="Bioenergy")
plt.bar(x=temps, height=offshore_profile_summer*offshore_capacity, bottom=bioenergy_capacity, color='darkblue', align='edge', label="Offshore wind")
plt.bar(x=temps, height=onshore_profile_summer*onshore_capacity, bottom=bioenergy_capacity+offshore_profile_summer*offshore_capacity, color='lightskyblue', align='edge', label="Onshore wind")
plt.bar(x=temps, height=solar_profile_summer*solar_capacity, bottom=bioenergy_capacity+offshore_profile_summer*offshore_capacity+onshore_profile_summer*onshore_capacity, color='orange', align='edge', label="Solar")
plt.xlabel("Hour (h)")
plt.ylabel("Power (MW)")
plt.title("Renewable Hourly Production Mix in Summer")
plt.legend(loc="upper right")

plt.tight_layout()

# == Model characteristics ==
plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
for t in temps:
    plt.step(
        pd.concat([pd.Series([0], index=["BEG"]), Demand_volume.iloc[:,t], pd.Series([Demand_volume.iloc[-1,t]], index=["END"])]), 
        pd.concat([pd.Series([Demand_price.iloc[0,t]], index=["BEG"]), Demand_price.iloc[:,t], pd.Series([0], index=["END"])]), 
        label=f"Hour {t+1}")
plt.xlabel("Volume (MWh)")
plt.ylabel("Price (€/MWh)")
plt.title("Price Demand Curve")
plt.grid()
# plt.legend()

plt.subplot(2,2,2)
plt.plot(Demand_volume_total, color="red", marker='.')
plt.bar(x=temps, height=Demand_volume_total, color='red', alpha=0.5, align='edge')
plt.ylim(bottom=0)
plt.xlabel("Hour (h)")
plt.ylabel("Cumulated demand (MWh)")
plt.title("Demand Over Time")
plt.grid()

plt.subplot(2,2,3)
plt.plot(RES, color="green", marker='.')
plt.bar(x=temps, height=RES, color='green', alpha=0.5, align='edge')
plt.plot(Demand_volume_total, color='red', linestyle='--', linewidth=1, label="Total Demand")
plt.xlabel("Hour (h)")
plt.ylabel("Power (MW)")
plt.title("Renewable Production Over Time")
plt.legend()
plt.grid()

plt.subplot(2,2,4)
plt.plot(Residual, color="red", marker='.')
plt.bar(x=temps, height=Residual, color='red', alpha=0.5, align='edge')
plt.xlabel("Time (h)")
plt.ylabel("Power (MW)")
plt.title("Residual Demand Over Time")
plt.grid()

plt.tight_layout()
# plt.savefig("energy-data.png")


# Battery/Storage parameters
Q_max_all = np.zeros(n_players)         # Maximum available power (MW)
Q_all = [[] for _ in range(n_players)]  # List of possible power bids
OC_all = np.zeros(n_players)            # Marginal cost (€/MW)
E_max_all = np.zeros(n_players)         # Maximum battery level (MWh)
Eta_all = np.zeros(n_players)           # Storage round-trip efficiency

# Offset with zero value to effectively compute battery requirements 
# (if first residual demand value x is positive, Local_cumul=0 though we would need x MWh to satisfy the demand - it is due to the min being updated over time but not initialized)
Residual = np.insert(Residual, 0, 0, 0)

# Storage requirement computation
Residual_corrected = np.where(Residual > 0, Residual / min_eta, Residual)   # Taking into account the round-trip efficiency when the battery discharge on the grid (=> need to discharge 100% + eta% energy to satisfy the corresponding demand)

Cummul_res_corr = np.cumsum(Residual_corrected)

Local_cumul = Cummul_res_corr - np.minimum.accumulate(Cummul_res_corr)
                                # minimum of sliced list[:t]

# Minimum total amount of energy the battery must store to satisfy the demand at any time step > informs E_max
Capacity_req = np.max(Local_cumul)                      
Capacity_req = int(np.ceil(Capacity_req))

# PowerRating_req = np.max(np.abs(Residual_corrected))    # Minimum instantaneous power the storage system need to have to satisfy the demand at any time step > informs Q_max
# PowerRating_req = int(np.round(PowerRating_req/10)*10)
PowerRating_available = Capacity_req * storage_Crate_default


# Plotting
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
x = range(T+1)

# Plot 1: Residual and Residual Corrected
axs[0].bar(x, height=Residual_corrected, color='tab:orange', label='Residual Corrected (battery inefficiency)', align='edge')
axs[0].bar(x, height=Residual, color='tab:blue', label='Residual (Demand - RES)', align='edge')
axs[0].axhline(PowerRating_available, color='tab:red', linestyle='--', label='[max] Available Storage Power (MW)')
axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axs[0].set_title('Residual Demand vs Corrected Residual')
axs[0].set_ylabel('Power [MW]')
axs[0].legend()
axs[0].grid(True)

# Plot 2: Cumulative and Local Cumulative
axs[1].plot(Cummul_res_corr, label='Cumulative Residual Corrected', color='tab:green', marker='.')
axs[1].plot(Local_cumul, label='Local Cumulative (Storage Level)', color='tab:red', marker='.')
axs[1].axhline(Capacity_req, color='tab:orange', linestyle='--', label='[max] Required Energy Capacity (MWh)')
axs[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axs[1].set_title('Cumulative Imbalance and Virtual Storage Level')
axs[1].set_xlabel('Hour')
axs[1].set_ylabel('Energy [MWh]')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
# plt.savefig("storage-data.png")


# Final initialization
M = [max(Demand_volume.iloc[-1, t], RES[t]) for t in range(T)]
M = [5 * m for m in M]
Demand_price = Demand_price.to_numpy()
Demand_volume = Demand_volume.to_numpy()

diff_table = []     # Store the difference between model outputs for each iteration



# Function containing the optimisation model
def model_run(q_ch_assumed, q_dis_assumed, player, state_ini=([],[])):

    # Initialization of model parameters
    # Data: RES[t], Demand_price[j,t], Demand_volume[j,t] - t in temps, j in range(D)
    forecasted_production = np.array([RES[t] + q_dis_assumed[t] - q_ch_assumed[t] for t in temps])

    if any(forecasted_production < 0):
        raise "Negative production: charging when no RES available"

    # Adjust demand curve as a price quota curve (based on residual demand) characterized by residual_demand_volume[j,t], residual_demand_price[j,t]
    residual_demand_volume = np.array([[Demand_volume[j,t] - forecasted_production[t] for t in temps] for j in range(D)])
    residual_demand_volume = np.where(residual_demand_volume < 0, 0, residual_demand_volume)

    residual_demand_price = np.zeros((D,T))
    for j in range(D):
        for t in temps:
            if residual_demand_volume[j,t] > 0:
                residual_demand_price[j,t] = Demand_price[j,t]
            else:
                residual_demand_price[j,t] = 0


    # Initialize price quota curve parameters
    step_min_demand = np.insert(residual_demand_volume[:-1,:], 0, 0, axis=0)
    step_max_additional_demand = np.array([[residual_demand_volume[j,t] - step_min_demand[j,t] for t in temps] for j in range(D)])

    # Used to constrain charging variables
    residual_production = np.array([forecasted_production[t] - Demand_volume[-1,t] for t in temps])
    residual_production = np.where(residual_production < 0, 0, residual_production)


    # Initiate optimization process
    model = gp.Model()
    model.Params.OutputFlag = 0
    # model.Params.NonConvex = 2  # Allow for quadratic constraints

    z_ch = model.addVars(temps, range(N), vtype=GRB.BINARY, name="z_ch")
    z_dis = model.addVars(temps, range(N), vtype=GRB.BINARY, name="z_dis")
    e = model.addVars(temps, lb=0, ub=E_max_all[player], name="e")
    u = model.addVars(temps, range(D), vtype=GRB.BINARY, name="u")
    b = model.addVars(temps, range(D), lb=0, name="b")
    q = model.addVars(temps, lb=0, ub=Q_max_all[player], name="q")

    # Define expression of variables
    q_ch = {t: gp.quicksum(Q_all[player][i] * z_ch[t, i] for i in range(N)) for t in temps}
    q_dis = {t: gp.quicksum(Q_all[player][i] * z_dis[t, i] for i in range(N)) for t in temps}
    revenue = {
        t:  gp.quicksum(residual_demand_price[j,t] * (b[t,j] + u[t,j] * step_min_demand[j,t]) for j in range(D)) - 
        OC_all[player] * (q_dis[t] + q_ch[t])
        for t in temps
    }

    # Linear objective function
    model.setObjective(gp.quicksum(revenue[t] for t in temps), GRB.MAXIMIZE)

    ## Storage feasible operating region / technical constraints
    # Energy storage intertemporal constraints: SOC update
    model.addConstr(e[0] == E_max_all[player] * alpha_batt + Eta_all[player] * q_ch[0] - q_dis[0])
    model.addConstrs(e[t] == e[t-1] + Eta_all[player] * q_ch[t] - q_dis[t] for t in range(1, T))
    model.addConstr(e[T-1] >= E_max_all[player] * alpha_batt)

    # Only one storage action possible: idle, charge or discharge at a fixed power rate
    model.addConstrs(gp.quicksum(z_ch[t, i] + z_dis[t, i] for i in range(N)) <= 1 for t in temps)

    ## Identification of the market price > linearization
    # Only a single active price level (corresponding to the market clearing price)
    model.addConstrs(gp.quicksum(u[t, j] for j in range(D)) == 1 for t in temps)

    # Identify amount of demand volume satisfied (for last demand step)
    model.addConstrs(b[t,j] <= u[t,j] * step_max_additional_demand[j,t]
                     for t in temps for j in range(D))
    
    # Constrain the discharging power to be equal to the satisfied demand ("balance equation")
    model.addConstrs(q[t] == q_dis[t] for t in temps)
    model.addConstrs(q[t] == gp.quicksum(b[t,j] + u[t,j] * step_min_demand[j,t] for j in range(D)) for t in temps)

    # In the case of residual production:
    # Market price = 0 because of data characteristics 
    # Charging is possible from residual RES
    model.addConstrs(q_ch[t] <= residual_production[t] for t in temps)
    
    if isinstance(state_ini[0], np.ndarray):
        # Force convergence
        model.addConstrs(z_ch[t,i] == state_ini[0][t,i] for t in temps for i in range(N))
        model.addConstrs(z_dis[t,i] == state_ini[1][t,i] for t in temps for i in range(N))


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
    state = [[z_ch[t, i].X for i in range(N)] for t in temps], \
            [[z_dis[t, i].X for i in range(N)] for t in temps]
    
    u = [[u[t, j].X for j in range(D)] for t in temps]

    y = [[1 - sum(u[t][k] for k in range(j+1)) for j in range(D)] for t in temps]

    CS = [sum((Demand_price[j, t] - Demand_price[j+1, t]) * Demand_volume[j, t] * y[t][j]
                for j in range(D-1)) for t in temps]    # not necessary to include computation for last satisfied load bc it sets the price hence does not increase CS

    price = [sum(residual_demand_price[j,t] * u[t][j] for j in range(D)) for t in temps]

    output = [[q_ch[t].getValue() for t in temps],
              [q_dis[t].getValue() for t in temps],
              [e[t].X for t in temps],
              price,
              [revenue[t].getValue() for t in temps],
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

    # Create summary dictionary for storage characteristics
    summary_data = {
        "Player": [],
        "OC": [],
        "Eta": [],
        "E_max": [],
        "Q_max": [],
        "Q_all": [],
    }

    iter = 0
    for player in range(n_players):
        OC_all[player] = OC_default
        Eta_all[player] = min_eta
        E_max_all[player] = int(np.floor(Capacity_req * size_stor[player] / 10) * 10)
        Q_max_all[player] = int(np.floor(PowerRating_available * size_stor[player]))
        Q_all[player] = [round(Q_max_all[player] * (i / N),2) for i in range(1, N+1)]

        # Fill the summary dictionnary
        summary_data["Player"].append(chr(65 + player))
        summary_data["OC"].append(OC_all[player])
        summary_data["Eta"].append(Eta_all[player])
        summary_data["Q_max"].append(Q_max_all[player])
        summary_data["Q_all"].append(Q_all[player])
        summary_data["E_max"].append(E_max_all[player])
            
        # Initialize optimization model
        state[player], output[player], u[player] = model_run(q_ch_assumed_ini, q_dis_assumed_ini, player)

        # Store profits for later plots
        profits[player].append(sum(output[player][4][t] for t in temps))

    # DataFrame of storage characteristics
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index("Player")
    print(summary_df)

    # Store outputs of initialization models
    state_sys = [state[player] for player in range(n_players)]
    ne.append(state_sys.copy())
    iter += 1

    if n_players == 1:
        return output, ne, iter, u, profits

    while not arrays_are_equal(state_sys, ne[-2], n_players, tol) and iter < max_iter:

        # Profit maximisation for each player
        for player in range(n_players):
            q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in temps]
            q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in temps]
            state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player)
            
            # Store profits for later plots
            profits[player].append(sum(output[player][4][t] for t in temps))

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
                    q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in temps]
                    q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in temps]
                    state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player, 
                                                                         state_ini=(np.array(state[player][0]), np.array(state[player][1])))  
                    profits[player].append(sum(output[player][4][t] for t in temps))      

                for player in range(p+1, n_players):
                    q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in temps]
                    q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in temps]
                    state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player)
                    profits[player].append(sum(output[player][4][t] for t in temps))

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


## -- Setting values to initialize the run --
q_ch_assumed_ini = [0 for _ in temps]
q_dis_assumed_ini = [0 for _ in temps]

output, ne, iter, u, profits = nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, n_players, tol)



## --- Export results ---

# 1. Proad = Discharge - Charge for each player and time
proad = [
    [output[player][1][t] - output[player][0][t] for t in temps]
    for player in range(n_players)
]

# 2. Battery storage level per player
batt = [
    [E_max_all[player] * alpha_batt] + [output[player][2][t] for t in temps]
    for player in range(n_players)
]

# 3. Market price over time (assumed same for all players)
market_price = [output[0][3][t] for t in temps]

# 4. Revenue per player and time
revenue = [
    [output[player][4][t] for t in temps]
    for player in range(n_players)
]

# 5. Total profit per player
profit_tot = [sum(revenue[player]) for player in range(n_players)]
profit_tot_by_cap = [profit_tot[p]/E_max_all[p] if E_max_all[p]!=0 else 0 for p in range(n_players)]

# 6. Total quantity offered to the market
supply_total = [sum(proad[player][t] for player in range(n_players) if proad[player][t] >= 0) for t in temps]   # positive for supply
demand_total = [sum(proad[player][t] for player in range(n_players) if proad[player][t] < 0) for t in temps]    # negative for demand
proad_total = [supply_total[t] + demand_total[t] for t in temps]
q_total = [RES[t] + proad_total[t] for t in temps]

# 7. Unmet demand
unmet_demand = sum(max(Demand_volume[-1, t] - q_total[t], 0) for t in temps)

# 8. Curtailed production
curtailed_prod = sum(max(-Demand_volume[-1, t] + q_total[t], 0) for t in temps)

# 9. Consumer Surplus
for p in range(1,n_players):
    if output[p][5][t] != output[0][5][t]:
        raise "Error in convergence"
    
CS = [output[0][5][t] for t in temps]    # Now we can assume each player outputs the same CS

# 10. Producer Surplus
PS = [
    sum(revenue[player][t] for player in range(n_players)) + 
    RES[t] * market_price[t]
    for t in temps
]

# 11. Social Welfare
SW = sum(CS) + sum(PS)


## --- Plots ---
plt.figure(figsize=(15,8))
temps_np = np.array(temps)
temps_with_zero_np = np.array([t for t in temps] + [T])

# 1. Market Price Plot
plt.subplot(2,2,1)

values_to_show = [p for p in market_price if p > 0]

for player in range(n_players):
    plt.step(temps_np, output[player][3], where='post')
for p in values_to_show:
    plt.axhline(y=p, linestyle='--', color='gray', linewidth=1)
    plt.text(x=temps_np[-1]+1.5, y=p + 0.5, s=f'y = {p}', color='black', ha='left', va='bottom')
plt.xlabel("Time (h)")
plt.ylabel("Market Price (€/MWh)")
plt.title("Market Price Over Time")
plt.grid(True)


# 2. Market Clearing View
plt.subplot(2,2,2)

plt.step(temps_with_zero_np, np.append(Demand_volume[-1, :], Demand_volume[-1, -1]), label="Demand", where='post', color='red', linestyle='--') 
plt.bar(temps_np+0.5, RES, label="RES Production", color='green')
plt.bar(temps_np+0.5, supply_total, label="Total Supply from Players", color='blue', bottom=RES)
plt.bar(temps_np+0.5, demand_total, label="Total Demand from Players", color='blue', alpha=0.7, bottom=0)
plt.xlabel("Time (h)")
plt.ylabel("Power (MW)")
bottom, top = plt.ylim()
plt.ylim(top=top+20)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.2))
plt.title("Market Clearing: Supply vs Demand Over Time")


# 3. Summary Bars for Unmet Demand, Curtailment and Market Metrics
ax1 = plt.subplot(2,2,3)

# --- Ax1: Energy metrics ---
ax1_labels = ["Unmet Demand", "Curtailed Production"]
ax1_heights = [unmet_demand, curtailed_prod]
x1 = np.arange(len(ax1_labels))

bars1 = ax1.bar(x1, ax1_heights, width=0.5, color=['tab:red', 'tab:green'], label="Energy Metrics")
ax1.bar_label(bars1, [f"{round(x)} MWh" for x in ax1_heights])
ax1.set_ylabel("Energy (MWh)")
ax1.set_ylim(0, max(ax1_heights) * 1.2)

# --- Ax2: Economic metrics ---
ax2_labels = ["Consumer Surplus", "Producer Surplus", "Social Welfare"]
ax2_heights = [sum(CS), sum(PS), SW]                # Can be improved by stacking each CS[t], PS[t]
x2 = np.arange(len(ax2_labels)) + len(x1) + 0.5     # offset to avoid overlap

ax2 = ax1.twinx()
bars2 = ax2.bar(x2, ax2_heights, width=0.5, color='tab:purple', label="Welfare Metrics")
ax2.bar_label(bars2, [f"{round(x)} €" for x in ax2_heights])
ax2.set_ylabel("Monetary Value (€)")
ax2.set_ylim(0, max(ax2_heights) * 1.2)

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
ax1.set_ylabel("Total profit")

ax2 = ax1.twinx()
container = ax2.bar(x=x+width/2, height=profit_tot_by_cap, width=width, label=player_labels, color="tab:orange")
ax2.bar_label(container, [f"{round(p)} €/MWh" for p in profit_tot_by_cap])
ax2.set_ylim(top=1.2*max(profit_tot_by_cap))
ax2.set_ylabel("Profit by Installed Capacity Unit")

ax1.set_title("Player Optimal Profits over the Period")

plt.tight_layout()


# 5. Production and SoC per Player
# Ax 1 for energy storage levels, ax 2 for energy storage discharging/charging power
fig, ax1 = plt.subplots(figsize=(15,8))

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
# fig.text(0.5, 0.01, "Player Production = Discharge - Charge Over Time", ha="center")


# 6. Nash Equilibrium Result
plt.figure(figsize=(15,8))
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


# Adjust layout and show the plot
plt.show()


