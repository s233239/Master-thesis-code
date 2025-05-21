# Import relevant packages
import gurobipy as gp                 # Gurobi Python API
from gurobipy import GRB              # Gurobi constants (e.g., GRB.MAXIMIZE)

import pandas as pd                   # DataFrames
import matplotlib.pyplot as plt       # Plotting
import numpy as np                    # Numerical operations (similar to Julia base)

from joblib import Parallel, delayed  # For parallel computing (optional alternative: multiprocessing)

import data

## --- Initialization of the problem ---

# Set time horizon parameters
T = 24   # number of time periods
temps = range(T)  # time periods

# Load demand curves
D = len(data.LOADS)

# Demand_price = pd.DataFrame({d: [round(data.load_bids[d],2)] * T for d in data.LOADS}).T
Demand_price_array = np.sort(np.random.rand(D))[::-1] * data.load_cost
Demand_price = pd.DataFrame({d: [round(Demand_price_array[d],2)] * T for d in range(D)}).T
Demand_price.columns = range(T)
Demand_price.index = data.LOADS

Demand_volume = pd.DataFrame({d: [round(data.load_profile[d,t],2) for t in data.TIMES] for d in data.LOADS}).T
Demand_volume_cumulative = Demand_volume.cumsum(axis=0)
Demand_volume = Demand_volume_cumulative

Demand_volume_total = Demand_volume.iloc[D-1, :].values

max_dem = data.load_capacity
min_dem = 0

# Load RES profile
RES = np.array([data.generator_availability['W3'][t] * 1.3 * max_dem for t in range(1,T+1)])
Residual = -RES + Demand_volume_total

# Plotting
plt.figure(figsize=(15,8))
plt.subplot(2,2,1)
for t in temps:
    plt.step(
        pd.concat([pd.Series([0], index=["BEG"]), Demand_volume.iloc[:,t], pd.Series([Demand_volume.iloc[-1,t]], index=["END"])]), 
        pd.concat([pd.Series([Demand_price.iloc[0,t]], index=["BEG"]), Demand_price.iloc[:,t], pd.Series([0], index=["END"])]), 
        label=f"Hour {t+1}")
plt.xlabel("Volume (MWh)")
plt.ylabel("Price (€/MWh)")
plt.title("Load Demand Curve")
# plt.legend()

plt.subplot(2,2,2)
plt.plot(Demand_volume_total, color="red")
plt.xlabel("Hour (h)")
plt.ylabel("Cumulated demand (MWh)")
plt.title("Demand Over Time")

plt.subplot(2,2,3)
plt.plot(RES, color="green")
plt.plot(Demand_volume_total, color='red', linestyle='--', linewidth=0.8, label="Total Demand")
plt.xlabel("Hour (h)")
plt.ylabel("Power (MW)")
plt.title("Renewable Production Over Time")
plt.legend()

plt.subplot(2,2,4)
plt.plot(Residual, color="red")
plt.xlabel("Time (h)")
plt.ylabel("Cumulated residual demand (MWh)")
plt.title("Residual Corrected Demand Over Time")
plt.tight_layout()

# Battery/Storage parameters
alpha_batt = 0.5
tolerance_end = 0.05
N = 20                  # Discretization number for power outputs

n_players = 2
Q_max_all = np.zeros(n_players)
OC_all = np.zeros(n_players)
E_max_all = np.zeros(n_players)
Q_all = [[] for _ in range(n_players)]
Eta_all = np.zeros(n_players)

# Storage requirement computation
min_eta = 0.85

# Storage needs
Residual_corrected = np.where(Residual > 0, Residual / min_eta, Residual)   # If residual demand, then battery will discharge -> 1/eta % more energy needed to satisfy the demand
Cummul_res_corr = np.cumsum(Residual_corrected)
# ..... if always surplus deficit, then offset by initial value to get full battery need
Local_cumul = Cummul_res_corr - np.minimum.accumulate(Cummul_res_corr)

Stor_req = np.max(Local_cumul)          
Stor_req = int(np.floor(Stor_req))

Capa_req = np.max(Residual_corrected)   # Max power needed over each hour to satisfy the demand > informs total Q_max that needs to be higher
Capa_req = int(np.floor(Capa_req))

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Residual and Residual Corrected
axs[0].plot(Residual, label='Residual (Demand - RES)', color='tab:blue')
axs[0].plot(Residual_corrected, label='Residual Corrected (battery inefficiency)', color='tab:orange')
axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
axs[0].set_title('Residual Demand vs Corrected Residual')
axs[0].set_ylabel('Power [MW]')
axs[0].legend()
axs[0].grid(True)

# Plot 2: Cumulative and Local Cumulative
axs[1].plot(Cummul_res_corr, label='Cumulative Residual Corrected', color='tab:green')
axs[1].plot(Local_cumul, label='Local Cumulative (Storage Level)', color='tab:red')
axs[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
axs[1].set_title('Cumulative Imbalance and Virtual Storage Level')
axs[1].set_xlabel('Hour')
axs[1].set_ylabel('Energy [MWh]')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()


# Final initialization
M = [max(Demand_volume.iloc[D-1, t], RES[t]) for t in range(T)]
Demand_price = Demand_price.to_numpy()
Demand_volume = Demand_volume.to_numpy()


# Function containing the optimisation model
def model_run(start_state, q_ch_assumed, q_dis_assumed, player, season):

    model = gp.Model()
    model.setParam("Threads", 8)
    model.setParam("Method", 2)

    z_ch = model.addVars(temps, range(N), vtype=GRB.BINARY, name="z_ch")
    z_dis = model.addVars(temps, range(N), vtype=GRB.BINARY, name="z_dis")
    e = model.addVars(temps, lb=0, ub=E_max_all[player], name="e")
    u = model.addVars(temps, range(D), vtype=GRB.BINARY, name="u")
    u_plus = model.addVars(temps, range(1, D-1), vtype=GRB.BINARY, name="u_plus")
    u_moins = model.addVars(temps, range(1, D-1), vtype=GRB.BINARY, name="u_moins")
    w_ch = model.addVars(temps, range(N), range(D), vtype=GRB.BINARY, name="w_ch")
    w_dis = model.addVars(temps, range(N), range(D), vtype=GRB.BINARY, name="w_dis")

    q_ch = {t: gp.quicksum(Q_all[player][i] * z_ch[t, i] for i in range(N)) for t in temps}
    q_dis = {t: gp.quicksum(Q_all[player][i] * z_dis[t, i] for i in range(N)) for t in temps}
    price = {t: gp.quicksum(u[t, j] * Demand_price[j, t] for j in range(D)) for t in temps}
    revenue = {
        t: gp.quicksum(
            gp.quicksum((w_dis[t, i, j] - w_ch[t, i, j]) * Demand_price[j, t] * Q_all[player][i] for j in range(D)) -
            OC_all[player] * Q_all[player][i] * (z_dis[t, i] + z_ch[t, i]) for i in range(N)
        )
        for t in temps
    }
    q_tot = {
        t: -q_ch[t] + q_dis[t] - q_ch_assumed[t] + q_dis_assumed[t] + RES[t]
        for t in temps
    }

    model.setObjective(gp.quicksum(revenue[t] for t in temps), GRB.MAXIMIZE)

    model.addConstrs(w_ch[t, i, j] <= u[t, j] for t in temps for i in range(N) for j in range(D))
    model.addConstrs(w_ch[t, i, j] <= z_ch[t, i] for t in temps for i in range(N) for j in range(D))
    model.addConstrs(w_ch[t, i, j] >= u[t, j] + z_ch[t, i] - 1 for t in temps for i in range(N) for j in range(D))
    model.addConstrs(w_dis[t, i, j] <= u[t, j] for t in temps for i in range(N) for j in range(D))
    model.addConstrs(w_dis[t, i, j] <= z_dis[t, i] for t in temps for i in range(N) for j in range(D))
    model.addConstrs(w_dis[t, i, j] >= u[t, j] + z_dis[t, i] - 1 for t in temps for i in range(N) for j in range(D))

    model.addConstrs(gp.quicksum(z_ch[t, i] + z_dis[t, i] for i in range(N)) <= 1 for t in temps)
    model.addConstrs(gp.quicksum(u[t, i] for i in range(D)) == 1 for t in temps)

    model.addConstr(e[0] == E_max_all[player] * alpha_batt + Eta_all[player] * q_ch[0] - q_dis[0])
    model.addConstrs(e[t] == e[t-1] + Eta_all[player] * q_ch[t] - q_dis[t] for t in range(1, T))
    model.addConstr(e[T-1] <= E_max_all[player] * alpha_batt * (1 + tolerance_end))
    model.addConstr(e[T-1] >= E_max_all[player] * alpha_batt * (1 - tolerance_end))

    model.addConstrs(q_tot[t] >= Demand_volume[j, t] - M[t] * (1 - u[t, j]) for t in temps for j in range(1, D-1))
    model.addConstrs(q_tot[t] <= Demand_volume[j+1, t] + M[t] * (1 - u[t, j]) for t in temps for j in range(1, D-1))

    model.addConstrs(q_tot[t] >= Demand_volume[j+1, t] - M[t] * (1 - u_plus[t, j]) for t in temps for j in range(1, D-1))
    model.addConstrs(q_tot[t] <= Demand_volume[j, t] + M[t] * (1 - u_moins[t, j]) for t in temps for j in range(1, D-1))
    model.addConstrs(u[t, j] + u_plus[t, j] + u_moins[t, j] == 1 for t in temps for j in range(1, D-1))

    model.addConstrs(q_tot[t] <= Demand_volume[D-1, t] + M[t] * u[t, D-1] for t in temps)
    model.addConstrs(q_tot[t] >= Demand_volume[D-1, t] - M[t] * (1 - u[t, D-1]) for t in temps)

    model.addConstrs(q_tot[t] >= Demand_volume[1, t] - M[t] * u[t, 0] for t in temps)
    model.addConstrs(q_tot[t] <= Demand_volume[1, t] + M[t] * (1 - u[t, 0]) for t in temps)

    model.Params.OutputFlag = 0
    model.optimize()

    state = [[z_ch[t, i].X for i in range(N)] for t in temps], \
            [[z_dis[t, i].X for i in range(N)] for t in temps], \
            [e[t].X for t in temps]

    y = [[sum(u[t, k].X for k in range(j+1, D)) for j in range(D-1)] for t in temps]

    CS = [sum((Demand_price[j, t] - Demand_price[j+1, t]) * (Demand_volume[j+1, t] - Demand_volume[0, t]) * y[t][j]
              for j in range(D-1)) for t in temps]

    output = [[q_ch[t] for t in temps],
              [q_dis[t] for t in temps],
              [e[t].X for t in temps],
              [price[t].getValue() for t in temps],
              [revenue[t].getValue() for t in temps],
              CS]

    return state, output, u


def arrays_are_equal(a1, a2, n_players, tol=1e-7):
    diff = 0
    for p in range(n_players):
        diff += abs(sum(a1[p][0] - a2[p][0] + a1[p][1] - a2[p][1]))
    return diff < tol


def nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, state_ini, season, n_players):

    ne = {}
    state = {}
    output = {}
    u = {}

    if n_players == 1:
        size_stor = [1]
    elif n_players == 2:
        size_stor = [1/3, 2/3]
    elif n_players == 4:
        size_stor = [0.1, 0.2, 0.2, 0.5]
    elif n_players == 6:
        size_stor = [0.05, 0.1, 0.1, 0.15, 0.25, 0.35]
    elif n_players == 8:
        size_stor = [0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.2, 0.25]

    for player in range(n_players):
        OC_all[player] = 0.5
        Eta_all[player] = min_eta
        Q_max_all[player] = Capa_req * size_stor[player]
        Q_all[player] = [Q_max_all[player] * (i / N) for i in range(1, N+1)]
        E_max_all[player] = Stor_req * size_stor[player]
        state[player], output[player], u[player] = model_run(state_ini, q_ch_assumed_ini, q_dis_assumed_ini, player, season)

    state_sys = [state[player][:2] for player in range(n_players)]

    if n_players == 1:
        return output, ne, 1

    iter = 0
    tol = 1e-6
    while iter < 50 and not any(arrays_are_equal(state_sys, ne_state, n_players, tol) for ne_state in ne.values()):
        ne[iter + 1] = state_sys.copy()
        print(iter)

        for player in range(n_players):
            others = [p for p in range(n_players) if p != player]
            q_ch_assumed = [0]*24
            q_dis_assumed = [0]*24
            for other in others:
                q_ch_assumed = [q_ch_assumed[t] + output[other][0][t] for t in temps]
                q_dis_assumed = [q_dis_assumed[t] + output[other][1][t] for t in temps]
            state[player], output[player], u[player] = model_run(state[player], q_ch_assumed, q_dis_assumed, player, season)

        state_sys = [state[player][:2] for player in range(n_players)]
        iter += 1

    return output, ne, iter, u


## -- Setting values to initialize the run --
q_ch_assumed_ini = [0 for _ in temps]
q_dis_assumed_ini = [0 for _ in temps]
state_ini = [
    [[0 for _ in range(N)] for _ in temps],  # First list: 2D list with dims (len(temps), N)
    [[0 for _ in range(N)] for _ in temps],  # Second list: same dims
    [0 for _ in temps]                       # Third list: 1D list with length len(temps)
]

output = nash_eq(q_ch_assumed_ini,q_dis_assumed_ini,state_ini,"winter",2)

# Flatten the data from output into column vectors
column_vectors = [
    output[0][p][i]
    for p in range(len(output[0]))
    for i in range(len(output[0][0]))
]

P = int(len(column_vectors) / 6)
column_labels = ["Charge", "Discharge", "Battery", "Price", "Revenue", "CS"]
column_names = [f"{label}{i}" for i in range(P) for label in column_labels]

df = pd.DataFrame([column_vectors], columns=column_names)

## --- Export results ---

# 1. Proad = Discharge - Charge for each player and time
proad = [
    [output[0][player][1][t] - output[0][player][0][t] for t in temps]
    for player in range(n_players)
]

# 2. Battery storage level per player
batt = [
    [E_max_all[player] * alpha_batt] + [output[0][player][2][t] for t in temps]
    for player in range(n_players)
]

# 3. Market price over time (assumed same for all players)
market_price = [output[0][0][3][t] for t in temps]

# 4. Revenue per player and time
revenue = [
    [output[0][player][4][t] for t in temps]
    for player in range(n_players)
]

# 5. Total profit per player
profit_tot = [sum(revenue[player]) for player in range(n_players)]

# 6. Total quantity offered to the market
q_total = [RES[t] + sum(proad[player][t] for player in range(n_players)) for t in temps]

# 7. Unmet demand
unmet_demand = sum(max(Demand_volume[D, t] - q_total[t], 0) for t in temps)

# 8. Curtailed production
curtailed_prod = sum(max(-Demand_volume[D, t] + q_total[t], 0) for t in temps)

# 9. Consumer Surplus
CS = [
    output[0][0][5][t] + (Demand_price[0, t] - market_price[t]) * Demand_volume[0, t]
    for t in temps
]

# 10. Producer Surplus
PS = sum(sum(revenue[player]) for player in range(n_players)) + sum(RES[t] * market_price[t] for t in temps)

# 11. Social Welfare
SW = sum(CS) + PS


## --- Plots ---
plt.figure(figsize=(15,8))
temps_np = np.array(temps)
q_total_pos = [q if q >= 0 else 0 for q in q_total]
q_total_neg = [q if q < 0 else 0 for q in q_total]

# 1. Market Price Plot
plt.subplot(2,2,1)

plt.step(temps_np, market_price, where='post', color='black')
plt.xlabel("Time (h)")
plt.ylabel("Market Price (€/MWh)")
plt.title("Market Price Over Time")
plt.grid(True)


# 2. Production and SoC per Player
# Ax 1 for energy storage levels, ax 2 for energy storage discharging/charging power
ax1 = plt.subplot(2,2,3) 

for player in range(n_players):
    ax1.plot(np.append(temps_np, T), batt[player], label=f"SoC for Player {player + 1}")
ax1.set_xlabel("Time (h)")
ax1.set_ylabel("Battery State of Charge (MWh)")
ax1.legend(loc="upper left")
ax1.set_title("Battery Cycle")

ax2 = ax1.twinx()
for player in range(n_players):
    ax2.step(np.append(temps_np, T), proad[player] + proad[player][-1], where="post", label=f"Production from Player {player + 1}", linestyle='--', linewidth=0.8)
ax2.set_ylabel('Power [MW]')
ax2.legend(loc='upper right')
ax2.grid()
ax2.figtext(0.5, 0.01, "Player Production = Discharge - Charge Over Time", ha="center")


# 3. Market Clearing View
plt.subplot(2,2,2)

plt.step(temps_np, Demand_volume[D, :], label="Demand", where='post', color='red', linestyle='--') 
plt.bar(temps_np+0.5, RES, label="RES Production", color='green')
plt.bar(temps_np+0.5, q_total_pos, label="Total Supply from Players", color='blue', bottom=RES)
plt.bar(temps_np+0.5, q_total_neg, label="Total Supply from Players", color='blue', alpha='0.5', bottom=0)

plt.xlabel("Time (h)")
plt.ylabel("Power (MW)")
plt.legend()
plt.title("Market Clearing: Supply vs Demand Over Time")


# 4. Summary Bars for Unmet Demand and Curtailment
plt.subplot(2,2,4)

plt.bar(["Unmet Demand", "Curtailed Production"], [unmet_demand, curtailed_prod], color=["orange", "purple"])
plt.title("Total Unmet Demand and Curtailed Production")
plt.ylabel("Energy (MWh)")


# Adjust layout and show the plot
plt.tight_layout()
plt.show()


