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

# Downscaling factor for demand curve
scale = 10.06

# Load demand curves (or generate random if missing)
demand_price_winter = pd.DataFrame(np.sort(np.random.rand(10, 24))[::-1])
demand_volume_winter = pd.DataFrame(np.random.rand(10, 24)) / scale
demand_price_summer = pd.DataFrame(np.sort(np.random.rand(10, 24))[::-1])
demand_volume_summer = pd.DataFrame(np.random.rand(10, 24)) / scale

D_winter = demand_price_winter.shape[0]
D_summer = demand_price_summer.shape[0]

max_dem_winter = demand_volume_winter.iloc[D_winter-1, :].values
min_dem_winter = demand_volume_winter.iloc[0, :].values
max_dem_summer = demand_volume_summer.iloc[D_summer-1, :].values
min_dem_summer = demand_volume_summer.iloc[0, :].values

# Load RES profiles (or use random)
RES_summer = np.random.rand(T)
RES_winter = np.random.rand(T)

# Battery/Storage parameters
alpha_batt = 0.5
tolerance_end = 0.05
N = 20

n_players = 2
Q_max_all = np.zeros(n_players)
OC_all = np.zeros(n_players)
E_max_all = np.zeros(n_players)
Q_all = [[] for _ in range(n_players)]
Eta_all = np.zeros(n_players)

# Storage requirement computation
min_eta = 0.85

# Summer storage needs
Residual_summer = -RES_summer + demand_volume_summer.iloc[D_summer-1, :].values
Residual_corrected_summer = np.where(Residual_summer > 0, Residual_summer / min_eta, Residual_summer)
Cummul_res_corr_summer = np.cumsum(Residual_corrected_summer)
Local_cumul_summer = Cummul_res_corr_summer - np.minimum.accumulate(Cummul_res_corr_summer)
Stor_req_summer = np.max(Local_cumul_summer)

# Winter storage needs
Residual_winter = -RES_winter + max_dem_winter
Residual_corrected_winter = np.where(Residual_winter > 0, Residual_winter / min_eta, Residual_winter)
Cummul_res_corr_winter = np.cumsum(Residual_corrected_winter)
Local_cumul_winter = Cummul_res_corr_winter - np.minimum.accumulate(Cummul_res_corr_winter)
Stor_req_winter = np.max(Local_cumul_winter)

Stor_req = int(np.floor(max(Stor_req_summer, Stor_req_winter)))

Capa_req_summer = np.max(Residual_corrected_summer)
Capa_req_winter = np.max(Residual_corrected_winter)
Capa_req = int(np.floor(max(Capa_req_summer, Capa_req_winter)))

# Plot
plt.figure()
plt.plot(Cummul_res_corr_winter)
plt.xlabel("Time (h)")
plt.ylabel("Cumulated residual corrected demand (MWh)")
plt.savefig("cr_plot.png")
plt.close()

# Fixing to winter season by default
RES = RES_winter
Demand_price = demand_price_winter
Demand_volume = demand_volume_winter
D = D_winter
max_dem = max_dem_winter
min_dem = min_dem_winter

# Load extreme production days (or use random)
extreme_winter = np.random.rand(24)
extreme_summer = np.random.rand(24)

# Season dictionary
season_dict = {
    "winter": {
        "RES": RES_winter,
        "Demand_price": demand_price_winter,
        "Demand_volume": demand_volume_winter,
        "D": D_winter,
        "max_dem": max_dem_winter,
        "min_dem": min_dem_winter,
        "M": [max(demand_volume_winter.iloc[D_winter-1, t], RES_winter[t]) for t in range(T)]
    },
    "summer": {
        "RES": RES_summer,
        "Demand_price": demand_price_summer,
        "Demand_volume": demand_volume_summer,
        "D": D_summer,
        "max_dem": max_dem_summer,
        "min_dem": min_dem_summer,
        "M": [max(demand_volume_summer.iloc[D_summer-1, t], RES_summer[t]) for t in range(T)]
    }
}


# Function containing the optimisation model
def model_run(start_state, q_ch_assumed, q_dis_assumed, player, season):

    season_vars = season_dict[season]
    RES = season_vars["RES"]
    Demand_price = season_vars["Demand_price"]
    Demand_volume = season_vars["Demand_volume"]
    D = season_vars["D"]
    max_dem = season_vars["max_dem"]
    min_dem = season_vars["min_dem"]
    M = season_vars["M"]

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

    model.addConstr(e[1] == E_max_all[player] * alpha_batt + Eta_all[player] * q_ch[1] - q_dis[1])
    model.addConstrs(e[t] == e[t-1] + Eta_all[player] * q_ch[t] - q_dis[t] for t in range(2, T+1))
    model.addConstr(e[T] <= E_max_all[player] * alpha_batt * (1 + tolerance_end))
    model.addConstr(e[T] >= E_max_all[player] * alpha_batt * (1 - tolerance_end))

    model.addConstrs(q_tot[t] >= Demand_volume[j, t] - M[t] * (1 - u[t, j]) for t in temps for j in range(1, D-1))
    model.addConstrs(q_tot[t] <= Demand_volume[j+1, t] + M[t] * (1 - u[t, j]) for t in temps for j in range(1, D-1))

    model.addConstrs(q_tot[t] >= Demand_volume[j+1, t] - M[t] * (1 - u_plus[t, j]) for t in temps for j in range(1, D-1))
    model.addConstrs(q_tot[t] <= Demand_volume[j, t] + M[t] * (1 - u_moins[t, j]) for t in temps for j in range(1, D-1))
    model.addConstrs(u[t, j] + u_plus[t, j] + u_moins[t, j] == 1 for t in temps for j in range(1, D-1))

    model.addConstrs(q_tot[t] <= Demand_volume[D-1, t] + M[t] * u[t, D-1] for t in temps)
    model.addConstrs(q_tot[t] >= Demand_volume[D-1, t] - M[t] * (1 - u[t, D-1]) for t in temps)

    model.addConstrs(q_tot[t] >= Demand_volume[1, t] - M[t] * u[t, 0] for t in temps)
    model.addConstrs(q_tot[t] <= Demand_volume[1, t] + M[t] * (1 - u[t, 0]) for t in temps)

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

    for player in range(1, n_players + 1):
        # OC_all[player] = 1 - (player // 2) * 0.15
        OC_all[player] = 0.5
        Eta_all[player] = 0.85
        # Eta_all[p] = 0.85 - 0.01 * ((p-1)//2)
        Q_max_all[player] = Capa_req * size_stor[player - 1]
        Q_all[player] = [Q_max_all[player] * (i / N) for i in range(1, N+1)]
        E_max_all[player] = Stor_req * size_stor[player - 1]
        state[player], output[player], u[player] = model_run(state_ini, q_ch_assumed_ini, q_dis_assumed_ini, player, season)

    state_sys = [state[player][:2] for player in range(n_players)]

    if n_players == 1:
        return output, ne, 1

    iter = 0
    tol = 1e-6
    while iter < 50 and not any(arrays_are_equal(state_sys, ne_state, n_players, tol) for ne_state in ne.values()):
        ne[iter + 1] = state_sys.copy()

        for player in range(n_players):
            others = [p for p in range(n_players) if p != player]
            q_ch_assumed = sum(output[other][0] for other in others)
            q_dis_assumed = sum(output[other][1] for other in others)
            state[player], output[player], u[player] = model_run(state[player], q_ch_assumed, q_dis_assumed, player + 1, season)

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


