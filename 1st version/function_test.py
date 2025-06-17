## Install relevant packages
import numpy as np
import random
import matplotlib.pyplot as plt

import gurobipy as gb
from gurobipy import GRB

random.seed(101)

## Parameter Initialization
MCa = 0         # marginal cost (€/MWh)
Q_max_a = 40    # maximum power output/input (MW)
E_max_a = 100   # maximum battery level (MWh)

MCb = 0
Q_max_b = 60
E_max_b = 200

# Discrete Bids
N = 10  # number of discrete values the players can bid from
Q_a = [Q_max_a * i / N for i in range(1,N+1)]   # list of available bids for player A
Q_b = [Q_max_b * i / N for i in range(1,N+1)]   # list of available bids for player B

# Battery System Parameters
eta = 0.9               # efficiency of charging
alpha_batt = 0.5        # percentage of initial battery level
tolerance_end = 0.01    # tolerance percentage between the initial and final battery level

# Time periods
T = 24
TIME = range(T)

# Residual Demand and Production
R = [(random.random() - 0.5) * 2 * (Q_max_a + Q_max_b) for _ in TIME]   # residual demand/production
R_ch = [max(0, r) for r in R]   # residual production if r>0
R_dis = [-min(0, r) for r in R] # residual demand if r>0
sign_R = [max(0, r) / r if r != 0 else 0 for r in R]  # 1 if residual production, 0 else

# Price Levels and Intervals
P_C = [45, 35, 25, 5]       # price levels for charge mode
C = len(P_C)                # number of price levels in charge mode
P_D = [70, 60, 40, 20, 10]  # price levels for discharge mode
D = len(P_D)                # number of price levels in discharge mode
level_ch = [i / C for i in range(C+1)]    # intervals for charge mode
level_dis = [i / D for i in range(D+1)]   # intervals for discharge mode
eps = min(Q_a[0], Q_b[0]) / max(R_ch + R_dis) / 2   # small value bigger than 0, chosen lower than the minimum value the players can bid so as not to influence the decision-making
level_ch[0] = eps
level_dis[0] = eps

# Big M Definition
M = max(max(R_ch), max(R_dis), max(Q_a), max(Q_b))

# Cournot iteration
tol = 1e-1          # Tolerance for convergence (MWh)
max_iter = 100      # Maximum number of iterations
state_changes = []  # Max changer for each iteration
max_change = M


## Optimisation model
def model_run(start_state:list, q_ch_assumed:list[int], q_dis_assumed:list[int], player:str):
    """
    The model outputs the decision state of a given player (charge and discharge profiles, battery level through time,
    market price and related binary variables and the profit.
    It requires in input using an initial state to make it easier to find a solution, the charge and discharge profile
    from the other player and the player for which to run the model ("A" or "B" expected).

    Parameters:
        start_state: Initialization list [q_ch, q_dis, e, price, z_ch, z_dis, u_ch, u_dis].
        q_ch_assumed: List of charging power for every time step.
        q_dis_assumed: List of discharging power for every time step.
        player: Player for whom the model is optimized: 'A' or 'B'.

    Returns:
        (state, profit): Output list [q_ch, q_dis, e, price, z_ch, z_dis, u_ch, u_dis], objective value.

    """
    
    # Parameters associated to the player
    Q = Q_a if player == "a" else Q_b
    E_max = E_max_a if player == "a" else E_max_b
    MC = MCa if player == "a" else MCb
    Q_max = Q_max_a if player == "a" else Q_max_b

    # Optimisation model
    model = gb.Model('my model 1')
    model.Params.NonConvex = 2  #Allow for quadratic constraints, used for the bi-level constraints


    # Define variables
    q_ch = model.addVars(T, lb=0, ub=Q_max, vtype=GRB.CONTINUOUS, name="q_ch")
    q_dis = model.addVars(T, lb=0, ub=Q_max, vtype=GRB.CONTINUOUS, name="q_dis")
    e = model.addVars(T, lb=0, ub=E_max, vtype=GRB.CONTINUOUS, name="e")
    price = model.addVars(T, lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY, vtype=GRB.CONTINUOUS, name="price")

    # Binary variables for discretization of q_ch and q_dis
    z_ch = model.addVars(T, N, vtype=GRB.BINARY, name="z_ch")
    z_dis = model.addVars(T, N, vtype=GRB.BINARY, name="z_dis")

    # Binary variables for price level encoding (charge mode)
    u_ch = model.addVars(T, C, vtype=GRB.BINARY, name="u_ch")
    u_plus_ch = model.addVars(T, C, vtype=GRB.BINARY, name="u_plus_ch")
    u_moins_ch = model.addVars(T, C, vtype=GRB.BINARY, name="u_moins_ch")

    # Binary variables for price level encoding (discharge mode)
    u_dis = model.addVars(T, D, vtype=GRB.BINARY, name="u_dis")
    u_plus_dis = model.addVars(T, D, vtype=GRB.BINARY, name="u_plus_dis")
    u_moins_dis = model.addVars(T, D, vtype=GRB.BINARY, name="u_moins_dis")

    # # Start values
    # for t in TIME:
    #     q_ch[t].start = start_state[0][t]
    #     q_dis[t].start = start_state[1][t]
    #     e[t].start = start_state[2][t]
    #     price[t].start = start_state[3][t]
    #     for i in range(N):
    #         z_ch[t, i].start = start_state[4][t,i]
    #         z_dis[t, i].start = start_state[5][t,i]
            

    # Objective function
    obj_expr = gb.quicksum(-price[t] * q_ch[t] + price[t] * q_dis[t] - MC*(q_dis[t] + q_ch[t]) for t in TIME)   # Profit as costs from charging + revenue from discharging
    model.setObjective(obj_expr, GRB.MAXIMIZE)


    # Constraints
    for t in TIME:

        # Discretization of charging/discharging power
        model.addConstr(q_ch[t] == gb.quicksum(Q[i] * z_ch[t, i] for i in range(N)))
        model.addConstr(q_dis[t] == gb.quicksum(Q[i] * z_dis[t, i] for i in range(N)))

        # Discretization of market price
        model.addConstr(
            price[t] == gb.quicksum(u_ch[t, j] * P_C[j] for j in range(C)) +
                        gb.quicksum(u_dis[t, j] * P_D[j] for j in range(D))
        )

        # Only one of charge/discharge power variable is active
        model.addConstr(
            gb.quicksum(z_ch[t, i] for i in range(N)) + gb.quicksum(z_dis[t, i] for i in range(N)) <= 1
        )

        # Only one price level variable active (charge or discharge)
        model.addConstr(
            gb.quicksum(u_ch[t, i] for i in range(C)) + gb.quicksum(u_dis[t, i] for i in range(D)) <= 1
        )

        # Logical constraints for residual production
        model.addConstr(gb.quicksum(u_dis[t, i] for i in range(D)) <= 1 - sign_R[t])
        model.addConstr(gb.quicksum(u_ch[t, i] for i in range(C)) <= sign_R[t])

    # Battery dynamics
    model.addConstr(e[0] == E_max * alpha_batt + eta * q_ch[0] - q_dis[0])
    
    for t in range(1, T):
        model.addConstr(
            e[t] == e[t - 1] + eta * q_ch[t] - q_dis[t]
        )

    model.addConstr(e[T-1] >= E_max * alpha_batt * (1 - tolerance_end)) # Final battery level constraint (optional)

    # Piecewise demand curve for discharge side
    model.addConstrs(
        q_dis[t] + q_dis_assumed[t] >= level_dis[i] * R_dis[t] - M * (1 - u_dis[t, i])
        for t in TIME for i in range(D)
    )
    model.addConstrs(
        q_dis[t] + q_dis_assumed[t] <= level_dis[i + 1] * R_dis[t] + M * (1 - u_dis[t, i])
        for t in TIME for i in range(D)
    )
    model.addConstrs(
        q_dis[t] + q_dis_assumed[t] >= level_dis[i + 1] * R_dis[t] - M * (1 - u_plus_dis[t, i])
        for t in TIME for i in range(D)
    )
    model.addConstrs(
        q_dis[t] + q_dis_assumed[t] <= level_dis[i] * R_dis[t] + M * (1 - u_moins_dis[t, i])
        for t in TIME for i in range(D)
    )
    model.addConstrs(
        u_dis[t, i] + u_plus_dis[t, i] + u_moins_dis[t, i] == 1
        for t in TIME for i in range(D)
    )

    # Piecewise demand curve for charge side
    model.addConstrs(
        q_ch[t] + q_ch_assumed[t] >= level_ch[i] * R_ch[t] - M * (1 - u_ch[t, i])
        for t in TIME for i in range(C)
    )
    model.addConstrs(
        q_ch[t] + q_ch_assumed[t] <= level_ch[i + 1] * R_ch[t] + M * (1 - u_ch[t, i])
        for t in TIME for i in range(C)
    )
    model.addConstrs(
        q_ch[t] + q_ch_assumed[t] >= level_ch[i + 1] * R_ch[t] - M * (1 - u_plus_ch[t, i])
        for t in TIME for i in range(C)
    )
    model.addConstrs(
        q_ch[t] + q_ch_assumed[t] <= level_ch[i] * R_ch[t] + M * (1 - u_moins_ch[t, i])
        for t in TIME for i in range(C)
    )
    model.addConstrs(
        u_ch[t, i] + u_plus_ch[t, i] + u_moins_ch[t, i] == 1
        for t in TIME for i in range(C)
    )

    

    # Solving and returning results
    model.Params.OutputFlag = 0
    model.optimize()

    model.write("myLP_model.lp")

    if model.status != GRB.OPTIMAL:
        print(f"Model status: {model.status}")

        model.computeIIS()        

        model.Params.OutputFlag = 1
        model.Params.LogFile = "gurobi_log.txt"

        return (None, None)
    
    ## Analyse the constraints (inactive/active)
    active_constraints = []
    inactive_constraints = []

    for constr in model.getConstrs():
        sense = constr.Sense  # Sense: '=' for equality, '<' for less-equal, '>' for greater-equal
        
        # Skip equality constraints
        if sense == '=':
            continue

        # Check if constraint is active
        if abs(constr.Slack) < 1e-6:  # Adjust tolerance as needed
            active_constraints.append(constr.ConstrName)
        else:
            inactive_constraints.append(constr.ConstrName)

    # print("Active constraints:", active_constraints)
    # print("Inactive constraints:", inactive_constraints)


    ## Get results
    state = [
        [q_ch[t].X for t in range(T)],
        [q_dis[t].X for t in range(T)],
        [e[t].X for t in range(T)],
        [price[t].X for t in range(T)],
        {(t, i): z_ch[t, i].X for t in range(T) for i in range(N)},
        {(t, i): z_dis[t, i].X for t in range(T) for i in range(N)},
        {(t, j): u_ch[t, j].X for t in range(T) for j in range(C)},
        {(t, j): u_dis[t, j].X for t in range(T) for j in range(D)},
    ]

    profit = model.ObjVal

    return (state, profit)


## Functions for Nash equilibrium iteration

def nash_eq_a(q_ch_assumed_ini, q_dis_assumed_ini, state_ini):
    """
    Finds the Nash equilibrium iteratively, starting from an initial bid by player A.
    """
    # A plays first assuming B bids 0
    state_a, profit_a = model_run(state_ini, q_ch_assumed_ini, q_dis_assumed_ini, 'A')
    q_ch_assumed_a, q_dis_assumed_a, batt_a, price_a = state_a[:4]

    # B plays based on A's previous decision
    state_b, profit_b = model_run(state_ini, q_ch_assumed_a, q_dis_assumed_a, 'B')
    q_ch_assumed_b, q_dis_assumed_b, batt_b, price_b = state_b[:4]

    # Initial (dummy) previous state
    old_state_sys = (
        q_ch_assumed_ini, q_dis_assumed_ini, q_dis_assumed_ini, 0,
        q_dis_assumed_ini, q_dis_assumed_ini, q_dis_assumed_ini, 0,
        q_dis_assumed_ini
    )
    state_sys = (
        q_ch_assumed_a, q_dis_assumed_a, batt_a, profit_a,
        q_ch_assumed_b, q_dis_assumed_b, batt_b, profit_b,
        price_b
    )

    iter_count = 0
    while iter_count < 10 and old_state_sys != state_sys:
        old_state_sys = state_sys  # store current system state

        state_a, profit_a = model_run(state_a, q_ch_assumed_b, q_dis_assumed_b, 'A')
        q_ch_assumed_a, q_dis_assumed_a, batt_a, price_a = state_a[:4]

        state_b, profit_b = model_run(state_b, q_ch_assumed_a, q_dis_assumed_a, 'B')
        q_ch_assumed_b, q_dis_assumed_b, batt_b, price_b = state_b[:4]

        state_sys = (
            q_ch_assumed_a, q_dis_assumed_a, batt_a, profit_a,
            q_ch_assumed_b, q_dis_assumed_b, batt_b, profit_b,
            price_b
        )
        iter_count += 1

    return (
        q_ch_assumed_a, q_dis_assumed_a, batt_a, profit_a,
        q_ch_assumed_b, q_dis_assumed_b, batt_b, profit_b,
        price_b, iter_count
    )


def nash_eq_b(q_ch_assumed_ini, q_dis_assumed_ini, state_ini):
    """
    Finds the Nash equilibrium iteratively, starting from an initial bid by player B.
    """
    # B plays first assuming A bids 0
    state_b, profit_b = model_run(state_ini, q_ch_assumed_ini, q_dis_assumed_ini, 'B')
    q_ch_assumed_b, q_dis_assumed_b, batt_b, price_b = state_b[:4]

    # A plays based on B’s previous decision
    state_a, profit_a = model_run(state_ini, q_ch_assumed_b, q_dis_assumed_b, 'A')
    q_ch_assumed_a, q_dis_assumed_a, batt_a, price_a = state_a[:4]

    # Dummy old state for convergence check
    old_state_sys = (
        q_ch_assumed_ini, q_dis_assumed_ini, q_dis_assumed_ini, 0,
        q_dis_assumed_ini, q_dis_assumed_ini, q_dis_assumed_ini, 0,
        q_dis_assumed_ini
    )
    state_sys = (
        q_ch_assumed_a, q_dis_assumed_a, batt_a, profit_a,
        q_ch_assumed_b, q_dis_assumed_b, batt_b, profit_b,
        price_b
    )

    iter_count = 0
    while iter_count < 10 and old_state_sys != state_sys:
        old_state_sys = state_sys

        state_b, profit_b = model_run(state_b, q_ch_assumed_a, q_dis_assumed_a, 'B')
        q_ch_assumed_b, q_dis_assumed_b, batt_b, price_b = state_b[:4]

        state_a, profit_a = model_run(state_a, q_ch_assumed_b, q_dis_assumed_b, 'A')
        q_ch_assumed_a, q_dis_assumed_a, batt_a, price_a = state_a[:4]

        state_sys = (
            q_ch_assumed_a, q_dis_assumed_a, batt_a, profit_a,
            q_ch_assumed_b, q_dis_assumed_b, batt_b, profit_b,
            price_b
        )
        iter_count += 1

    return (
        q_ch_assumed_a, q_dis_assumed_a, batt_a, profit_a,
        q_ch_assumed_b, q_dis_assumed_b, batt_b, profit_b,
        price_a, iter_count
    )


def nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, state_ini):
    """
    Finds Nash equilibrium iteratively with both players starting from a 0-action assumption.
    At each iteration, they assume the previous strategy of the opponent.
    """
    # Initial strategy computation
    state_a, profit_a = model_run(state_ini, q_ch_assumed_ini, q_dis_assumed_ini, 'A')
    q_ch_assumed_a, q_dis_assumed_a, batt_a, price_a = state_a[:4]

    state_b, profit_b = model_run(state_ini, q_ch_assumed_ini, q_dis_assumed_ini, 'B')
    q_ch_assumed_b, q_dis_assumed_b, batt_b, price_b = state_b[:4]

    old_state_sys = (
        q_ch_assumed_ini, q_dis_assumed_ini, q_dis_assumed_ini, 0,
        q_dis_assumed_ini, q_dis_assumed_ini, q_dis_assumed_ini, 0,
        q_dis_assumed_ini
    )
    state_sys = (
        q_ch_assumed_a, q_dis_assumed_a, batt_a, profit_a,
        q_ch_assumed_b, q_dis_assumed_b, batt_b, profit_b,
        price_b
    )

    max_change = M
    iter_count = 0
    while iter_count < max_iter and max_change > tol:
        old_state_sys = state_sys

        state_a, profit_a = model_run(state_a, q_ch_assumed_b, q_dis_assumed_b, 'A')
        q_ch_assumed_a, q_dis_assumed_a, batt_a, price_a = state_a[:4]

        state_b, profit_b = model_run(state_b, q_ch_assumed_a, q_dis_assumed_a, 'B')
        q_ch_assumed_b, q_dis_assumed_b, batt_b, price_b = state_b[:4]

        state_sys = (
            q_ch_assumed_a, q_dis_assumed_a, batt_a, profit_a,
            q_ch_assumed_b, q_dis_assumed_b, batt_b, profit_b,
            price_b
        )

        iter_count += 1

        # Calculation of the maximum change
        # max_change = max([
        #     max([abs(x[t]-y[t]) for t in TIME]) if hasattr(x, '__getitem__') else abs(x - y)
        #     for (x,y) in zip(old_state_sys,state_sys)
        #     ])
        
        change = (
            max([abs(state_sys[0][t]-old_state_sys[0][t]) for t in TIME]) / Q_max_a,
            max([abs(state_sys[1][t]-old_state_sys[1][t]) for t in TIME]) / Q_max_a,
            max([abs(state_sys[2][t]-old_state_sys[2][t]) for t in TIME]) / E_max_a,
            max([abs(state_sys[4][t]-old_state_sys[4][t]) for t in TIME]) / Q_max_b,
            max([abs(state_sys[5][t]-old_state_sys[5][t]) for t in TIME]) / Q_max_b,
            max([abs(state_sys[6][t]-old_state_sys[6][t]) for t in TIME]) / E_max_b
        )
        max_change = max(change)

        state_changes.append(max_change)        

    return (
        q_ch_assumed_a, q_dis_assumed_a, batt_a, profit_a,
        q_ch_assumed_b, q_dis_assumed_b, batt_b, profit_b,
        price_b, iter_count
    )


## Looking for Nash equilibirum

# Setting values to initialize the run
q_ch_assumed_ini = [0 for _ in TIME]
q_dis_assumed_ini = [0 for _ in TIME]
state_ini = [
    [0 for _ in range(T)],  # q_ch
    [0 for _ in range(T)],  # q_dis
    [0 for _ in range(T)],  # e (battery level)
    [0 for _ in range(T)],  # price
    {(t, j): 0 for t in range(T) for j in range(N)},  # z_ch
    {(t, j): 0 for t in range(T) for j in range(N)},  # z_dis
    {(t, j): 0 for t in range(T) for j in range(C)},  # u_ch
    {(t, j): 0 for t in range(T) for j in range(D)},  # u_dis
    {(t, j): 0 for t in range(T) for j in range(3)}   # potentially for something else (like u_plus/u_moins?)
]

# Run the Nash equilibrium model
system_state = nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, state_ini)

# Extract components for easier access
q_ch_a = system_state[0]
q_dis_a = system_state[1]
batt_a = system_state[2]
profit_a = system_state[3]

q_ch_b = system_state[4]
q_dis_b = system_state[5]
batt_b = system_state[6]
profit_b = system_state[7]

market_price = system_state[8]
iterations = system_state[9]

proad_a = [q_ch_a[t] - q_dis_a[t] for t in TIME]
proad_b = [q_ch_b[t] - q_dis_b[t] for t in TIME]

q_ch_a_plot = [-q for q in q_ch_a]
q_ch_b_plot = [-q for q in q_ch_b]
R_plot = [-r for r in R]

# Battery profiles with initial level
batt_a_plot = [E_max_a * alpha_batt] + list(batt_a)
batt_b_plot = [E_max_b * alpha_batt] + list(batt_b)
time_with_zero = list(TIME) + [T]

# --- Print results ---
print(f"Profit A: {round(profit_a)}, Profit B: {round(profit_b)}, Iterations to converge: {iterations}") # Profit A: 7170, Profit B: 8580, Iterations to converge: 100
print(max_change)

# Create a single figure with subplots
plt.figure(figsize=(15,8))

# --- Plot 1: System Residual Demand/Production and Power from A & B ---
plt.subplot(2,2,2)

plt.step(np.array(TIME), R_plot, label="Residual Demand (>0) / Production (<0)", where='post', color='red', linestyle='--') 
# When R<0, it corresponds to residual demand
# When R>0, it corresponds to residual production
# With R_plot = -R
plt.bar(np.array(TIME)+0.5, q_ch_a_plot, label="Power from A", color='blue')
plt.bar(np.array(TIME)+0.5, q_ch_b_plot, label="Power from B", bottom=q_ch_a_plot, color='green')
plt.bar(np.array(TIME)+0.5, q_dis_a, color='blue')
plt.bar(np.array(TIME)+0.5, q_dis_b, bottom=q_dis_a, color='green')
plt.xlabel("Time (h)")
plt.ylabel("Power (MW)")
plt.legend(loc="upper left")
plt.title("System Residual Demand/Production and Power from A & B")

# --- Plot 2: Market Price ---
plt.subplot(2,2,1)

plt.step(np.array(TIME), market_price, where='post', color='orange')
plt.xlabel("Time (h)")
plt.ylabel("Market Price (€/MWh)")
plt.title("Market Price")

# --- Plot 3: Battery Levels ---
# Ax 1 for energy storage levels, ax 2 for energy storage discharging/charging power
ax1 = plt.subplot(2,2,3) 

ax1.plot(np.array(time_with_zero), batt_a_plot, label="Battery Level A", color='blue')
ax1.plot(np.array(time_with_zero), batt_b_plot, label="Battery Level B", color='green', linestyle='--')
ax1.set_xlabel("Time (h)")
ax1.set_ylabel("Battery Level (MWh)")
ax1.legend(loc="upper left")
ax1.set_title("Battery Cycle")

ax2 = ax1.twinx()
ax2.step(np.array(time_with_zero), proad_a+[proad_a[-1]], where="post", label="Power from A", color='blue', linestyle='--', linewidth=0.8)
ax2.step(np.array(time_with_zero), proad_b+[proad_b[-1]], where="post", label="Power from B", color='green', linestyle='--', linewidth=0.8)
ax2.set_ylabel('Power [MW]')
ax2.legend(loc='upper right')
ax2.grid()

# --- Plot 4: Cournot iteration convergence ---
ax1 = plt.subplot(2,2,4) 

plt.plot(range(1, len(state_changes) + 1), state_changes, label="Max Change per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Quantity Change (MWh)")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.title("Cournot Iteration Convergence Plot")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


