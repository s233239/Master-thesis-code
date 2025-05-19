"""
- model_run(): optimisation model for the bidding decision of a given player
- nash_eq(): finds Nash equilibrium for 2 players
"""

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
