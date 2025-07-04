import numpy as np
import gurobipy as gp                 # Gurobi Python API
from gurobipy import GRB              # Gurobi constants (e.g., GRB.MAXIMIZE)


# Function containing the optimisation model
def model_run(q_ch_assumed, q_dis_assumed, player, model_variables, storage_variables, state_ini=([],[])):

    # Unpack variables
    [max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table] = model_variables
    [alpha_batt, OC_all, Eta_all, E_max_all, Q_max_all, Q_all] = storage_variables

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


def arrays_are_equal(a1, a2, n_players, diff_table, tol=1e-7):
    if not a2:
        return False, diff_table 

    diff = 0
    for p in range(n_players):
        a_new, a_old = np.array(a1[p]).flatten(), np.array(a2[p]).flatten()
        diff_p = np.sum(np.abs(a_new - a_old))
        diff = max(diff, diff_p)
    
    # Other method to compute convergence
    # a1, a2 = np.array(a1), np.array(a2)
    # diff = np.linalg.norm((a1 - a2)/a2)

    diff_table.append(diff)
    are_equal = diff < tol

    return are_equal, diff_table


def nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, n_players, model_variables, storage_variables, tol=1e-7):

    ne = [[], []]
    state = {}
    output = {}
    u = {}
    profits = {p: [] for p in range(n_players)}

    # Unpack variables
    [max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table] = model_variables
    [alpha_batt, OC_all, Eta_all, E_max_all, Q_max_all, Q_all] = storage_variables
    
    iter = 0
    for player in range(n_players):
        # Initialize optimization model
        state[player], output[player], u[player] = model_run(q_ch_assumed_ini, q_dis_assumed_ini, player, model_variables, storage_variables)

        # Store profits for later plots
        profits[player].append(sum(output[player][4][t] for t in TIME))


    # Store outputs of initialization models
    state_sys = [state[player] for player in range(n_players)]
    ne.append(state_sys.copy())

    if n_players == 1:
        return output, ne, iter, u, profits

    # Next iteration
    iter += 1
    are_equal, diff_table = arrays_are_equal(state_sys, ne[-2], n_players, diff_table, tol)

    while not are_equal and iter < max_iter:

        # Profit maximisation for each player
        for player in range(n_players):
            q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in TIME]
            q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in TIME]
            state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player, model_variables, storage_variables)
            
            # Store profits for later plots
            profits[player].append(sum(output[player][4][t] for t in TIME))

        state_sys = [state[player] for player in range(n_players)]
        ne.append(state_sys.copy())
        
        # Next iteration
        iter += 1
        are_equal, diff_table = arrays_are_equal(state_sys, ne[-2], n_players, diff_table, tol)

    if iter == max_iter:
        convergence = False
        # Iterate the profit maximisation for all players again WHILE fixing one or more players' decision variables
        for p in range(n_players-1):
            print(f"Convergence has not been reached. Let's try again by fixing player {chr(65 + p)} outputs.")

            for it in range(max_iter//10):

                for player in range(p+1):
                    q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in TIME]
                    q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in TIME]
                    state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player, model_variables, storage_variables, 
                                                                         state_ini=(np.array(state[player][0]), np.array(state[player][1])))  
                    profits[player].append(sum(output[player][4][t] for t in TIME))      

                for player in range(p+1, n_players):
                    q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in TIME]
                    q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in TIME]
                    state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player, model_variables, storage_variables)
                    profits[player].append(sum(output[player][4][t] for t in TIME))

                state_sys = [state[player] for player in range(n_players)]
                ne.append(state_sys.copy())
                
                # Next iteration
                iter += 1
                are_equal, diff_table = arrays_are_equal(state_sys, ne[-2], n_players, diff_table, tol)

                if are_equal:
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
