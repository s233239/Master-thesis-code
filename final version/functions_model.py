import numpy as np
import gurobipy as gp                 # Gurobi Python API
from gurobipy import GRB              # Gurobi constants (e.g., GRB.MAXIMIZE)
import matplotlib.pyplot as plt

from functions_policy import apply_policy_to_revenue


# Function containing the optimisation model
def model_run(q_ch_assumed, q_dis_assumed, player, model_parameters, storage_parameters, 
              policy_type, policy_parameters, reserve_policy:bool,
              state_ini=([],[])):

    # Unpack variables
    [max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table_initial] = model_parameters
    [alpha_batt, OC_all, Eta_all, E_max_all, Q_max_all, Q_all] = storage_parameters

    # Policy-related parameters
    if reserve_policy:
        [T_reserve, price_reserve, Q_reserve] = policy_parameters

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

    # Apply reserve policy
    if reserve_policy:
        revenue = {t: 
                   revenue[t] + price_reserve*Q_reserve*Q_max_all[player] if t in T_reserve 
                   else revenue[t] 
                   for t in TIME
                   }
    
    # Apply policy constraints to revenue computations
    if policy_type != "none":
        residual_series = Demand_volume[D-1,:] - RES
        adjust_to_revenue = apply_policy_to_revenue(revenue, q_ch, q_dis, residual_series, policy_type, policy_parameters)

        # Linear objective function
        model.setObjective(gp.quicksum(revenue[t] + adjust_to_revenue[t] for t in TIME), GRB.MAXIMIZE)

    else: 
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


    # Apply reserve policy
    if reserve_policy:
        model.addConstrs(e[t] >= Q_reserve*Q_max_all[player] for t in T_reserve)


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

    price = [sum(residual_demand_price[j,t] * u[t][j] for j in range(D)) for t in TIME]

    revenue = [revenue[t].getValue() for t in TIME]

    CS = [sum((Demand_price[j, t] - Demand_price[j+1, t]) * Demand_volume[j, t] * y[t][j]
                for j in range(D-1)) for t in TIME]    # not necessary to include computation for last satisfied load bc it sets the price hence does not increase CS

    # CS_bis = [sum((Demand_price[j, t] - price[t]) * step_max_additional_demand[j, t] for j in range(D-1)) for t in TIME]

    # print("Check CS:", CS==CS_bis)

    if policy_type == "none":
        adjust_to_revenue = [0.0 for t in TIME]
    else:
        adjust_to_revenue = [adjust_to_revenue[t].getValue() for t in TIME]

    proad = [q_dis[t].getValue() - q_ch[t].getValue() for t in TIME]

    supply_total = [q_dis_assumed[t] + q_dis[t].getValue() for t in TIME]   # positive serie
    demand_total = [q_ch_assumed[t] + q_ch[t].getValue() for t in TIME]     # positive serie
    proad_total = [supply_total[t] - demand_total[t] for t in TIME]         # positive when supply
    q_total = [RES[t] + proad_total[t] for t in TIME]
    unmet_demand = [max(Demand_volume[-1, t] - q_total[t], 0) for t in TIME]
    curtailed_prod = [max(-Demand_volume[-1, t] + q_total[t], 0) for t in TIME]

    PS = np.array([(q_total[t] - curtailed_prod[t]) * price[t] for t in TIME])

    output = [[q_ch[t].getValue() for t in TIME],   # 0
              [q_dis[t].getValue() for t in TIME],  # 1
              [e[t].X for t in TIME],               # 2
              price,                                # 3
              revenue,                              # 4
              adjust_to_revenue,                    # 5
              y,                                    # 6
              CS,                                   # 7
              PS,                                   # 8
              unmet_demand,                         # 9
              curtailed_prod]                       # 10
    

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


def nash_eq(q_ch_assumed_ini, q_dis_assumed_ini, n_players, 
            model_parameters, storage_parameters, 
            policy_type, policy_parameters, reserve_policy, 
            tol=1e-7):

    ne = [[], []]
    state = {}
    output = {}
    u = {}
    profits = {p: [] for p in range(n_players)}

    # Unpack variables
    [max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table_initial] = model_parameters
    [alpha_batt, OC_all, Eta_all, E_max_all, Q_max_all, Q_all] = storage_parameters
    diff_table = diff_table_initial.copy()

    iter = 0
    for player in range(n_players):
        # Initialize optimization model
        state[player], output[player], u[player] = model_run(q_ch_assumed_ini, q_dis_assumed_ini, player, model_parameters, storage_parameters, policy_type, policy_parameters, reserve_policy)

        # Store profits for later plots
        profits[player].append(sum(output[player][4][t] for t in TIME))


    # Store outputs of initialization models
    state_sys = [state[player] for player in range(n_players)]
    ne.append(state_sys.copy())

    if n_players == 1:
        return output, ne, iter, u, profits, diff_table

    # Next iteration
    iter += 1
    are_equal, diff_table = arrays_are_equal(state_sys, ne[-2], n_players, diff_table, tol)

    while not are_equal and iter < max_iter:

        # Profit maximisation for each player
        for player in range(n_players):
            q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in TIME]
            q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in TIME]
            state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player, model_parameters, storage_parameters, policy_type, policy_parameters, reserve_policy)
            
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
                    state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player, model_parameters, storage_parameters, policy_type, policy_parameters, reserve_policy, 
                                                                         state_ini=(np.array(state[player][0]), np.array(state[player][1])))  
                    profits[player].append(sum(output[player][4][t] for t in TIME))      

                for player in range(p+1, n_players):
                    q_ch_assumed = [sum(output[p][0][t] for p in range(n_players) if p != player) for t in TIME]
                    q_dis_assumed = [sum(output[p][1][t] for p in range(n_players) if p != player) for t in TIME]
                    state[player], output[player], u[player] = model_run(q_ch_assumed, q_dis_assumed, player, model_parameters, storage_parameters, policy_type, policy_parameters, reserve_policy)
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
        
    return output, ne, iter, u, profits, diff_table


def market_clearing_no_storage(model_parameters, plots=True):
    """
    Solves a market-clearing optimization without storage using Gurobi.
    """
    # Unpack variables
    [max_iter, TIME, T, D, N, RES, Demand_volume, Demand_price, diff_table_initial] = model_parameters

    results = {
        "quantity": np.zeros(T),
        "price": np.zeros(T),
        "producer_surplus": np.zeros(T),
        "consumer_surplus": np.zeros(T),
        "unmet_demand": np.zeros(T),
        "curtailed_supply": np.zeros(T)
    }

    for t in range(T):
        res_t = RES[t]
        price_t = Demand_price[:, t]
        volume_cum_t = Demand_volume[:, t]
        volume_step = np.diff(np.insert(volume_cum_t, 0, 0.0))

        J = range(len(volume_step))

        model = gp.Model(f"market_clearing_t{t}")
        model.setParam("OutputFlag", 0)

        q_supply = model.addVar(name="q_supply", lb=0)
        q_demand = model.addVars(J, name="q_demand", lb=0)

        model.addConstr(q_supply <= res_t)
        for j in J:
            model.addConstr(q_demand[j] <= volume_step[j])

        model.addConstr(q_supply == gp.quicksum(q_demand[j] for j in J), name="market_balance")

        model.setObjective(gp.quicksum(price_t[j] * q_demand[j] for j in J), GRB.MAXIMIZE)

        model.optimize()

        if model.Status != GRB.OPTIMAL:
            raise ValueError(f"Hour {t}: Optimization did not converge")

        cleared_q = q_supply.X
        opt_q_demand = [q_demand[j].X for j in J]
        clearing_price = -model.getConstrByName("market_balance").Pi
        
        # for j in J:
        #     if opt_q_demand[j] > 0:
        #         if opt_q_demand[j] != volume_step[j]:
        #             print(f"Check hour {t} clearing price:", clearing_price==price_t[j])
        #             break
        #     else:
        #         print(f"Check hour {t} clearing price:", clearing_price==price_t[j])
        #         break
        
        psurplus = clearing_price * cleared_q
        csurplus = sum((price_t[j] - clearing_price) * opt_q_demand[j] for j in J)
        unmet = sum(volume_step) - cleared_q
        curtailed = res_t - cleared_q

        results["quantity"][t] = cleared_q
        results["price"][t] = clearing_price
        results["producer_surplus"][t] = psurplus
        results["consumer_surplus"][t] = csurplus
        results["unmet_demand"][t] = unmet
        results["curtailed_supply"][t] = curtailed

    if plots:
        ## === PLOTTING ===
        plt.figure(figsize=(14,7))
        temps_np = np.array(TIME)
        temps_with_zero_np = np.array([t for t in TIME] + [T])

        # 1. Market Price Plot
        plt.subplot(2,2,1)

        values_to_show = [round(p,2) for p in results["price"] if p > 0]
        values_to_show.sort()
        values_to_show_filtered = [x for i, x in enumerate(values_to_show) if i == 0 or abs(x - values_to_show[i-1]) >= 2]
        index=1
        while len(values_to_show_filtered) > 4:
            values_to_show_filtered.remove(values_to_show_filtered[index])
            index += 1
            if index >= len(values_to_show_filtered):
                index = index // 2

        plt.step(temps_with_zero_np, np.append(results["price"], results["price"][-1]), where='post')
        for p in values_to_show_filtered:
            plt.axhline(y=p, linestyle='--', color='gray', linewidth=1)
            plt.text(x=temps_with_zero_np[-1]+1.5, y=p, s=f'y={round(p)}', color='black', ha='left', va='bottom')
        plt.xlabel("Time (h)")
        plt.ylabel("Market Price (€/MWh)")
        plt.title("Market Price Over Time")
        plt.grid(True)


        # 2. Market Clearing View
        plt.subplot(2,2,2)

        plt.step(temps_with_zero_np, np.append(Demand_volume[-1, :], Demand_volume[-1, -1]), label="Demand", where='post', color='red', linestyle='--', linewidth=1.5) 
        plt.step(temps_with_zero_np, np.append(results["quantity"], results["quantity"][-1]), label="Supplied Energy", where='post', color='red') 
        plt.bar(temps_np+0.5, RES, label="RES Production", color='green')
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
        ax1_heights = [sum(results["unmet_demand"]), sum(results["curtailed_supply"])]
        x1 = np.arange(len(ax1_labels))

        bars1 = ax1.bar(x1, ax1_heights, width=0.5, color='tab:red', label="Energy Metrics")
        ax1.bar_label(bars1, [f"{engineering_notation(x)} MWh" for x in ax1_heights])
        ax1.set_ylabel("Energy (MWh)")
        ax1.set_ylim(0, max(ax1_heights) * 10)
        ax1.set_yscale('symlog', linthresh=1e2)
        ax1.tick_params(axis='y', colors='tab:red')
        ax1.set_title("Market Metrics")
        ax1.set_xticks(x1)
        ax1.set_xticklabels(ax1_labels, rotation=20)

        # === Ax2: Economic metrics ===
        ax2 = plt.subplot(2,2,4)
        ax2_labels = ["Consumer Surplus", "Producer Surplus"]
        CS, PS = results["consumer_surplus"], results["producer_surplus"]
        ax2_heights = [np.average(CS), np.average(PS)]
        x2 = np.arange(len(ax2_labels))

        bars2 = ax2.bar(x2, ax2_heights, width=0.5, color='tab:purple', label="Welfare Metrics")
        ax2.bar_label(bars2, [f"{engineering_notation(x)} €/h" for x in ax2_heights])
        ax2.set_ylabel("Average Amount per Hour (€/h)")
        ax2.set_ylim(0, max(ax2_heights) * 10)
        ax2.set_yscale('symlog', linthresh=10)
        ax2.tick_params(axis='y', colors='tab:purple')
        ax2.set_title("Market Metrics")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(ax2_labels, rotation=20)

        plt.tight_layout()

    return results