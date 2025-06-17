#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 03 15:37:33 2025
"""

# Import packages
import random
import gurobipy as gb
from gurobipy import GRB
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt
import seaborn as sb


# Set plot parameters
""" sb.set_style('ticks')
size_pp = 15
font = {'family': 'times new roman',
        'color':  'black',
        'weight': 'normal',
        'size': size_pp,
        } """
linestyles = ["-","--","-.",":"]


# Define problem auxiliary parameters
random.seed(101)
n_load = 10 #number of loads
n_hour_load = 1

# Define auxiliary functions
def generate_demand_function(n, RANGE):
    percentages = [random.random() for _ in range(n)]  # Creates a list of n random float number between 0 and 1
    total = sum(percentages)  # Computes the accumulated percentage sum of list elements
    return {i: p / total for i, p in zip(RANGE, percentages)}  # Normalisation such that the sum amounts to 1

def generate_demand_bids(RANGE, mean=1, std=0.5):
    # Generate n normally distributed integers
    return {i: abs(random.gauss(mean, std)) for i in RANGE}  # Store in a dictionary with indices as keys


# Define ranges
GENERATORS = ['W1','W2','W3','G1','G2'] #range of generators
LOADS = [f'D{i}' for i in range(1,n_load+1)] #range of loads (D1,...)
BESSs = ['B1','B2','B3'] #range of BESSs (B1,...)
TIMES = [t for t in range(1, 25)] # 24 hour time-steps] #range of Time steps between 1,...,24
         
         
# Define input parameters of loads and generators
generator_cost = {'W1':0,'W2':0,'W3':0,'G1':100,'G2':40} # Generators costs
generator_capacity = {'W1':150,'W2':100,'W3':50,'G1':500,'G2':100} # Generators capacity (overline{P}^G_i)
generator_availability = {
    'W1': {1: 0.384460432271812, 2: 0.334138265321714, 3: 0.39211021799672, 4: 0.320718432808069, 5: 0.511097833226585, 6: 0.670195008873801, 7: 0.732582856766447, 8: 0.715879043358315, 9: 0.81648374993383, 10: 0.863173498473644, 11: 0.834677137635638, 12: 0.809602492154417, 13: 0.779704415873997, 14: 0.737250632940909, 15: 0.720228322487059, 16: 0.74521023455785, 17: 0.682319241783506, 18: 0.656484829180529, 19: 0.734256359587207, 20: 0.72407359526009, 21: 0.736487840564778, 22: 0.631564115077659, 23: 0.624393969124891, 24: 0.689311008256275}, 
    'W2': {1: 0.563373256644606, 2: 0.556427015328695, 3: 0.62006417150311, 4: 0.565904840310144, 5: 0.662962577330435, 6: 0.672047757071126, 7: 0.683555715690049, 8: 0.690681448075447, 9: 0.706195409946672, 10: 0.655940860017876, 11: 0.68401154022735, 12: 0.701908509163195, 13: 0.707423438938512, 14: 0.705133604858192, 15: 0.744796435626026, 16: 0.763055085806633, 17: 0.727549023726404, 18: 0.695863325033817, 19: 0.724752795465507, 20: 0.771747268782961, 21: 0.824356631287494, 22: 0.782885432728708, 23: 0.815080918457579, 24: 0.75197694599735}, 
    'W3': {1: 0.362613676602771, 2: 0.527927185676463, 3: 0.53357862582856, 4: 0.619152657560184, 5: 0.68612903725525, 6: 0.70974387706599, 7: 0.720573889475637, 8: 0.710547365158622, 9: 0.714516933056123, 10: 0.68126689566097, 11: 0.707796211142351, 12: 0.671199507398572, 13: 0.633474188784057, 14: 0.68557495287973, 15: 0.691890864192462, 16: 0.743445055398032, 17: 0.695043386042528, 18: 0.657822627338983, 19: 0.772778287545165, 20: 0.802106089261677, 21: 0.835111761001025, 22: 0.825193141814975, 23: 0.821347810888441, 24: 0.772766234701015},
    'G1': {t: 1 for t in TIMES},
    'G2': {t: 1 for t in TIMES}}  # Controllable generators are always available while wind turbines availability is determined by (wind scenario='V1', time step)
load_cost = 2000 # EUR/MWh
load_cost_normalized = generate_demand_bids(LOADS) # Generates bids for every loads (keeping the same bid profile for every hour)
load_bids = {d: load_cost*load_cost_normalized[d] for d in LOADS}
load_capacity = 400 # Total load capacity (overline{P}^D_i)
load_distribution = generate_demand_function(n_load, LOADS)
load_profile_IEEE = {
    1: 1775.835,
    2: 1669.815,
    3: 1590.3,
    4: 1563.795,
    5: 1563.795,
    6: 1590.3,
    7: 1961.37,
    8: 2279.43,
    9: 2517.975,
    10: 2544.48,
    11: 2544.48,
    12: 2517.975,
    13: 2517.975,
    14: 2517.975,
    15: 2464.965,
    16: 2464.965,
    17: 2623.995,
    18: 2650.5,
    19: 2650.5,
    20: 2544.48,
    21: 2411.955,
    22: 2199.915,
    23: 1934.865,
    24: 1669.815} # load profile of IEEE 24-bus system
load_capacity_IEEE = max([load_profile_IEEE[t] for t in TIMES]) # max load of IEEE 24-bus system
load_profile_normalized = {t: load_profile_IEEE[t]/load_capacity_IEEE for t in TIMES} # normalized load profile between (0,1)
load_profile = {(d,t): load_distribution[d]*load_profile_normalized[t]*load_capacity for d in LOADS for t in TIMES}


# Define input parameters of batteries
BESS_cost = {g: 3 for g in BESSs}
BESS_soc_capacity = {'B1':300,'B2':300,'B3':150} # Battery maximum SOC (\overline{SOC}_i)
BESS_soc_init = {'B1':0.5,'B2':0.5,'B3':0.5} # Battery initial SOC (SOC^{init}_i)
BESS_ch_dis_capacity = {'B1':0.5,'B2':0.25,'B3':0.15} # Battery maximum charging/discharging (\overline{P}^{ch/dis}_i)
BESS_ch_eff = {'B1':0.9,'B2':0.9,'B3':0.9} #Battery charging efficiency (\rho^{ch}_i)
BESS_dis_eff = {'B1':1.1,'B2':1.1,'B3':1.1} #Battery discharging efficiency (\rho^{dis}_i)


# Plot market clearing function ----------------------------------------------------------------------------------------
def market_clearing(optimal_BESS_charge, optimal_BESS_discharge, t_clearing=TIMES[0]):
    # Sort generators by increasing cost
    GENERATORS = GENERATORS + BESSs
    generator_cost = generator_cost.update(BESS_cost)
    generator_capacity = generator_capacity.update(optimal_BESS_discharge)
    generator_availability = generator_availability.update({b: {t: 1 for t in TIMES} for b in BESSs})
    generators = sorted(generator_cost.items(), key=lambda x: x[1])  # Sorting by price (ascending)
    gen_quantities = np.cumsum([generator_capacity[g[0]]*generator_availability[g[0]][t_clearing] for g in generators])  # Cumulative supply capacity
    gen_prices = [g[1] for g in generators]  # Corresponding prices

    supply_prices = list(gen_prices) # Make sure it's a list
    supply_quantities = list(gen_quantities) # Make sure it's a list
    supply_prices.insert(0, 0)  # Add 0 price at the beginning of the supply curve
    supply_quantities.insert(0, 0)  # Add 0 quantity at the beginning of the supply curve

    # Sort loads by decreasing bids
    LOADS = LOADS + BESSs
    load_bids = load_bids.update(BESS_cost)
    load_profile_clearing = {d: load_distribution[d]*load_profile_normalized[t_clearing]*load_capacity for d in LOADS}  # Load profile for the corresponding time step
    load_profile_clearing = load_profile_clearing.update(optimal_BESS_charge)
    loads = sorted(load_bids.items(), key=lambda x: x[1], reverse=True)  # Sorting by bids (descending)
    load_quantities = np.cumsum([load_profile_clearing[d[0]] for d in loads])  # Cumulative demand 
    load_prices = [d[1] for d in loads]  # Corresponding bids

    demand_prices = list(load_prices)  # Make sure it's a list
    demand_quantities = list(load_quantities)  # Make sure it's a list
    demand_prices.append(0)  # Add 0 price at the end of the demand curve
    demand_quantities.append(demand_quantities[-1])  # Add 0 quantity at the end of the demand curve

    # Create the plot
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # First subplot (Full range)
    # Plot the demand curve (decreasing)
    axs[0].step(demand_quantities, demand_prices, where='pre', label='Demand Curve (Loads)', color='r', linestyle='-')

    # Plot the supply curve (increasing)
    axs[0].step(supply_quantities, supply_prices, where='pre', label='Supply Curve (Generators)', color='b', linestyle='-')

    # Labels and legend
    axs[0].set_xlabel('Cumulative Quantity (MW)')
    axs[0].set_ylabel('Price (€/MWh)')
    axs[0].legend()
    axs[0].grid()

    # Second subplot (Zoomed-in range)
    axs[1].step(demand_quantities, demand_prices, where='pre', label='Demand Curve (Loads)', color='r', linestyle='-')
    axs[1].step(supply_quantities, supply_prices, where='pre', label='Supply Curve (Generators)', color='b', linestyle='-')

    axs[1].set_xlabel('Cumulative Quantity (MW)')
    axs[1].set_ylabel('Price (€/MWh)')
    axs[1].set_ylim(-10,gen_prices[-1]*2)
    axs[1].legend()
    axs[1].grid()

    fig.suptitle('Market Supply and Demand Curves')

    # Show the plot
    plt.show()


def solve_ED(BESS='NO', show_plots=True): #define function that builds the ED optimization problem taking as argument the parameter BESS = 'NO' by default and 'YES' if we include BESS
    
    # Create a Gurobi model for economic dispatch problem  
    if BESS!='NO':    
        ED_model = gb.Model('Multi-period economic dispatch problem with BESS')
    else:
        ED_model = gb.Model('Multi-period economic dispatch problem without BESS')
    
    
    # Set time limit
    ED_model.Params.LogToConsole = 0
    ED_model.Params.TimeLimit = 100 
    
    
    # Add variables to the Gurobi model
    generator_production = {(g,t):ED_model.addVar(lb=0,ub=generator_capacity[g],name='Electricity production of generator {0} at time {1}'.format(g,t)) for g in GENERATORS for t in TIMES} # electricity production of generators at each time step (p^G_i)
    demand_supplied = {(d,t):ED_model.addVar(lb=0,ub=load_profile[d,t],name='Demand supplied of load {0} at time {1}'.format(d,t)) for d in LOADS for t in TIMES}
    if BESS!='NO':
        BESS_soc = {(g,t):ED_model.addVar(lb=0,ub=BESS_soc_capacity[g],name='State of charge of battery {0} at time {1}'.format(g,t)) for g in BESSs for t in TIMES} # state of charge of batteries at each time step (SOC_i)
        BESS_ch = {(g,t):ED_model.addVar(lb=0,ub=BESS_ch_dis_capacity[g]*BESS_soc_capacity[g],name='Electricity charged by battery {0} at time {1}'.format(g,t)) for g in BESSs for t in TIMES} # Energy charged by batteries at each time step (P^ch_it)
        BESS_dis = {(g,t):ED_model.addVar(lb=0,ub=BESS_ch_dis_capacity[g]*BESS_soc_capacity[g],name='Electricity discharged by battery {0} at time {1}'.format(g,t)) for g in BESSs for t in TIMES} # Energy discharged by batteries at each time step (P^dis_it)
        #BESS_bid = 
    
    # update gurobi model
    ED_model.update()
    
    
    # Set objective function and optimization direction of the Gurobi model
    ED_objective = gb.quicksum(load_bids[d]*demand_supplied[d,t] for d in LOADS for t in TIMES) \
                 - gb.quicksum(BESS_cost[g]*(BESS_ch[g,t]+BESS_dis[g,t]) for g in BESSs for t in TIMES) \
                 - gb.quicksum(generator_cost[g]*generator_production[g,t] for g in GENERATORS for t in TIMES) # demand utility - total electricity production cost over all time steps (z)   
    ED_model.setObjective(ED_objective, gb.GRB.MAXIMIZE) # Max total social welfare (z)
    
    
    # Add constraints to the Gurobi model
    if BESS!='NO': #constraints valid if there are BESSs
        balance_constraint = {t:ED_model.addLConstr(
                    gb.quicksum(generator_production[g,t] for g in GENERATORS) + gb.quicksum(BESS_dis[g,t] for g in BESSs),
                    gb.GRB.EQUAL,
                    gb.quicksum(demand_supplied[d,t] for d in LOADS) + gb.quicksum(BESS_ch[g,t] for g in BESSs),
                    name='Balance equation at time {0}'.format(t)) for t in TIMES} # Balance equation at each time step: Eq. (1b)

        BESS_soc_update_constraint = {(g,t2):ED_model.addLConstr(
                BESS_soc[g,t2],
                gb.GRB.EQUAL,
                BESS_soc[g,t1]+BESS_ch_eff[g]*BESS_ch[g,t2]-BESS_dis_eff[g]*BESS_dis[g,t2],name='Update of BESS {0} state of charge at time {1}'.format(g,t2)) for g in BESSs for (t1,t2) in zip(TIMES[:-1],TIMES[1:])} # Update of BESS state of charge at each time step             
        
        for g in BESSs:
            BESS_soc_update_constraint[g,TIMES[0]] = ED_model.addLConstr(
                    BESS_soc[g,TIMES[0]],
                    gb.GRB.EQUAL,
                    BESS_soc_init[g]*BESS_soc_capacity[g]+BESS_ch_eff[g]*BESS_ch[g,TIMES[0]]-BESS_dis_eff[g]*BESS_dis[g,TIMES[0]],name='Update of BESS {0} state of charge at time T1'.format(g)) # Update of BESS state of charge at initial time step
        
        BESS_soc_final_constraint = {g:ED_model.addLConstr(
                BESS_soc[g,TIMES[-1]],
                gb.GRB.GREATER_EQUAL,
                BESS_soc_init[g]*BESS_soc_capacity[g],name='Final soc of BESS {0}'.format(g)) for g in BESSs} # Lower bound on final soc of BESS 

    else: # constraints valid if there is no BESS
        balance_constraint = {t:ED_model.addLConstr(
                    gb.quicksum(generator_production[g,t] for g in GENERATORS),
                    gb.GRB.EQUAL,
                    gb.quicksum(demand_supplied[d,t] for d in LOADS),
                    name='Balance equation at time {0}'.format(t)) for t in TIMES} # Balance equation at each time step: Eq. (1b)

    # Constraint valid with or without BESS
    production_max_constraint = {(g,t): ED_model.addLConstr(
                        generator_production[g,t],
                        gb.GRB.LESS_EQUAL,
                        generator_capacity[g]*generator_availability[g][t],
                        name='Maximum production constraint of generator {0} at time {1}'.format(g,t))
                        for t in TIMES for g in GENERATORS}

    # Optimize the Gurobi model
    ED_model.optimize()

    if not ED_model.status == gb.GRB.OPTIMAL:
        print("Optimization of {0} was not successful".format(ED_model.ModelName))
        return ED_model


    # Save results at optimality ----------------------------------------------
    else:
        if BESS != 'NO':        
            optimal_objective = round(ED_model.ObjVal, 2) # save objective value of primal optimization problem at optimality (z^*)
            optimal_demand_supplied = np.array([[demand_supplied[l,t].x for t in TIMES] for l in LOADS])
            optimal_generator_production = np.array([[generator_production[g,t].x for t in TIMES] for g in GENERATORS])
            optimal_BESS_charge = np.array([[BESS_ch[b,t].x for t in TIMES] for b in BESSs])
            optimal_BESS_discharge = np.array([[BESS_dis[b,t].x for t in TIMES] for b in BESSs])
            optimal_BESS_soc = np.array([[BESS_soc[b,t].x for t in TIMES] for b in BESSs])
            optimal_uniform_price = np.array([abs(balance_constraint[t].Pi) for t in TIMES]) # save values of uniform prices at optimality
            print("optimization of {0} was successful".format(ED_model.ModelName))
    
        else:
            optimal_objective = round(ED_model.ObjVal, 2) # save objective value of primal optimization problem at optimality (z^*)
            optimal_demand_supplied = np.array([[demand_supplied[l,t].x for t in TIMES] for l in LOADS])
            optimal_generator_production = np.array([[generator_production[g,t].x for t in TIMES] for g in GENERATORS])
            optimal_uniform_price = np.array([abs(balance_constraint[t].Pi) for t in TIMES]) # save values of uniform prices at optimality
            print("optimization of {0} was successful".format(ED_model.ModelName))


    # Print results -----------------------------------------------------------
    print('\n')
    with_without_BESSs = 'with' if BESS != 'NO' else 'without'
    print(f'Social welfare {with_without_BESSs} BESSs:', optimal_objective)


    #####################################################################
    # PLOTS
    #####################################################################

    if not show_plots:
        return ED_model

    plt.figure(figsize=(15,8))
    width = 0.35

    # Plot production / demand -------------------------------------------------------
    plt.subplot(2,2,2)

    total_demand = np.array([sum(load_profile[l,t] for l in LOADS) for t in TIMES])
    repeat_time = np.ravel(np.column_stack((np.array(TIMES)-width, np.array(TIMES)+width)))
    repeat_total_demand = np.repeat(total_demand,2)
    plt.plot(repeat_time,repeat_total_demand,label='Total demand',color='red',linestyle='--')

    plt.bar(np.array(TIMES)-width/2, sum(optimal_generator_production[i] for i in range(len(GENERATORS)) if GENERATORS[i][0]=='W'), width, color='green', label='Wind producers')
    plt.bar(np.array(TIMES)-width/2, sum(optimal_generator_production[i] for i in range(len(GENERATORS)) if GENERATORS[i][0]=='G'), width, bottom=sum(optimal_generator_production[i] for i in range(len(GENERATORS)) if GENERATORS[i][0]=='W'), color='royalblue', label='Controllable generators')
    plt.bar(np.array(TIMES)+width/2, sum(optimal_demand_supplied), width, color='red',label='Supplied demand')
    if BESS != 'NO':
        plt.bar(np.array(TIMES)+width/2, sum(optimal_BESS_charge[i] for i in range(len(BESSs))), width, color='black', bottom=sum(optimal_demand_supplied))
        plt.bar(np.array(TIMES)-width/2, sum(optimal_BESS_discharge[i] for i in range(len(BESSs))), width, color='black', bottom=sum(optimal_generator_production),label='Battery charge/discharge power')

    plt.grid(axis='y')
    plt.xlabel("Time [h]")
    plt.ylabel("Production [MWh]")
    plt.title("Production & Demand")
    plt.legend(loc='lower right')


    # Plot prices --------------------------------------------------------------------
    plt.subplot(2,2,1)
    plt.step(TIMES+[TIMES[-1]+1],np.append(optimal_uniform_price,optimal_uniform_price[-1]), label="Spot price",where='post')
    plt.grid()
    plt.xlabel("Time [h]")
    plt.ylabel("Price [$]")
    plt.title('Spot price')
    plt.legend()


    if BESS != 'NO':
        # Plot battery cycle -------------------------------------------------------------
        ax1=plt.subplot(2,2,3) 
        for b in BESSs:
            ax1.plot(TIMES+[TIMES[-1]+1], np.concatenate([[BESS_soc_init[b]*BESS_soc_capacity[b]],optimal_BESS_soc[BESSs.index(b),:]]), label=f"{b} energy storage")
        ax1.set_ylabel('Energy storage [MWh]')
        ax1.set_xlabel("Time [h]")
        ax1.set_ylim(-10, max(BESS_soc_capacity[b] for b in BESSs)+10)
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        for b in BESSs:
            optimal_BESS_production = optimal_BESS_charge[BESSs.index(b),:] - optimal_BESS_discharge[BESSs.index(b),:]
            ax2.step(list(range(1,26)), np.append(optimal_BESS_production, optimal_BESS_production[-1]), label=f"{b} power", where='post', linestyle='--', linewidth=0.8)
        ax2.set_ylabel('Power [MW]')
        ax2.legend(loc='upper right')
        ax2.grid()
        plt.title("Battery cycle")
        plt.tight_layout()


    plt.show()


solve_ED(BESS='YES', show_plots=True)

# Print market clearing for a characteristic time step
market_clearing(t_clearing=6)